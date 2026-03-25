"""
Abstract base pipeline for offline RL experiments.

This module provides shared infrastructure for running offline RL pipelines
across different environments. Environment-specific scripts should subclass
`BasePipelineConfig` and implement the required abstract methods.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets import OfflineDataset
from src.fqe import estimate_V_from_Q_on_s0
from src.mopo import MopoDynamicsEnsemble, rollout_in_penalized_mdp, train_mopo_dynamics_ensemble
from src.morel import MorelDynamicsEnsemble, rollout_in_pessimistic_mdp, train_dynamics_ensemble
from src.networks import DynamicsNet, QNet
from src.policies import TorchPolicy


# =============================================================================
# Global State for W&B Metrics
# =============================================================================

_DEFINED_WANDB_METRICS: Set[str] = set()
_DEFINED_WANDB_STEP_KEYS: Set[str] = set()


# =============================================================================
# Utility Functions
# =============================================================================

def default_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to a dictionary suitable for W&B config."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


# =============================================================================
# W&B Logging Utilities
# =============================================================================

def initialize_wandb(args: argparse.Namespace) -> Optional[Any]:
    """Initialize a W&B run if project is specified."""
    if not args.wandb_project:
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=args_to_config(args),
    )
    return run


def wandb_log(run: Optional[Any], payload: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to W&B if a run is active."""
    if run is None:
        return
    run.log(payload, step=step)


def _register_wandb_metric(run: Any, metric_key: str) -> str:
    """Register a custom metric with its step axis in W&B."""
    step_key = f"{metric_key}_step"
    if step_key not in _DEFINED_WANDB_STEP_KEYS:
        run.define_metric(step_key, summary="max")
        _DEFINED_WANDB_STEP_KEYS.add(step_key)
    if metric_key not in _DEFINED_WANDB_METRICS:
        run.define_metric(metric_key, step_metric=step_key)
        _DEFINED_WANDB_METRICS.add(metric_key)
    return step_key


def make_epoch_logger(
    run: Optional[Any],
    metric_key: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Callable[..., None]]:
    """Create a logging hook for training epochs."""
    if run is None:
        return None

    step_key = _register_wandb_metric(run, metric_key)

    def hook(epoch: int, loss: float, val_loss: Optional[float] = None) -> None:
        payload = {metric_key: loss, step_key: epoch + 1}
        if val_loss is not None:
            payload[f"{metric_key}_val"] = val_loss
        if extra:
            payload.update(extra)
        wandb_log(run, payload)

    return hook


# =============================================================================
# Callbacks
# =============================================================================

class FractionCheckpointCallback(BaseCallback):
    """Save policy checkpoints at specified training milestones."""
    
    def __init__(
        self,
        milestones: Sequence[int],
        save_dir: Path,
        prefix: str = "ppo_frac",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.milestones = list(sorted(set(int(m) for m in milestones if m > 0)))
        self.save_dir = save_dir
        self.prefix = prefix
        self.saved: List[Dict[str, int]] = []

    def _on_step(self) -> bool:
        while self.milestones and self.num_timesteps >= self.milestones[0]:
            step = self.milestones.pop(0)
            path = self.save_dir / f"{self.prefix}_{step}"
            self.model.save(path.as_posix())
            self.saved.append({"name": path.name, "path": path.as_posix(), "timesteps": step})
            if self.verbose:
                print(f"[Checkpoint] Saved policy at {step} steps -> {path}")
        return True


class WandbMetricsCallback(BaseCallback):
    """Log PPO training metrics to W&B during rollouts."""
    
    def __init__(self, run: Optional[Any], prefix: str = "ppo"):
        super().__init__(verbose=0)
        self.run = run
        self.prefix = prefix
        self._metrics_defined = False
        self._step_key = f"{prefix}/step"

    def _on_training_start(self) -> None:
        if self.run is None or self._metrics_defined:
            return
        self.run.define_metric(self._step_key)
        self.run.define_metric(f"{self.prefix}/*", step_metric=self._step_key)
        self._metrics_defined = True

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if self.run is None:
            return True
        log_dict = getattr(self.logger, "name_to_value", None)
        if not log_dict:
            return True
        payload = {self._step_key: self.num_timesteps}
        for key, value in log_dict.items():
            if isinstance(value, (int, float)):
                payload[f"{self.prefix}/{key}"] = value
        if payload:
            wandb_log(self.run, payload)
        return True


# =============================================================================
# Policy Adapter
# =============================================================================

class SB3PolicyAdapter:
    """Wrap a Stable-Baselines3 PPO model to conform to TorchPolicy interface."""
    
    def __init__(
        self,
        name: str,
        model: PPO,
        batch_size: int = 2048,
        act_low: float = -1.0,
        act_high: float = 1.0,
    ):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.act_low = act_low
        self.act_high = act_high
        action_space = model.action_space
        self.action_dim = int(np.prod(action_space.shape, dtype=int))

    def sample_torch_actions(
        self,
        states: torch.Tensor,
        repeats: int = 1,
        deterministic: bool = False,
        act_low: Optional[float] = None,
        act_high: Optional[float] = None,
    ) -> torch.Tensor:
        if act_low is None:
            act_low = self.act_low
        if act_high is None:
            act_high = self.act_high
            
        with torch.no_grad():
            base = states
            if repeats > 1:
                base = states.repeat_interleave(repeats, dim=0)
            obs = base.detach().to("cpu").to(torch.float32)
            obs_np = obs.numpy()
            np.nan_to_num(obs_np, copy=False)
            if obs.shape[0] == 0:
                return torch.zeros((0, self.action_dim), device=states.device)
            chunks = max(1, math.ceil(obs_np.shape[0] / self.batch_size))
            actions = []
            for chunk in np.array_split(obs_np, chunks):
                if chunk.size == 0:
                    continue
                act, _ = self.model.predict(chunk, deterministic=deterministic)
                actions.append(np.asarray(act, dtype=np.float32))
            action_np = np.concatenate(actions, axis=0)
            action_tensor = torch.tensor(action_np, device=states.device, dtype=torch.float32)
            low = max(act_low, self.act_low)
            high = min(act_high, self.act_high)
            return action_tensor.clamp_(min=low, max=high)


# =============================================================================
# Model Loading/Saving Utilities
# =============================================================================

def load_policy_models(policy_snapshots: Sequence[Dict[str, object]]) -> Dict[str, PPO]:
    """Load PPO models from saved checkpoints."""
    models: Dict[str, PPO] = {}
    for snap in policy_snapshots:
        models[snap["name"]] = PPO.load(snap["path"])
    return models


def make_policy_adapters(
    policy_models: Dict[str, PPO],
    act_low: float = -1.0,
    act_high: float = 1.0,
) -> Dict[str, TorchPolicy]:
    """Create SB3PolicyAdapter instances for each loaded model."""
    return {
        name: SB3PolicyAdapter(name, model, act_low=act_low, act_high=act_high)
        for name, model in policy_models.items()
    }


def save_snapshots(manifest_path: Path, snapshots: List[Dict[str, object]]) -> None:
    """Save policy snapshot manifest to JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, indent=2)


def load_snapshots(manifest_path: Path) -> List[Dict[str, object]]:
    """Load policy snapshot manifest from JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_q_models(q_models: Dict[str, torch.nn.Module], directory: Path) -> Dict[str, str]:
    """Save Q-networks and create a manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, model in q_models.items():
        path = directory / f"{name}.pt"
        torch.save(copy.deepcopy(model).cpu(), path)
        saved[name] = path.as_posix()
    manifest = directory / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    return saved


def load_q_models(
    directory: Path,
    device: torch.device,
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, str]]:
    """Load Q-networks from a manifest."""
    manifest = directory / "manifest.json"
    with open(manifest, "r", encoding="utf-8") as f:
        saved = json.load(f)
    models: Dict[str, torch.nn.Module] = {}
    for name, path_str in saved.items():
        model = torch.load(path_str, map_location=device, weights_only=False)
        model.to(device)
        models[name] = model
    return models, saved


def save_dynamics_models(
    sup_model: Optional[DynamicsNet],
    ranking_new_models: Dict[str, DynamicsNet],
    directory: Path,
    value_aware_model: Optional[DynamicsNet] = None,
    morel_model: Optional[MorelDynamicsEnsemble] = None,
    morel_info: Optional[Dict[str, Any]] = None,
    mopo_model: Optional[MopoDynamicsEnsemble] = None,
    mopo_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """Save dynamics models and create a manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, object] = {
        "supervised": None,
        "ranking": (directory / "dynamics_ranking.pt").as_posix(),
        "q_aware": {},
        "ranking_new": {},
        "value_aware": None,
        "morel": None,
        "morel_info": None,
        "mopo": None,
        "mopo_info": None,
    }
    if sup_model is not None:
        paths["supervised"] = (directory / "dynamics_supervised.pt").as_posix()
        torch.save(copy.deepcopy(sup_model).cpu(), paths["supervised"])

    for loss_name, model in ranking_new_models.items():
        path = directory / f"dynamics_ranking_new_{loss_name}.pt"
        torch.save(copy.deepcopy(model).cpu(), path)
        paths["ranking_new"][loss_name] = path.as_posix()

    if value_aware_model is not None:
        paths["value_aware"] = (directory / "dynamics_value_aware.pt").as_posix()
        torch.save(copy.deepcopy(value_aware_model).cpu(), paths["value_aware"])

    if morel_model is not None:
        paths["morel"] = (directory / "dynamics_morel.pt").as_posix()
        torch.save(copy.deepcopy(morel_model).cpu(), paths["morel"])
    if morel_info is not None:
        paths["morel_info"] = morel_info

    if mopo_model is not None:
        paths["mopo"] = (directory / "dynamics_mopo.pt").as_posix()
        torch.save(copy.deepcopy(mopo_model).cpu(), paths["mopo"])
    if mopo_info is not None:
        paths["mopo_info"] = mopo_info

    manifest = directory / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    return paths


def load_dynamics_models(
    directory: Path,
    device: torch.device,
) -> Tuple[Optional[DynamicsNet], Dict[str, DynamicsNet], Dict[str, object], Optional[DynamicsNet]]:
    """Load dynamics models from a manifest.
    
    Returns:
        Tuple of (supervised_model, ranking_new_models, paths, value_aware_model)
    """
    manifest = directory / "manifest.json"
    with open(manifest, "r", encoding="utf-8") as f:
        paths = json.load(f)

    sup_model: Optional[DynamicsNet] = None
    if paths.get("supervised") is not None:
        sup_model = torch.load(paths["supervised"], map_location=device, weights_only=False)
        sup_model.to(device)

    ranking_new_models: Dict[str, DynamicsNet] = {}
    for loss_name, path in paths.get("ranking_new", {}).items():
        model: DynamicsNet = torch.load(path, map_location=device, weights_only=False)
        model.to(device)
        ranking_new_models[loss_name] = model

    value_aware_model: Optional[DynamicsNet] = None
    if paths.get("value_aware") is not None:
        value_aware_model = torch.load(paths["value_aware"], map_location=device, weights_only=False)
        value_aware_model.to(device)

    return sup_model, ranking_new_models, paths, value_aware_model


def load_morel_model_from_paths(
    paths: Mapping[str, object],
    device: torch.device,
) -> Optional[MorelDynamicsEnsemble]:
    """Load a saved MOReL ensemble from manifest paths (if present)."""
    path_obj = paths.get("morel")
    if path_obj is None:
        return None
    model = torch.load(str(path_obj), map_location=device, weights_only=False)
    if hasattr(model, "to"):
        model.to(device)
    return model


def load_mopo_model_from_paths(
    paths: Mapping[str, object],
    device: torch.device,
) -> Optional[MopoDynamicsEnsemble]:
    """Load a saved MOPO ensemble from manifest paths (if present)."""
    path_obj = paths.get("mopo")
    if path_obj is None:
        return None
    model = torch.load(str(path_obj), map_location=device, weights_only=False)
    if hasattr(model, "to"):
        model.to(device)
    return model


def _safe_json_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def compact_morel_info(raw_info: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Convert MOReL training info to a compact JSON-serializable dict."""
    if raw_info is None:
        return None

    info: Dict[str, Any] = {}
    scalar_fields = (
        "disagreement_mean",
        "disagreement_std",
        "disagreement_max",
        "threshold",
        "beta_effective",
        "halt_reward",
        "reward_min",
        "reward_offset",
    )
    for field in scalar_fields:
        if hasattr(raw_info, field):
            info[field] = _safe_json_number(getattr(raw_info, field))

    if hasattr(raw_info, "model_losses"):
        losses = getattr(raw_info, "model_losses")
        final_losses: List[Optional[float]] = []
        epochs_per_member: List[int] = []
        for member_losses in losses:
            if member_losses:
                final_losses.append(_safe_json_number(float(member_losses[-1])))
                epochs_per_member.append(int(len(member_losses)))
            else:
                final_losses.append(None)
                epochs_per_member.append(0)
        info["member_final_losses"] = final_losses
        info["member_epochs"] = epochs_per_member

    return info


def compact_mopo_info(raw_info: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Convert MOPO training info to a compact JSON-serializable dict."""
    if raw_info is None:
        return None

    info: Dict[str, Any] = {}
    scalar_fields = (
        "uncertainty_mean",
        "uncertainty_std",
        "uncertainty_max",
        "penalty_coef",
        "holdout_size",
        "ensemble_size",
        "elite_size",
        "hidden_dim",
        "hidden_layers",
        "epochs",
        "batch_size",
        "lr",
    )
    for field in scalar_fields:
        if hasattr(raw_info, field):
            value = getattr(raw_info, field)
            if isinstance(value, bool):
                info[field] = bool(value)
            elif isinstance(value, (int, np.integer)):
                info[field] = int(value)
            else:
                info[field] = _safe_json_number(float(value))

    if hasattr(raw_info, "bootstrap"):
        info["bootstrap"] = bool(getattr(raw_info, "bootstrap"))

    if hasattr(raw_info, "elite_indices"):
        elite = getattr(raw_info, "elite_indices")
        info["elite_indices"] = [int(v) for v in elite]

    if hasattr(raw_info, "member_holdout_nll"):
        holdout_losses = getattr(raw_info, "member_holdout_nll")
        info["member_holdout_nll"] = [_safe_json_number(float(v)) for v in holdout_losses]

    if hasattr(raw_info, "model_losses"):
        losses = getattr(raw_info, "model_losses")
        final_losses: List[Optional[float]] = []
        epochs_per_member: List[int] = []
        for member_losses in losses:
            if member_losses:
                final_losses.append(_safe_json_number(float(member_losses[-1])))
                epochs_per_member.append(int(len(member_losses)))
            else:
                final_losses.append(None)
                epochs_per_member.append(0)
        info["member_final_losses"] = final_losses
        info["member_epochs"] = epochs_per_member

    return info


def find_missing_requested_dynamics_models(
    requested_models: Sequence[str],
    supervised_model: Optional[DynamicsNet],
    ranking_new_models: Mapping[str, DynamicsNet],
    value_aware_model: Optional[DynamicsNet],
    morel_model: Optional[MorelDynamicsEnsemble],
    mopo_model: Optional[MopoDynamicsEnsemble],
    value_aware_only: bool = False,
) -> List[str]:
    """Return requested dynamics model names that are missing from loaded artifacts."""
    if value_aware_only:
        return [] if value_aware_model is not None else ["value_aware"]

    requested_set = set(requested_models)
    missing: List[str] = []
    if "supervised" in requested_set and supervised_model is None:
        missing.append("supervised")
    for loss_name in ("kendall", "hinge", "listnet"):
        if loss_name in requested_set and loss_name not in ranking_new_models:
            missing.append(loss_name)
    if "morel" in requested_set and morel_model is None:
        missing.append("morel")
    if "mopo" in requested_set and mopo_model is None:
        missing.append("mopo")
    return missing


# =============================================================================
# PPO Training
# =============================================================================

def train_ppo_with_checkpoints(
    env_id: str,
    total_steps: int,
    save_dir: Path,
    fractions: Sequence[float],
    seed: int,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    device: str,
    wandb_run: Optional[Any] = None,
    policy_type: str = "auto",
    env_factory: Optional[Callable[[], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, object]]:
    """Train PPO and save checkpoints at specified fractions of training.
    
    Args:
        env_id: Environment ID (used if env_factory is None)
        env_factory: Optional callable that creates an environment instance.
                    If provided, this is used instead of env_id.
        env_kwargs: Optional keyword arguments to pass to gym.make().
    """
    set_random_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    save_dir.mkdir(parents=True, exist_ok=True)

    if env_kwargs is None:
        env_kwargs = {}

    if env_factory is not None:
        vec_env = make_vec_env(env_factory, n_envs=n_envs, seed=seed)
    else:
        vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    vec_env = VecMonitor(vec_env)

    # Auto-detect policy type
    if policy_type == "auto":
        probe_env = gym.make(env_id, **env_kwargs)
        obs_space = probe_env.observation_space
        policy = "MlpPolicy"
        if hasattr(obs_space, "shape") and len(obs_space.shape or []) == 3:
            policy = "CnnPolicy"
        probe_env.close()
    else:
        policy = policy_type

    model = PPO(
        policy=policy,
        env=vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        device=device,
        verbose=0,
    )

    snapshots: List[Dict[str, object]] = []
    init_path = save_dir / "ppo_frac_0"
    model.save(init_path.as_posix())
    snapshots.append({"name": init_path.name, "path": init_path.as_posix(), "timesteps": 0})

    milestone_steps = [int(total_steps * f) for f in fractions if f > 0]
    checkpoint_cb = FractionCheckpointCallback(milestone_steps, save_dir, prefix="ppo_frac", verbose=1)
    callbacks: List[BaseCallback] = [checkpoint_cb, WandbMetricsCallback(wandb_run)]
    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)

    vec_env.close()
    snapshots.extend(checkpoint_cb.saved)
    return snapshots


# =============================================================================
# Dataset Collection and Processing
# =============================================================================

def rollout_policy(
    model: PPO,
    env_id: str,
    total_steps: int,
    seed: int,
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]]:
    """Collect transitions by rolling out a policy in an environment."""
    if env_kwargs is None:
        env_kwargs = {}
    env = gym.make(env_id, **env_kwargs)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    initial_states: List[np.ndarray] = [obs.copy()]
    transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
    steps = 0
    while steps < total_steps:
        action, _ = model.predict(obs, deterministic=False)
        action = np.asarray(action, dtype=np.float32)
        next_obs, _, terminated, truncated, _ = env.step(action)
        reward = reward_fn(obs, action)
        done = float(terminated or truncated)
        transitions.append((obs.copy(), action.copy(), float(reward), next_obs.copy(), done))
        obs = next_obs
        steps += 1
        if done:
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            initial_states.append(obs.copy())
    env.close()
    return initial_states, transitions


def build_offline_dataset(
    policy_snapshots: Sequence[Dict[str, object]],
    policy_models: Dict[str, PPO],
    env_id: str,
    steps_per_policy: int,
    seed: int,
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> OfflineDataset:
    """Build an offline dataset by rolling out multiple policies."""
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_next_states: List[np.ndarray] = []
    all_dones: List[float] = []
    initial_states: List[np.ndarray] = []

    for idx, snap in enumerate(policy_snapshots):
        model = policy_models[snap["name"]]
        init_states, transitions = rollout_policy(
            model, env_id, steps_per_policy, seed + idx, reward_fn, env_kwargs
        )
        initial_states.extend(init_states)
        for s, a, r, sn, d in transitions:
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)
            all_next_states.append(sn)
            all_dones.append(d)

    return OfflineDataset(
        states=np.asarray(all_states, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
        rewards=np.asarray(all_rewards, dtype=np.float32),
        next_states=np.asarray(all_next_states, dtype=np.float32),
        dones=np.asarray(all_dones, dtype=np.float32),
        initial_states=np.asarray(initial_states, dtype=np.float32),
    )


def load_offline_dataset(npz_path: Path) -> OfflineDataset:
    """Load an offline dataset from a .npz file."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Offline dataset not found at {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return OfflineDataset(
            states=data["s"].astype(np.float32, copy=False),
            actions=data["a"].astype(np.float32, copy=False),
            rewards=data["r"].astype(np.float32, copy=False),
            next_states=data["s_next"].astype(np.float32, copy=False),
            dones=data["done"].astype(np.float32, copy=False),
            initial_states=data["s0"].astype(np.float32, copy=False),
            mask=data["mask"].astype(np.float32, copy=False) if "mask" in data else None,
        )


def make_sequence_dataset(
    dataset: OfflineDataset,
    seq_len: int,
    overlap: int = 0,
) -> OfflineDataset:
    """Convert flat transitions into fixed-length episode-respecting sequences."""
    if seq_len <= 1:
        return dataset

    stride = max(1, seq_len - max(0, overlap))

    s_all = dataset.states
    a_all = dataset.actions
    r_all = dataset.rewards
    sn_all = dataset.next_states
    d_all = dataset.dones

    episode_starts = [0]
    for i, done in enumerate(d_all):
        if bool(done):
            episode_starts.append(i + 1)
    if episode_starts[-1] != len(d_all):
        episode_starts.append(len(d_all))

    seq_s, seq_a, seq_r, seq_sn, seq_d, seq_mask = [], [], [], [], [], []

    for start_idx, end_idx in zip(episode_starts[:-1], episode_starts[1:]):
        ep_s = s_all[start_idx:end_idx]
        ep_a = a_all[start_idx:end_idx]
        ep_r = r_all[start_idx:end_idx]
        ep_sn = sn_all[start_idx:end_idx]
        ep_d = d_all[start_idx:end_idx]

        ep_len = ep_s.shape[0]
        if ep_len == 0:
            continue

        for window_start in range(0, ep_len, stride):
            window_end = window_start + seq_len
            s_slice = ep_s[window_start:window_end]
            a_slice = ep_a[window_start:window_end]
            r_slice = ep_r[window_start:window_end]
            sn_slice = ep_sn[window_start:window_end]
            d_slice = ep_d[window_start:window_end]

            cur_len = s_slice.shape[0]
            if cur_len < seq_len:
                pad_len = seq_len - cur_len
                s_slice = np.pad(s_slice, ((0, pad_len), (0, 0)), mode="constant")
                a_slice = np.pad(a_slice, ((0, pad_len), (0, 0)), mode="constant")
                r_slice = np.pad(r_slice, ((0, pad_len)), mode="constant")
                sn_slice = np.pad(sn_slice, ((0, pad_len), (0, 0)), mode="constant")
                d_slice = np.pad(d_slice, ((0, pad_len)), mode="constant")
                mask_slice = np.concatenate([
                    np.ones(cur_len, dtype=np.float32),
                    np.zeros(pad_len, dtype=np.float32),
                ])
            else:
                mask_slice = np.ones(seq_len, dtype=np.float32)

            seq_s.append(s_slice)
            seq_a.append(a_slice)
            seq_r.append(r_slice)
            seq_sn.append(sn_slice)
            seq_d.append(d_slice)
            seq_mask.append(mask_slice)

    if not seq_s:
        raise ValueError(f"Not enough transitions to form sequences of length {seq_len}.")

    return OfflineDataset(
        states=np.stack(seq_s, axis=0).astype(np.float32, copy=False),
        actions=np.stack(seq_a, axis=0).astype(np.float32, copy=False),
        rewards=np.stack(seq_r, axis=0).astype(np.float32, copy=False),
        next_states=np.stack(seq_sn, axis=0).astype(np.float32, copy=False),
        dones=np.stack(seq_d, axis=0).astype(np.float32, copy=False),
        initial_states=dataset.initial_states,
        mask=np.stack(seq_mask, axis=0).astype(np.float32, copy=False),
    )


# =============================================================================
# Q-Network Training
# =============================================================================

def train_q_networks(
    dataset: OfflineDataset,
    policies: Dict[str, TorchPolicy],
    device: torch.device,
    gamma: float,
    epochs: int,
    batch_size: int,
    lr: float,
    samples: int,
    wandb_run: Optional[Any] = None,
) -> Dict[str, QNet]:
    """Train Q-networks for each policy using FQE."""
    q_models: Dict[str, QNet] = {}
    state_dim = dataset.states.shape[1]
    act_dim = dataset.actions.shape[1]
    for name, policy in policies.items():
        q_net = QNet(state_dim=state_dim, act_dim=act_dim).to(device)
        rescaled_q = q_net.train(
            dataset,
            target_policy=policy,
            gamma=gamma,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            samples=samples,
            device=device,
            use_amp=device.type == "cuda",
            log_hook=make_epoch_logger(wandb_run, f"q_train/{name}"),
        )
        q_models[name] = rescaled_q
    return q_models


# =============================================================================
# Dynamics Model Training
# =============================================================================

def train_dynamics_models(
    dataset: OfflineDataset,
    policies: Dict[str, TorchPolicy],
    q_models: Dict[str, QNet],
    device: torch.device,
    dyn_epochs: int,
    dyn_batch: int,
    dyn_lr: float,
    gamma: float,
    lambda_td: float,
    lambda_rank: float,
    rank_temperature: float,
    rank_rollout_horizon: int,
    rank_rollout_episodes: int,
    val_fraction: float,
    early_stop_patience: int,
    min_epochs: int,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dynamics_loss: str = "nll",
    hidden_dim: int = 256,
    backbone: str = "mlp",
    wandb_run: Optional[Any] = None,
    state_low: Optional[torch.Tensor] = None,
    state_upper: Optional[torch.Tensor] = None,
    wrapped_dims: Optional[List[int]] = None,
    dynamics_models: Optional[List[str]] = None,
    value_aware_only: bool = False,
    act_low: float = -1.0,
    act_high: float = 1.0,
    seed: int = 0,
    env_name: Optional[str] = None,
    morel_ensemble_size: int = 4,
    morel_hidden_dim: Optional[int] = None,
    morel_epochs: Optional[int] = None,
    morel_batch_size: int = 256,
    morel_lr: float = 5e-4,
    morel_threshold: Optional[float] = None,
    morel_threshold_mode: str = "mean_std",
    morel_threshold_beta: float = 5.0,
    morel_threshold_frac_of_max: float = 1.0,
    morel_reward_offset: Optional[float] = None,
    morel_halt_reward: Optional[float] = None,
    morel_bootstrap: bool = True,
    mopo_ensemble_size: int = 7,
    mopo_elite_size: int = 5,
    mopo_hidden_dim: int = 200,
    mopo_hidden_layers: int = 4,
    mopo_epochs: int = 300,
    mopo_batch_size: int = 256,
    mopo_lr: float = 1e-3,
    mopo_holdout_size: int = 1000,
    mopo_penalty_coef: float = 1.0,
    mopo_bootstrap: bool = True,
) -> Tuple[
    Optional[DynamicsNet],
    Dict[str, DynamicsNet],
    Optional[DynamicsNet],
    Optional[MorelDynamicsEnsemble],
    Optional[Dict[str, Any]],
    Optional[MopoDynamicsEnsemble],
    Optional[Dict[str, Any]],
]:
    """Train supervised and ranking-aware dynamics models.
    
    Args:
        dynamics_models: List of model types to train. Options: 'supervised', 'kendall', 'hinge', 'listnet', 'morel', 'mopo'.
                        If None, trains all models.
        value_aware_only: If True, only train the value-aware model (ignore dynamics_models).
        act_low: Lower bound for actions (used by value-aware model).
        act_high: Upper bound for actions (used by value-aware model).
    
    Returns:
        Tuple of (supervised_model, ranking_new_models, value_aware_model, morel_model, morel_info, mopo_model, mopo_info)
    """
    if dynamics_models is None:
        dynamics_models = ["supervised", "kendall", "hinge", "listnet"]
    state_dim = dataset.states.shape[-1]
    act_dim = dataset.actions.shape[-1]

    # If value_aware_only, skip supervised and ranking models entirely
    policy_q_pairs = [(policies[name], q_models[name]) for name in policies]
    if value_aware_only:
        value_aware_model = DynamicsNet(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden=hidden_dim,
            backbone=backbone,
            state_low=state_low,
            state_upper=state_upper,
            wrapped_dims=wrapped_dims if wrapped_dims is not None else [],
        ).to(device)
        value_aware_model.train_value_aware_model(
            dataset,
            policy_q_pairs=policy_q_pairs,
            epochs=dyn_epochs,
            batch_size=dyn_batch,
            lr=dyn_lr,
            device=device,
            log_hook=make_epoch_logger(wandb_run, "dynamics/value_aware"),
            val_fraction=val_fraction,
            early_stop_patience=early_stop_patience,
            min_epochs=min_epochs,
            act_low=act_low,
            act_high=act_high,
        )
        return None, {}, value_aware_model, None, None, None, None

    # Supervised model
    sup_model: Optional[DynamicsNet] = None
    if "supervised" in dynamics_models:
        sup_model = DynamicsNet(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden=hidden_dim,
            backbone=backbone,
            state_low=state_low,
            state_upper=state_upper,
            wrapped_dims=wrapped_dims if wrapped_dims is not None else [],
        ).to(device)
        sup_model.train(
            dataset,
            epochs=dyn_epochs,
            batch_size=dyn_batch,
            lr=dyn_lr,
            device=device,
            log_hook=make_epoch_logger(wandb_run, "dynamics/supervised"),
            val_fraction=val_fraction,
            early_stop_patience=early_stop_patience,
            min_epochs=min_epochs,
            dynamics_loss=dynamics_loss,
        )

    # Ranking-aware models
    ranking_new_models: Dict[str, DynamicsNet] = {}
    value_aware_model: Optional[DynamicsNet] = None
    morel_model: Optional[MorelDynamicsEnsemble] = None
    morel_info: Optional[Dict[str, Any]] = None
    mopo_model: Optional[MopoDynamicsEnsemble] = None
    mopo_info: Optional[Dict[str, Any]] = None

    ranking_losses_to_train = [loss for loss in ("kendall", "hinge", "listnet") if loss in dynamics_models]
    for loss_name in ranking_losses_to_train:
        model = DynamicsNet(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden=hidden_dim,
            backbone=backbone,
            state_low=state_low,
            state_upper=state_upper,
            wrapped_dims=wrapped_dims if wrapped_dims is not None else [],
        ).to(device)
        model.train_ranking_aware_model(
            dataset,
            policy_q_pairs=policy_q_pairs,
            gamma=gamma,
            lambda_rank=lambda_rank,
            rollout_horizon=rank_rollout_horizon,
            rollout_episodes=rank_rollout_episodes,
            epochs=dyn_epochs,
            batch_size=dyn_batch,
            lr=dyn_lr,
            reward_fn=reward_fn_torch,
            device=device,
            log_hook=make_epoch_logger(wandb_run, f"dynamics/ranking_new/{loss_name}"),
            val_fraction=val_fraction,
            early_stop_patience=early_stop_patience,
            min_epochs=min_epochs,
            ranking_loss_type=loss_name,
            dynamics_loss=dynamics_loss,
            rank_temperature=rank_temperature,
        )
        ranking_new_models[loss_name] = model

    if "morel" in dynamics_models:
        morel_log_hook: Optional[Callable[[int, int, float], None]] = None
        if wandb_run is not None:
            def _hook(member_idx: int, epoch: int, loss: float) -> None:
                step_key = f"dynamics/morel/member_{member_idx}/loss_step"
                wandb_log(
                    wandb_run,
                    {
                        f"dynamics/morel/member_{member_idx}/loss": float(loss),
                        step_key: int(epoch + 1),
                    },
                )
            morel_log_hook = _hook

        morel_model, raw_morel_info = train_dynamics_ensemble(
            dataset=dataset,
            ensemble_size=int(max(1, morel_ensemble_size)),
            hidden_dim=morel_hidden_dim,
            epochs=morel_epochs,
            batch_size=morel_batch_size,
            lr=morel_lr,
            seed=seed,
            env_name=env_name,
            reward_offset=morel_reward_offset,
            halt_reward=morel_halt_reward,
            threshold=morel_threshold,
            threshold_beta=morel_threshold_beta,
            threshold_mode=morel_threshold_mode,
            threshold_frac_of_max=morel_threshold_frac_of_max,
            state_low=state_low,
            state_high=state_upper,
            bootstrap=morel_bootstrap,
            device=device,
            log_hook=morel_log_hook,
        )
        morel_info = compact_morel_info(raw_morel_info)

    if "mopo" in dynamics_models:
        mopo_log_hook: Optional[Callable[[int, int, float], None]] = None
        if wandb_run is not None:
            def _hook(member_idx: int, epoch: int, loss: float) -> None:
                step_key = f"dynamics/mopo/member_{member_idx}/loss_step"
                wandb_log(
                    wandb_run,
                    {
                        f"dynamics/mopo/member_{member_idx}/loss": float(loss),
                        step_key: int(epoch + 1),
                    },
                )
            mopo_log_hook = _hook

        mopo_model_raw, raw_mopo_info = train_mopo_dynamics_ensemble(
            dataset=dataset,
            ensemble_size=int(max(1, mopo_ensemble_size)),
            elite_size=int(max(1, mopo_elite_size)),
            hidden_dim=int(max(1, mopo_hidden_dim)),
            hidden_layers=int(max(1, mopo_hidden_layers)),
            epochs=int(max(1, mopo_epochs)),
            batch_size=int(max(1, mopo_batch_size)),
            lr=float(mopo_lr),
            holdout_size=int(max(1, mopo_holdout_size)),
            penalty_coef=float(mopo_penalty_coef),
            seed=seed,
            env_name=env_name,
            state_low=state_low,
            state_high=state_upper,
            bootstrap=mopo_bootstrap,
            device=device,
            log_hook=mopo_log_hook,
        )
        mopo_model = mopo_model_raw
        mopo_info = compact_mopo_info(raw_mopo_info)

    return sup_model, ranking_new_models, value_aware_model, morel_model, morel_info, mopo_model, mopo_info


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_q_estimate(
    q_model: QNet,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    samples: int,
) -> float:
    """Estimate V(s0) using the Q-network."""
    return float(
        estimate_V_from_Q_on_s0(
            q_model,
            initial_states,
            policy,
            K=samples,
        )
    )


def evaluate_in_dynamics(
    dynamics: DynamicsNet,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    horizon: int,
    gamma: float,
    device: torch.device,
    rollouts: int,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    termination_fn_torch: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> float:
    """Evaluate a policy by rolling out in a learned dynamics model.
    
    Args:
        dynamics: Learned dynamics model
        policy: Policy to evaluate
        initial_states: Array of initial states to start rollouts from
        horizon: Maximum number of steps per rollout
        gamma: Discount factor
        device: PyTorch device
        rollouts: Number of rollouts to average over
        reward_fn_torch: Reward function (states, actions) -> rewards
        termination_fn_torch: Optional termination function (states) -> bool tensor.
            If provided, rewards after termination are zeroed out.
    """
    if initial_states.shape[0] == 0:
        raise ValueError("No initial states available for dynamics evaluation.")
    idx = np.random.choice(
        initial_states.shape[0],
        size=rollouts,
        replace=initial_states.shape[0] < rollouts,
    )
    states = torch.tensor(initial_states[idx], dtype=torch.float32, device=device)
    total = torch.zeros(states.size(0), device=device)
    discount = torch.ones_like(total)
    alive = torch.ones_like(total, dtype=torch.bool)  # Track which rollouts are still alive
    
    for _ in range(horizon):
        actions = policy.sample_torch_actions(states, deterministic=True)
        rewards = reward_fn_torch(states, actions)
        # Only accumulate rewards for alive rollouts
        total += discount * rewards * alive.float()
        discount = discount * gamma
        states = dynamics.sample_next(states, actions, deterministic=True)
        
        # Check termination if function provided
        if termination_fn_torch is not None:
            terminated = termination_fn_torch(states)
            alive = alive & ~terminated
            # Exit early if all rollouts have terminated
            if not alive.any():
                break
    
    return float(total.mean().item())


def evaluate_in_morel_pessimistic_mdp(
    morel_model: MorelDynamicsEnsemble,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    act_low: float = -1.0,
    act_high: float = 1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate a fixed policy in the MOReL pessimistic MDP via MC rollouts."""
    return rollout_in_pessimistic_mdp(
        ensemble=morel_model,
        policy=policy,
        initial_states=initial_states,
        reward_fn_torch=reward_fn_torch,
        horizon=horizon,
        gamma=gamma,
        rollouts=rollouts,
        seed=seed,
        act_low=act_low,
        act_high=act_high,
        deterministic_policy=True,
        deterministic_dynamics=True,
    )


def evaluate_in_mopo_penalized_mdp(
    mopo_model: MopoDynamicsEnsemble,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
    act_low: float = -1.0,
    act_high: float = 1.0,
    deterministic_policy: bool = True,
    deterministic_dynamics: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    """Evaluate a fixed policy in the MOPO uncertainty-penalized model MDP."""
    return rollout_in_penalized_mdp(
        ensemble=mopo_model,
        policy=policy,
        initial_states=initial_states,
        horizon=horizon,
        gamma=gamma,
        rollouts=rollouts,
        seed=seed,
        act_low=act_low,
        act_high=act_high,
        deterministic_policy=deterministic_policy,
        deterministic_dynamics=deterministic_dynamics,
    )


def evaluate_sb3_policy_fixed_horizon(
    model: PPO,
    env_id: str,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    respect_termination: bool = True,
    termination_fn: Optional[Callable[[np.ndarray], bool]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """Roll out PPO in the true env for a fixed horizon.
    
    Args:
        model: PPO model to evaluate
        env_id: Gymnasium environment ID
        horizon: Maximum number of steps per rollout
        gamma: Discount factor
        rollouts: Number of rollouts to average over
        seed: Random seed
        reward_fn: Reward function (state, action) -> float
        respect_termination: If True, stop accumulating rewards after termination.
            If False, reset and continue (legacy behavior).
        termination_fn: Optional user-defined termination function (state) -> bool.
            If provided, this is used instead of the environment's termination signal
            to ensure consistency with dynamics-based evaluation.
        env_kwargs: Optional keyword arguments to pass to gym.make().
    """
    if env_kwargs is None:
        env_kwargs = {}
    env = gym.make(env_id, **env_kwargs)
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    for ep in range(rollouts):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        discount = 1.0
        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            total += discount * reward_fn(obs, action)
            discount = discount * gamma
            obs, _, terminated, truncated, _ = env.step(action)
            # Use user-defined termination if provided, otherwise use env's signal
            if termination_fn is not None:
                should_terminate = termination_fn(obs)
            else:
                should_terminate = terminated or truncated
            if should_terminate:
                if respect_termination:
                    break  # Exit loop, episode is done
                else:
                    # Legacy behavior: reset and continue
                    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def calc_dynamics_mse(
    dynamics_model: DynamicsNet,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    device: torch.device,
) -> float:
    """Calculate MSE between predicted and true next states."""
    with torch.no_grad():
        s = torch.tensor(states, dtype=torch.float32, device=device)
        a = torch.tensor(actions, dtype=torch.float32, device=device)
        s_next_true = torch.tensor(next_states, dtype=torch.float32, device=device)
        s_next_pred = dynamics_model.sample_next(s, a, deterministic=True)
        mse = torch.mean((s_next_pred - s_next_true) ** 2).item()
    return mse


def sample_training_transitions(
    dataset: OfflineDataset,
    n_samples: int = 1000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample transitions from dataset for MSE evaluation."""
    rng = np.random.default_rng(seed)
    states = dataset.states
    actions = dataset.actions
    next_states = dataset.next_states
    mask = getattr(dataset, "mask", None)
    
    if states.ndim == 3:
        B, T, D = states.shape
        flat_s = states.reshape(B * T, D)
        flat_a = actions.reshape(B * T, actions.shape[-1])
        flat_sn = next_states.reshape(B * T, D)
        if mask is not None:
            flat_mask = mask.reshape(B * T)
            keep = flat_mask > 0.5
            flat_s = flat_s[keep]
            flat_a = flat_a[keep]
            flat_sn = flat_sn[keep]
        states = flat_s
        actions = flat_a
        next_states = flat_sn
        
    n_total = states.shape[0]
    idx = rng.choice(n_total, size=min(n_samples, n_total), replace=n_samples > n_total)
    return states[idx], actions[idx], next_states[idx]


def generate_test_transitions(
    env_id: str,
    policies: Dict[str, TorchPolicy],
    n_transitions_per_policy: int = 200,
    seed: int = 0,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test transitions by rolling out policies."""
    if env_kwargs is None:
        env_kwargs = {}
    env = gym.make(env_id, **env_kwargs)
    rng = np.random.default_rng(seed)
    all_states, all_actions, all_next_states = [], [], []
    
    for policy_name, policy in policies.items():
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        for _ in range(n_transitions_per_policy):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_tensor = policy.sample_torch_actions(obs_tensor, deterministic=True)
            action = action_tensor.squeeze(0).cpu().numpy()
            next_obs, _, terminated, truncated, _ = env.step(action)
            all_states.append(obs.copy())
            all_actions.append(action.copy())
            all_next_states.append(next_obs.copy())
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    env.close()
    
    return (
        np.asarray(all_states, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        np.asarray(all_next_states, dtype=np.float32),
    )


# =============================================================================
# Abstract Pipeline Configuration
# =============================================================================

@dataclass
class BasePipelineConfig:
    """Base configuration for pipeline experiments."""
    
    # Environment
    env_id: str = ""
    
    # PPO Training
    total_steps: int = 1_000_000
    ppo_fractions: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    n_envs: int = 8
    n_steps: int = 2048
    batch_size: int = 1024
    learning_rate: float = 3e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    
    # Dataset
    rollout_steps: int = 10_000
    
    # Q-Network
    q_epochs: int = 1000
    q_batch: int = 1024
    q_lr: float = 3e-4
    q_samples: int = 32
    
    # Dynamics
    dyn_epochs: int = 2000
    dyn_batch: int = 1024
    dyn_lr: float = 3e-4
    dyn_hidden_dim: int = 256
    dyn_val_fraction: float = 0.1
    dyn_early_stop_patience: int = 200
    dyn_min_epochs: int = 50
    dynamics_loss: str = "nll"
    lambda_td: float = 0.1
    lambda_rank: float = 0.1
    rank_temperature: float = 1.0
    rank_rollout_horizon: int = 10
    rank_rollout_episodes: int = 128
    backbone: str = "mlp"
    dyn_seq_len: int = 1
    dyn_seq_overlap: int = 0
    morel_ensemble_size: int = 4
    morel_hidden_dim: int = -1  # -1 -> use environment defaults
    morel_epochs: int = -1      # -1 -> use environment defaults
    morel_batch_size: int = 256
    morel_lr: float = 5e-4
    morel_threshold: Optional[float] = None
    morel_threshold_mode: str = "mean_std"
    morel_threshold_beta: float = 5.0
    morel_threshold_frac_of_max: float = 1.0
    morel_reward_offset: float = -1.0  # <0 -> use environment defaults
    morel_halt_reward: Optional[float] = None
    morel_bootstrap: bool = True
    mopo_ensemble_size: int = 7
    mopo_elite_size: int = 5
    mopo_hidden_dim: int = 200
    mopo_hidden_layers: int = 4
    mopo_epochs: int = 300
    mopo_batch_size: int = 256
    mopo_lr: float = 1e-3
    mopo_holdout_size: int = 1000
    mopo_penalty_coef: float = 1.0
    mopo_bootstrap: bool = True
    mopo_deterministic_dynamics_eval: bool = False
    
    # Evaluation
    eval_episodes: int = 20
    eval_rollouts: int = 500
    eval_horizon: int = 500
    
    # Paths and misc
    output_dir: Path = Path("results/pipeline")
    seed: int = 42
    device: str = "auto"
    
    # Flags
    force_policy_training: bool = False
    force_dataset_collection: bool = False
    force_q_training: bool = False
    force_dynamics_training: bool = False
    results_only: bool = False
    
    # W&B
    wandb_project: str = ""
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    
    # Action bounds (environment-specific)
    act_low: float = -1.0
    act_high: float = 1.0
    
    # Environment kwargs
    env_kwargs: Dict[str, Any] = field(default_factory=dict)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared across all environment pipelines."""
    # PPO Training
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--ppo-fractions", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    
    # Dataset
    parser.add_argument("--rollout-steps", type=int, default=10_000, help="steps per policy for dataset collection")
    
    # Q-Network
    parser.add_argument("--q-epochs", type=int, default=1000)
    parser.add_argument("--q-batch", type=int, default=1024)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--q-samples", type=int, default=32)
    
    # Dynamics
    parser.add_argument("--dyn-epochs", type=int, default=2000)
    parser.add_argument("--dyn-batch", type=int, default=1024)
    parser.add_argument("--dyn-lr", type=float, default=3e-4)
    parser.add_argument("--dyn-hidden-dim", type=int, default=256)
    parser.add_argument("--dyn-val-fraction", type=float, default=0.1)
    parser.add_argument("--dyn-early-stop-patience", type=int, default=200)
    parser.add_argument("--dyn-min-epochs", type=int, default=50)
    parser.add_argument("--dynamics-loss", type=str, default="nll", choices=["nll", "mse"])
    parser.add_argument(
        "--dynamics-models",
        type=str,
        nargs="+",
        default=["supervised", "kendall", "hinge", "listnet"],
        choices=["supervised", "kendall", "hinge", "listnet", "morel", "mopo"],
        help="Dynamics models to train/evaluate. Choose from: supervised, kendall, hinge, listnet, morel, mopo",
    )
    parser.add_argument("--lambda-td", type=float, default=0.1)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--record-lambda-val", action="store_true", help="Include lambda-rank value in summary filename for sweeps")
    parser.add_argument("--rank-temperature", type=float, default=1.0, help="Temperature for soft ranking losses (lower = sharper)")
    parser.add_argument("--rank-rollout-horizon", type=int, default=10, help="Rollout horizon for ranking loss return estimation")
    parser.add_argument("--rank-rollout-episodes", type=int, default=128, help="Number of rollout episodes for ranking loss")
    parser.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "resnet", "ode", "transformer", "gru"])
    parser.add_argument("--dyn-seq-len", type=int, default=1)
    parser.add_argument("--dyn-seq-overlap", type=int, default=0)
    parser.add_argument("--value-aware-only", action="store_true", help="Train only the value-aware dynamics model (no supervised or ranking models)")
    parser.add_argument("--morel-ensemble-size", type=int, default=4, help="MOReL ensemble size (paper default: 4)")
    parser.add_argument("--morel-hidden-dim", type=int, default=-1, help="MOReL hidden dim; -1 uses env-specific defaults")
    parser.add_argument("--morel-epochs", type=int, default=-1, help="MOReL epochs; -1 uses env-specific defaults")
    parser.add_argument("--morel-batch-size", type=int, default=256, help="MOReL dynamics batch size (paper default: 256)")
    parser.add_argument("--morel-lr", type=float, default=5e-4, help="MOReL dynamics Adam stepsize (paper default: 5e-4)")
    parser.add_argument("--morel-threshold", type=float, default=None, help="USAD threshold override; if unset, computed from dataset disagreements")
    parser.add_argument(
        "--morel-threshold-mode",
        type=str,
        default="mean_std",
        choices=["mean_std", "fraction_max"],
        help="How to compute MOReL USAD threshold when --morel-threshold is unset",
    )
    parser.add_argument("--morel-threshold-beta", type=float, default=5.0, help="Beta in threshold=mean+beta*std for mean_std mode")
    parser.add_argument("--morel-threshold-frac-max", type=float, default=1.0, help="Threshold fraction of max disagreement for fraction_max mode")
    parser.add_argument("--morel-reward-offset", type=float, default=-1.0, help="Offset in halt reward r_min(D)-offset; negative means use env defaults")
    parser.add_argument("--morel-halt-reward", type=float, default=None, help="Direct halt reward override (takes precedence over offset)")
    parser.add_argument("--morel-no-bootstrap", action="store_true", help="Disable bootstrap resampling across MOReL ensemble members")
    parser.add_argument("--mopo-ensemble-size", type=int, default=7, help="MOPO ensemble size (paper: 7)")
    parser.add_argument("--mopo-elite-size", type=int, default=5, help="MOPO number of elite models kept by holdout NLL (paper: 5)")
    parser.add_argument("--mopo-hidden-dim", type=int, default=200, help="MOPO hidden width (paper: 200)")
    parser.add_argument("--mopo-hidden-layers", type=int, default=4, help="MOPO hidden layers in feedforward dynamics (paper: 4)")
    parser.add_argument("--mopo-epochs", type=int, default=300, help="MOPO dynamics epochs")
    parser.add_argument("--mopo-batch-size", type=int, default=256, help="MOPO dynamics batch size (paper reports 256 for SAC updates)")
    parser.add_argument("--mopo-lr", type=float, default=1e-3, help="MOPO dynamics Adam learning rate")
    parser.add_argument("--mopo-holdout-size", type=int, default=1000, help="MOPO holdout transitions for elite selection (paper: 1000)")
    parser.add_argument("--mopo-penalty-coef", type=float, default=1.0, help="MOPO uncertainty penalty coefficient lambda")
    parser.add_argument("--mopo-no-bootstrap", action="store_true", help="Disable bootstrap resampling across MOPO ensemble members")
    parser.add_argument("--mopo-deterministic-dynamics-eval", action="store_true", help="Use deterministic MOPO dynamics means at evaluation time")
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-rollouts", type=int, default=500)
    parser.add_argument("--eval-horizon", type=int, default=500)
    
    # General
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force-policy-training", action="store_true")
    parser.add_argument("--force-dataset-collection", action="store_true")
    parser.add_argument("--force-q-training", action="store_true")
    parser.add_argument("--force-dynamics-training", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    
    # W&B
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    
    # Environment
    parser.add_argument("--env-kwargs", type=json.loads, default={}, help="JSON dict of kwargs to pass to gym.make()")


# =============================================================================
# Main Pipeline Runner
# =============================================================================

def run_pipeline(
    args: argparse.Namespace,
    env_id: str,
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
    reward_fn_torch: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    act_low: float = -1.0,
    act_high: float = 1.0,
    state_low: Optional[torch.Tensor] = None,
    state_upper: Optional[torch.Tensor] = None,
    wrapped_dims: Optional[List[int]] = None,
    termination_fn_torch: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete offline RL pipeline.
    
    Args:
        args: Parsed command-line arguments
        env_id: Gymnasium environment ID
        reward_fn: NumPy reward function (state, action) -> float
        reward_fn_torch: PyTorch reward function (states, actions) -> Tensor
        act_low: Action space lower bound
        act_high: Action space upper bound
        state_low: Optional lower bounds for state dimensions (for DynamicsNet)
        state_upper: Optional upper bounds for state dimensions (for DynamicsNet)
        wrapped_dims: Optional list of dimension indices that are wrapped (e.g., angles)
        termination_fn_torch: Optional termination function (states) -> bool tensor.
            Used for dynamics-based evaluation to check if episode should terminate.
        env_kwargs: Optional keyword arguments to pass to gym.make().
    
    Returns:
        Summary dictionary with all results
    """
    # Validate arguments
    if args.results_only and (
        args.force_policy_training
        or args.force_dataset_collection
        or args.force_q_training
        or args.force_dynamics_training
    ):
        raise ValueError("--results-only cannot be combined with force-training flags.")

    # Merge env_kwargs from args and function parameter
    if env_kwargs is None:
        env_kwargs = {}
    if hasattr(args, "env_kwargs") and args.env_kwargs:
        env_kwargs = {**env_kwargs, **args.env_kwargs}

    device = torch.device(args.device) if args.device != "auto" else default_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(args)

    # =========================================================================
    # Step 1: Train or load PPO policies
    # =========================================================================
    policy_dir = args.output_dir / "policies"
    snapshots_manifest = policy_dir / "snapshots.json"
    policy_trained = False
    policy_train_time: Optional[float] = None

    if snapshots_manifest.exists() and not args.force_policy_training:
        print("[Step 1] Using existing PPO checkpoints...")
        snapshots = load_snapshots(snapshots_manifest)
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved PPO checkpoints at {snapshots_manifest}. "
                "Run without --results-only to generate them."
            )
        print("[Step 1] Training PPO policies with checkpoints...")
        start_time = time.perf_counter()
        snapshots = train_ppo_with_checkpoints(
            env_id=env_id,
            total_steps=args.total_steps,
            save_dir=policy_dir,
            fractions=args.ppo_fractions,
            seed=args.seed,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            device=device.type if device.type in {"cuda", "cpu"} else "cpu",
            wandb_run=wandb_run,
            env_kwargs=env_kwargs,
        )
        save_snapshots(snapshots_manifest, snapshots)
        policy_trained = True
        policy_train_time = time.perf_counter() - start_time

    print(f"Loaded {len(snapshots)} policy checkpoints.")
    policy_log = {"policies/count": len(snapshots), "policies/trained": int(policy_trained)}
    if policy_train_time is not None:
        policy_log["timing/ppo_training_sec"] = policy_train_time
    wandb_log(wandb_run, policy_log)

    policy_models = load_policy_models(snapshots)

    # =========================================================================
    # Step 2: Collect or load offline dataset
    # =========================================================================
    dataset_path = args.output_dir / "offline_dataset.npz"
    dataset_loaded = False

    if dataset_path.exists() and not args.force_policy_training and not args.force_dataset_collection:
        print("[Step 2] Loading existing offline dataset from disk...")
        dataset = load_offline_dataset(dataset_path)
        dataset_loaded = True
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved offline dataset at {dataset_path}. "
                "Run without --results-only to generate it."
            )
        print("[Step 2] Collecting offline dataset...")
        dataset = build_offline_dataset(
            snapshots, policy_models, env_id, args.rollout_steps, args.seed, reward_fn, env_kwargs
        )
        np.savez_compressed(
            dataset_path,
            s=dataset.states,
            a=dataset.actions,
            r=dataset.rewards,
            s_next=dataset.next_states,
            done=dataset.dones,
            s0=dataset.initial_states,
        )
        print(f"Dataset saved to {dataset_path} with {len(dataset.states)} transitions")

    wandb_log(
        wandb_run,
        {
            "dataset/transitions": int(len(dataset.states)),
            "dataset/initial_states": int(len(dataset.initial_states)),
            "dataset/loaded_from_disk": int(dataset_loaded),
        },
    )

    torch_policies = make_policy_adapters(policy_models, act_low=act_low, act_high=act_high)
    dyn_dataset = make_sequence_dataset(dataset, args.dyn_seq_len, args.dyn_seq_overlap)

    # =========================================================================
    # Step 3: Train or load Q-networks
    # =========================================================================
    q_dir = args.output_dir / "q_models"
    q_manifest = q_dir / "manifest.json"
    q_trained = False
    q_train_time: Optional[float] = None

    if q_manifest.exists() and not args.force_q_training:
        print("[Step 3] Using existing Q networks...")
        q_models, q_model_paths = load_q_models(q_dir, device)
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved Q networks at {q_manifest}. "
                "Run training once without --results-only."
            )
        print("[Step 3] Training Q networks for each policy...")
        start_time = time.perf_counter()
        q_models = train_q_networks(
            dataset,
            torch_policies,
            device=device,
            gamma=args.gamma,
            epochs=args.q_epochs,
            batch_size=args.q_batch,
            lr=args.q_lr,
            samples=args.q_samples,
            wandb_run=wandb_run,
        )
        q_model_paths = save_q_models(q_models, q_dir)
        q_trained = True
        q_train_time = time.perf_counter() - start_time

    q_log = {"q_models/count": len(q_models), "q_models/trained": int(q_trained)}
    if q_train_time is not None:
        q_log["timing/q_training_sec"] = q_train_time
    wandb_log(wandb_run, q_log)

    # =========================================================================
    # Step 4: Train or load dynamics models
    # =========================================================================
    dynamics_dir = args.output_dir / args.backbone / "dynamics"
    dynamics_manifest = dynamics_dir / "manifest.json"
    dynamics_trained = False
    dynamics_train_time: Optional[float] = None
    sup_model: Optional[DynamicsNet] = None
    ranking_new_models: Dict[str, DynamicsNet] = {}
    value_aware_model: Optional[DynamicsNet] = None
    dynamics_paths: Dict[str, object] = {}
    morel_model: Optional[MorelDynamicsEnsemble] = None
    morel_info: Optional[Dict[str, Any]] = None
    mopo_model: Optional[MopoDynamicsEnsemble] = None
    mopo_info: Optional[Dict[str, Any]] = None

    value_aware_only = getattr(args, "value_aware_only", False)
    missing_requested_models: List[str] = []
    need_dynamics_training = (not dynamics_manifest.exists()) or args.force_dynamics_training

    if not need_dynamics_training:
        print("[Step 4] Using existing dynamics models...")
        sup_model, ranking_new_models, dynamics_paths, value_aware_model = load_dynamics_models(dynamics_dir, device)
        morel_model = load_morel_model_from_paths(dynamics_paths, device)
        mopo_model = load_mopo_model_from_paths(dynamics_paths, device)
        raw_morel_info = dynamics_paths.get("morel_info")
        if isinstance(raw_morel_info, dict):
            morel_info = raw_morel_info
        raw_mopo_info = dynamics_paths.get("mopo_info")
        if isinstance(raw_mopo_info, dict):
            mopo_info = raw_mopo_info

        missing_requested_models = find_missing_requested_dynamics_models(
            requested_models=args.dynamics_models,
            supervised_model=sup_model,
            ranking_new_models=ranking_new_models,
            value_aware_model=value_aware_model,
            morel_model=morel_model,
            mopo_model=mopo_model,
            value_aware_only=value_aware_only,
        )
        if missing_requested_models:
            print(
                "[Step 4] Existing dynamics manifest is missing requested models "
                f"{missing_requested_models}; retraining dynamics."
            )
            need_dynamics_training = True

    if need_dynamics_training:
        if args.results_only:
            if missing_requested_models:
                raise FileNotFoundError(
                    "Saved dynamics manifest is missing requested models "
                    f"{missing_requested_models}. Run once without --results-only "
                    "to train and save them."
                )
            if args.force_dynamics_training:
                raise FileNotFoundError(
                    "Cannot use --results-only with --force-dynamics-training; "
                    "disable one of these flags."
                )
            raise FileNotFoundError(
                f"No saved dynamics models at {dynamics_manifest}. "
                "Run training once without --results-only."
            )
        if value_aware_only:
            print("[Step 4] Training value-aware dynamics model only...")
        else:
            print("[Step 4] Training dynamics models...")
            print(f"  Models to train: {args.dynamics_models}")
        start_time = time.perf_counter()
        sup_model, ranking_new_models, value_aware_model, morel_model, morel_info, mopo_model, mopo_info = train_dynamics_models(
            dyn_dataset,
            torch_policies,
            q_models,
            device=device,
            dyn_epochs=args.dyn_epochs,
            dyn_batch=args.dyn_batch,
            dyn_lr=args.dyn_lr,
            gamma=args.gamma,
            lambda_td=args.lambda_td,
            lambda_rank=args.lambda_rank,
            rank_temperature=args.rank_temperature,
            rank_rollout_horizon=args.rank_rollout_horizon,
            rank_rollout_episodes=args.rank_rollout_episodes,
            val_fraction=args.dyn_val_fraction,
            early_stop_patience=args.dyn_early_stop_patience,
            min_epochs=args.dyn_min_epochs,
            reward_fn_torch=reward_fn_torch,
            dynamics_loss=args.dynamics_loss,
            hidden_dim=args.dyn_hidden_dim,
            backbone=args.backbone,
            wandb_run=wandb_run,
            state_low=state_low,
            state_upper=state_upper,
            wrapped_dims=wrapped_dims,
            dynamics_models=args.dynamics_models,
            value_aware_only=value_aware_only,
            act_low=act_low,
            act_high=act_high,
            seed=args.seed,
            env_name=env_id,
            morel_ensemble_size=args.morel_ensemble_size,
            morel_hidden_dim=None if args.morel_hidden_dim <= 0 else args.morel_hidden_dim,
            morel_epochs=None if args.morel_epochs <= 0 else args.morel_epochs,
            morel_batch_size=args.morel_batch_size,
            morel_lr=args.morel_lr,
            morel_threshold=args.morel_threshold,
            morel_threshold_mode=args.morel_threshold_mode,
            morel_threshold_beta=args.morel_threshold_beta,
            morel_threshold_frac_of_max=args.morel_threshold_frac_max,
            morel_reward_offset=None if args.morel_reward_offset < 0 else args.morel_reward_offset,
            morel_halt_reward=args.morel_halt_reward,
            morel_bootstrap=not args.morel_no_bootstrap,
            mopo_ensemble_size=args.mopo_ensemble_size,
            mopo_elite_size=args.mopo_elite_size,
            mopo_hidden_dim=args.mopo_hidden_dim,
            mopo_hidden_layers=args.mopo_hidden_layers,
            mopo_epochs=args.mopo_epochs,
            mopo_batch_size=args.mopo_batch_size,
            mopo_lr=args.mopo_lr,
            mopo_holdout_size=args.mopo_holdout_size,
            mopo_penalty_coef=args.mopo_penalty_coef,
            mopo_bootstrap=not args.mopo_no_bootstrap,
        )
        dynamics_paths = save_dynamics_models(
            sup_model,
            ranking_new_models,
            dynamics_dir,
            value_aware_model=value_aware_model,
            morel_model=morel_model,
            morel_info=morel_info,
            mopo_model=mopo_model,
            mopo_info=mopo_info,
        )
        dynamics_trained = True
        dynamics_train_time = time.perf_counter() - start_time

    dyn_log = {"dynamics/trained": int(dynamics_trained)}
    if dynamics_train_time is not None:
        dyn_log["timing/dynamics_training_sec"] = dynamics_train_time
    if morel_info is not None:
        if morel_info.get("threshold") is not None:
            dyn_log["dynamics/morel_threshold"] = morel_info["threshold"]
        if morel_info.get("halt_reward") is not None:
            dyn_log["dynamics/morel_halt_reward"] = morel_info["halt_reward"]
        if morel_info.get("disagreement_mean") is not None:
            dyn_log["dynamics/morel_disagreement_mean"] = morel_info["disagreement_mean"]
    if mopo_info is not None:
        if mopo_info.get("penalty_coef") is not None:
            dyn_log["dynamics/mopo_penalty_coef"] = mopo_info["penalty_coef"]
        if mopo_info.get("uncertainty_mean") is not None:
            dyn_log["dynamics/mopo_uncertainty_mean"] = mopo_info["uncertainty_mean"]
        if mopo_info.get("uncertainty_max") is not None:
            dyn_log["dynamics/mopo_uncertainty_max"] = mopo_info["uncertainty_max"]
        if mopo_info.get("elite_size") is not None:
            dyn_log["dynamics/mopo_elite_size"] = mopo_info["elite_size"]
    wandb_log(wandb_run, dyn_log)

    # =========================================================================
    # Step 5-7: Evaluation
    # =========================================================================
    print("[Step 5-7] Evaluating policies across true env, Q, and dynamics...")
    results = []
    initial_states = dataset.initial_states

    # Generate test/train transitions for MSE evaluation
    test_states, test_actions, test_next_states = generate_test_transitions(
        env_id, torch_policies, n_transitions_per_policy=200, seed=args.seed, env_kwargs=env_kwargs
    )
    train_states, train_actions, train_next_states = sample_training_transitions(
        dyn_dataset, n_samples=1000, seed=args.seed
    )

    for snap in snapshots:
        name = snap["name"]
        policy = torch_policies[name]
        ppo_model = policy_models[name]
        q_net = q_models[name]

        q_est = evaluate_q_estimate(q_net, policy, initial_states, args.eval_rollouts)
        
        # Supervised model evaluation (if trained/loaded)
        dyn_sup: Optional[float] = None
        dyn_sup_mse: Optional[float] = None
        dyn_sup_train_mse: Optional[float] = None
        if sup_model is not None:
            dyn_sup = evaluate_in_dynamics(
                sup_model, policy, initial_states, args.eval_horizon,
                args.gamma, device, args.eval_rollouts, reward_fn_torch,
                termination_fn_torch=termination_fn_torch,
            )
            dyn_sup_mse = calc_dynamics_mse(sup_model, test_states, test_actions, test_next_states, device)
            dyn_sup_train_mse = calc_dynamics_mse(sup_model, train_states, train_actions, train_next_states, device)

        # Ranking models evaluation
        dyn_rank_new: Dict[str, float] = {}
        dyn_rank_new_mse: Dict[str, float] = {}
        dyn_rank_new_train_mse: Dict[str, float] = {}
        for loss_name, dyn_model in ranking_new_models.items():
            dyn_rank_new[loss_name] = evaluate_in_dynamics(
                dyn_model, policy, initial_states, args.eval_horizon,
                args.gamma, device, args.eval_rollouts, reward_fn_torch,
                termination_fn_torch=termination_fn_torch,
            )
            dyn_rank_new_mse[loss_name] = calc_dynamics_mse(
                dyn_model, test_states, test_actions, test_next_states, device
            )
            dyn_rank_new_train_mse[loss_name] = calc_dynamics_mse(
                dyn_model, train_states, train_actions, train_next_states, device
            )

        # Value-aware model evaluation
        dyn_value_aware: Optional[float] = None
        dyn_value_aware_mse: Optional[float] = None
        dyn_value_aware_train_mse: Optional[float] = None
        if value_aware_model is not None:
            dyn_value_aware = evaluate_in_dynamics(
                value_aware_model, policy, initial_states, args.eval_horizon,
                args.gamma, device, args.eval_rollouts, reward_fn_torch,
                termination_fn_torch=termination_fn_torch,
            )
            dyn_value_aware_mse = calc_dynamics_mse(
                value_aware_model, test_states, test_actions, test_next_states, device
            )
            dyn_value_aware_train_mse = calc_dynamics_mse(
                value_aware_model, train_states, train_actions, train_next_states, device
            )

        # MOReL pessimistic-MDP evaluation (fixed-policy rollouts)
        dyn_morel_pess: Optional[float] = None
        dyn_morel_pess_stderr: Optional[float] = None
        dyn_morel_unknown_rate: Optional[float] = None
        dyn_morel_halted_fraction: Optional[float] = None
        if morel_model is not None:
            morel_seed = int(args.seed + snap["timesteps"])
            dyn_morel_pess, dyn_morel_pess_stderr, morel_metrics = evaluate_in_morel_pessimistic_mdp(
                morel_model=morel_model,
                policy=policy,
                initial_states=initial_states,
                horizon=args.eval_horizon,
                gamma=args.gamma,
                rollouts=args.eval_rollouts,
                seed=morel_seed,
                reward_fn_torch=reward_fn_torch,
                act_low=act_low,
                act_high=act_high,
            )
            dyn_morel_unknown_rate = float(morel_metrics.get("unknown_rate", 0.0))
            dyn_morel_halted_fraction = float(morel_metrics.get("halted_fraction", 0.0))

        # MOPO uncertainty-penalized model evaluation (fixed-policy rollouts)
        dyn_mopo_penalized: Optional[float] = None
        dyn_mopo_penalized_stderr: Optional[float] = None
        dyn_mopo_uncertainty_mean: Optional[float] = None
        dyn_mopo_penalty_mean: Optional[float] = None
        dyn_mopo_model_reward_mean: Optional[float] = None
        if mopo_model is not None:
            mopo_seed = int(args.seed + snap["timesteps"] + 17)
            dyn_mopo_penalized, dyn_mopo_penalized_stderr, mopo_metrics = evaluate_in_mopo_penalized_mdp(
                mopo_model=mopo_model,
                policy=policy,
                initial_states=initial_states,
                horizon=args.eval_horizon,
                gamma=args.gamma,
                rollouts=args.eval_rollouts,
                seed=mopo_seed,
                act_low=act_low,
                act_high=act_high,
                deterministic_policy=True,
                deterministic_dynamics=bool(args.mopo_deterministic_dynamics_eval),
            )
            dyn_mopo_uncertainty_mean = float(mopo_metrics.get("uncertainty_mean", 0.0))
            dyn_mopo_penalty_mean = float(mopo_metrics.get("penalty_mean", 0.0))
            dyn_mopo_model_reward_mean = float(mopo_metrics.get("model_reward_mean", 0.0))

        # Create numpy wrapper for termination function if provided
        termination_fn_np: Optional[Callable[[np.ndarray], bool]] = None
        if termination_fn_torch is not None:
            def termination_fn_np(state: np.ndarray) -> bool:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    return bool(termination_fn_torch(state_t).item())
        
        fixed_env = evaluate_sb3_policy_fixed_horizon(
            ppo_model, env_id, args.eval_horizon, args.gamma,
            args.eval_rollouts, args.seed, reward_fn,
            termination_fn=termination_fn_np,
            env_kwargs=env_kwargs,
        )

        # Log to W&B
        wandb_payload = {
            "eval/policy_name": name,
            "eval/q_estimate": q_est,
            "eval/env_mc": fixed_env,
        }
        if dyn_sup is not None:
            wandb_payload["eval/dynamics_supervised"] = dyn_sup
        if dyn_sup_mse is not None:
            wandb_payload["eval/dynamics_supervised_mse"] = dyn_sup_mse
        if dyn_sup_train_mse is not None:
            wandb_payload["eval/dynamics_supervised_train_mse"] = dyn_sup_train_mse
        for loss_name, val in dyn_rank_new.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}"] = val
        for loss_name, val in dyn_rank_new_mse.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}_mse"] = val
        for loss_name, val in dyn_rank_new_train_mse.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}_train_mse"] = val
        if dyn_value_aware is not None:
            wandb_payload["eval/dynamics_value_aware"] = dyn_value_aware
        if dyn_value_aware_mse is not None:
            wandb_payload["eval/dynamics_value_aware_mse"] = dyn_value_aware_mse
        if dyn_value_aware_train_mse is not None:
            wandb_payload["eval/dynamics_value_aware_train_mse"] = dyn_value_aware_train_mse
        if dyn_morel_pess is not None:
            wandb_payload["eval/morel_pessimistic_return"] = dyn_morel_pess
        if dyn_morel_pess_stderr is not None:
            wandb_payload["eval/morel_pessimistic_stderr"] = dyn_morel_pess_stderr
        if dyn_morel_unknown_rate is not None:
            wandb_payload["eval/morel_unknown_rate"] = dyn_morel_unknown_rate
        if dyn_morel_halted_fraction is not None:
            wandb_payload["eval/morel_halted_fraction"] = dyn_morel_halted_fraction
        if dyn_mopo_penalized is not None:
            wandb_payload["eval/mopo_penalized_return"] = dyn_mopo_penalized
        if dyn_mopo_penalized_stderr is not None:
            wandb_payload["eval/mopo_penalized_stderr"] = dyn_mopo_penalized_stderr
        if dyn_mopo_uncertainty_mean is not None:
            wandb_payload["eval/mopo_uncertainty_mean"] = dyn_mopo_uncertainty_mean
        if dyn_mopo_penalty_mean is not None:
            wandb_payload["eval/mopo_penalty_mean"] = dyn_mopo_penalty_mean
        if dyn_mopo_model_reward_mean is not None:
            wandb_payload["eval/mopo_model_reward_mean"] = dyn_mopo_model_reward_mean

        wandb_log(wandb_run, wandb_payload, step=int(snap["timesteps"]))

        results.append({
            "name": name,
            "checkpoint": snap["path"],
            "timesteps": snap["timesteps"],
            "q_estimate": q_est,
            "dynamics": {
                "supervised": dyn_sup,
                "supervised_mse": dyn_sup_mse,
                "supervised_train_mse": dyn_sup_train_mse,
                "ranking_new": dyn_rank_new,
                "ranking_new_mse": dyn_rank_new_mse,
                "ranking_new_train_mse": dyn_rank_new_train_mse,
                "value_aware": dyn_value_aware,
                "value_aware_mse": dyn_value_aware_mse,
                "value_aware_train_mse": dyn_value_aware_train_mse,
                "morel_pessimistic": dyn_morel_pess,
                "morel_pessimistic_stderr": dyn_morel_pess_stderr,
                "morel_unknown_rate": dyn_morel_unknown_rate,
                "morel_halted_fraction": dyn_morel_halted_fraction,
                "mopo_penalized": dyn_mopo_penalized,
                "mopo_penalized_stderr": dyn_mopo_penalized_stderr,
                "mopo_uncertainty_mean": dyn_mopo_uncertainty_mean,
                "mopo_penalty_mean": dyn_mopo_penalty_mean,
                "mopo_model_reward_mean": dyn_mopo_model_reward_mean,
            },
            "env_mc": fixed_env,
        })

    # =========================================================================
    # Save summary
    # =========================================================================
    config = args_to_config(args)
    summary = {
        "config": config,
        "num_transitions": int(len(dataset.states)),
        "q_model_paths": q_model_paths,
        "dynamics_paths": dynamics_paths,
        "morel_info": morel_info,
        "mopo_info": mopo_info,
        "results": results,
    }

    if value_aware_only:
        summary_path = args.output_dir / args.backbone / f"summary_VAML_{args.seed}.json"
    elif args.dynamics_loss == "mse":
        summary_path = args.output_dir / args.backbone / f"summary_MSE_{args.seed}.json"
    elif args.record_lambda_val:
        summary_path = args.output_dir / args.backbone / f"summary_{args.lambda_rank}_{args.seed}.json"
    else:
        summary_path = args.output_dir / args.backbone / f"summary_{args.seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    if wandb_run is not None:
        wandb_run.summary.update({
            "num_transitions": int(len(dataset.states)),
            "q_model_paths": q_model_paths,
            "dynamics_paths": dynamics_paths,
            "morel_info": morel_info,
            "mopo_info": mopo_info,
            "results": results,
        })
        wandb_run.finish()

    return summary
