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
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

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
    sup_model: DynamicsNet,
    ranking_new_models: Dict[str, DynamicsNet],
    directory: Path,
) -> Dict[str, object]:
    """Save dynamics models and create a manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, object] = {
        "supervised": (directory / "dynamics_supervised.pt").as_posix(),
        "ranking": (directory / "dynamics_ranking.pt").as_posix(),
        "q_aware": {},
        "ranking_new": {},
    }
    torch.save(copy.deepcopy(sup_model).cpu(), paths["supervised"])

    for loss_name, model in ranking_new_models.items():
        path = directory / f"dynamics_ranking_new_{loss_name}.pt"
        torch.save(copy.deepcopy(model).cpu(), path)
        paths["ranking_new"][loss_name] = path.as_posix()

    manifest = directory / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    return paths


def load_dynamics_models(
    directory: Path,
    device: torch.device,
) -> Tuple[DynamicsNet, Dict[str, DynamicsNet], Dict[str, object]]:
    """Load dynamics models from a manifest."""
    manifest = directory / "manifest.json"
    with open(manifest, "r", encoding="utf-8") as f:
        paths = json.load(f)

    sup_model: DynamicsNet = torch.load(paths["supervised"], map_location=device, weights_only=False)
    sup_model.to(device)

    ranking_new_models: Dict[str, DynamicsNet] = {}
    for loss_name, path in paths.get("ranking_new", {}).items():
        model: DynamicsNet = torch.load(path, map_location=device, weights_only=False)
        model.to(device)
        ranking_new_models[loss_name] = model

    return sup_model, ranking_new_models, paths


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
) -> List[Dict[str, object]]:
    """Train PPO and save checkpoints at specified fractions of training.
    
    Args:
        env_id: Environment ID (used if env_factory is None)
        env_factory: Optional callable that creates an environment instance.
                    If provided, this is used instead of env_id.
    """
    set_random_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    save_dir.mkdir(parents=True, exist_ok=True)

    if env_factory is not None:
        vec_env = make_vec_env(env_factory, n_envs=n_envs, seed=seed)
    else:
        vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    vec_env = VecMonitor(vec_env)

    # Auto-detect policy type
    if policy_type == "auto":
        probe_env = gym.make(env_id)
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
        verbose=1,
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
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]]:
    """Collect transitions by rolling out a policy in an environment."""
    env = gym.make(env_id)
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
            model, env_id, steps_per_policy, seed + idx, reward_fn
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
) -> Tuple[DynamicsNet, Dict[str, DynamicsNet]]:
    """Train supervised and ranking-aware dynamics models."""
    state_dim = dataset.states.shape[-1]
    act_dim = dataset.actions.shape[-1]

    # Supervised model
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
    policy_q_pairs = [(policies[name], q_models[name]) for name in policies]
    ranking_new_models: Dict[str, DynamicsNet] = {}
    for loss_name in ("kendall", "hinge", "listnet"):
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
        )
        ranking_new_models[loss_name] = model

    return sup_model, ranking_new_models


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
) -> float:
    """Evaluate a policy by rolling out in a learned dynamics model."""
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
    for _ in range(horizon):
        actions = policy.sample_torch_actions(states, deterministic=True)
        rewards = reward_fn_torch(states, actions)
        total += discount * rewards
        discount = discount * gamma
        states = dynamics.sample_next(states, actions, deterministic=True)
    return float(total.mean().item())


def evaluate_sb3_policy_fixed_horizon(
    model: PPO,
    env_id: str,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
    reward_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Roll out PPO in the true env for a fixed horizon, ignoring terminations."""
    env = gym.make(env_id)
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
            if terminated or truncated:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test transitions by rolling out policies."""
    env = gym.make(env_id)
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
    backbone: str = "mlp"
    dyn_seq_len: int = 1
    dyn_seq_overlap: int = 0
    
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
    parser.add_argument("--lambda-td", type=float, default=0.1)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "resnet", "ode", "transformer", "gru"])
    parser.add_argument("--dyn-seq-len", type=int, default=1)
    parser.add_argument("--dyn-seq-overlap", type=int, default=0)
    
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
            snapshots, policy_models, env_id, args.rollout_steps, args.seed, reward_fn
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

    if dynamics_manifest.exists() and not args.force_dynamics_training:
        print("[Step 4] Using existing dynamics models...")
        sup_model, ranking_new_models, dynamics_paths = load_dynamics_models(dynamics_dir, device)
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved dynamics models at {dynamics_manifest}. "
                "Run training once without --results-only."
            )
        print("[Step 4] Training dynamics models...")
        start_time = time.perf_counter()
        sup_model, ranking_new_models = train_dynamics_models(
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
        )
        dynamics_paths = save_dynamics_models(sup_model, ranking_new_models, dynamics_dir)
        dynamics_trained = True
        dynamics_train_time = time.perf_counter() - start_time

    dyn_log = {"dynamics/trained": int(dynamics_trained)}
    if dynamics_train_time is not None:
        dyn_log["timing/dynamics_training_sec"] = dynamics_train_time
    wandb_log(wandb_run, dyn_log)

    # =========================================================================
    # Step 5-7: Evaluation
    # =========================================================================
    print("[Step 5-7] Evaluating policies across true env, Q, and dynamics...")
    results = []
    initial_states = dataset.initial_states

    # Generate test/train transitions for MSE evaluation
    test_states, test_actions, test_next_states = generate_test_transitions(
        env_id, torch_policies, n_transitions_per_policy=200, seed=args.seed + 999
    )
    train_states, train_actions, train_next_states = sample_training_transitions(
        dyn_dataset, n_samples=1000, seed=args.seed + 123
    )

    for snap in snapshots:
        name = snap["name"]
        policy = torch_policies[name]
        ppo_model = policy_models[name]
        q_net = q_models[name]

        q_est = evaluate_q_estimate(q_net, policy, initial_states, args.eval_rollouts)
        dyn_sup = evaluate_in_dynamics(
            sup_model, policy, initial_states, args.eval_horizon,
            args.gamma, device, args.eval_rollouts, reward_fn_torch
        )

        # MSE evaluation
        dyn_sup_mse = calc_dynamics_mse(sup_model, test_states, test_actions, test_next_states, device)
        dyn_sup_train_mse = calc_dynamics_mse(sup_model, train_states, train_actions, train_next_states, device)

        dyn_rank_new: Dict[str, float] = {}
        dyn_rank_new_mse: Dict[str, float] = {}
        dyn_rank_new_train_mse: Dict[str, float] = {}
        for loss_name, dyn_model in ranking_new_models.items():
            dyn_rank_new[loss_name] = evaluate_in_dynamics(
                dyn_model, policy, initial_states, args.eval_horizon,
                args.gamma, device, args.eval_rollouts, reward_fn_torch
            )
            dyn_rank_new_mse[loss_name] = calc_dynamics_mse(
                dyn_model, test_states, test_actions, test_next_states, device
            )
            dyn_rank_new_train_mse[loss_name] = calc_dynamics_mse(
                dyn_model, train_states, train_actions, train_next_states, device
            )

        fixed_env = evaluate_sb3_policy_fixed_horizon(
            ppo_model, env_id, args.eval_horizon, args.gamma,
            args.eval_rollouts, args.seed, reward_fn
        )

        # Log to W&B
        wandb_payload = {
            "eval/policy_name": name,
            "eval/q_estimate": q_est,
            "eval/dynamics_supervised": dyn_sup,
            "eval/dynamics_supervised_mse": dyn_sup_mse,
            "eval/dynamics_supervised_train_mse": dyn_sup_train_mse,
            "eval/env_mc": fixed_env,
        }
        for loss_name, val in dyn_rank_new.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}"] = val
        for loss_name, val in dyn_rank_new_mse.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}_mse"] = val
        for loss_name, val in dyn_rank_new_train_mse.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}_train_mse"] = val

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
        "results": results,
    }

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
            "results": results,
        })
        wandb_run.finish()

    return summary
