from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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


PENDULUM_ACT_LOW = -2.0
PENDULUM_ACT_HIGH = 2.0
PENDULUM_STATE_LOW = torch.tensor([-1.0, -1.0, -8.0], dtype=torch.float32)
PENDULUM_STATE_HIGH = torch.tensor([1.0, 1.0, 8.0], dtype=torch.float32)

_DEFINED_WANDB_METRICS: Set[str] = set()
_DEFINED_WANDB_STEP_KEYS: Set[str] = set()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def pendulum_reward_np(state: np.ndarray, action: np.ndarray) -> float:
    cos_theta, sin_theta, theta_dot = state
    theta = math.atan2(sin_theta, cos_theta)
    torque = float(np.clip(action, PENDULUM_ACT_LOW, PENDULUM_ACT_HIGH).item())
    return -(
        theta * theta
        + 0.1 * theta_dot * theta_dot
        + 0.001 * torque * torque
    )


def pendulum_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    cos_theta, sin_theta, theta_dot = states.unbind(dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    torque_penalty = torch.sum(actions * actions, dim=-1)
    return -(theta.pow(2) + 0.1 * theta_dot.pow(2) + 0.001 * torque_penalty)


class FractionCheckpointCallback(BaseCallback):
    def __init__(self, milestones: Sequence[int], save_dir: Path, prefix: str = "ppo_frac", verbose: int = 0):
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
    def __init__(self, run: Optional[Any], prefix: str = "ppo"):
        super().__init__(verbose=0)
        self.run = run
        self.prefix = prefix

    def _on_step(self) -> bool:
        # No per-step logging; method exists to satisfy BaseCallback API.
        return True

    def _on_rollout_end(self) -> bool:
        if self.run is None:
            return True
        log_dict = getattr(self.logger, "name_to_value", None)
        if not log_dict:
            return True
        payload = {}
        for key, value in log_dict.items():
            if isinstance(value, (int, float)):
                payload[f"{self.prefix}/{key}"] = value
        if payload:
            wandb_log(self.run, payload, step=self.num_timesteps)
        return True


class SB3PolicyAdapter:
    def __init__(self, name: str, model: PPO, batch_size: int = 2048):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        action_space = model.action_space
        self.action_dim = int(np.prod(action_space.shape, dtype=int))
        self.act_low = float(np.min(action_space.low))
        self.act_high = float(np.max(action_space.high))

    def sample_torch_actions(
        self,
        states: torch.Tensor,
        repeats: int = 1,
        deterministic: bool = False,
        act_low: float = PENDULUM_ACT_LOW,
        act_high: float = PENDULUM_ACT_HIGH,
    ) -> torch.Tensor:
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


def load_policy_models(policy_snapshots: Sequence[Dict[str, object]]) -> Dict[str, PPO]:
    return {snap["name"]: PPO.load(snap["path"]) for snap in policy_snapshots}


def make_policy_adapters(policy_models: Dict[str, PPO]) -> Dict[str, TorchPolicy]:
    return {name: SB3PolicyAdapter(name, model) for name, model in policy_models.items()}


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def initialize_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.wandb_project:
        return None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=args_to_config(args),
    )


def wandb_log(run: Optional[Any], payload: Dict[str, Any], step: Optional[int] = None) -> None:
    if run is not None:
        run.log(payload, step=step)


def _register_wandb_metric(run: Any, metric_key: str) -> str:
    step_key = f"{metric_key}_step"
    if step_key not in _DEFINED_WANDB_STEP_KEYS:
        run.define_metric(step_key, summary="max")
        _DEFINED_WANDB_STEP_KEYS.add(step_key)
    if metric_key not in _DEFINED_WANDB_METRICS:
        run.define_metric(metric_key, step_metric=step_key)
        _DEFINED_WANDB_METRICS.add(metric_key)
    return step_key


def make_epoch_logger(run: Optional[Any], metric_key: str, extra: Optional[Dict[str, Any]] = None):
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


def save_snapshots(manifest_path: Path, snapshots: List[Dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, indent=2)


def load_snapshots(manifest_path: Path) -> List[Dict[str, object]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
) -> List[Dict[str, object]]:
    set_random_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MlpPolicy",
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
    callback_obj: BaseCallback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
    model.learn(total_timesteps=total_steps, callback=callback_obj, progress_bar=True)

    vec_env.close()
    snapshots.extend(checkpoint_cb.saved)
    return snapshots


def rollout_policy(model: PPO, env_id: str, total_steps: int, seed: int) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]]:
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
        reward = pendulum_reward_np(obs, action)
        done = float(terminated or truncated)
        transitions.append((obs.copy(), action.copy(), reward, next_obs.copy(), done))
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
) -> OfflineDataset:
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_next_states: List[np.ndarray] = []
    all_dones: List[float] = []
    initial_states: List[np.ndarray] = []

    for idx, snap in enumerate(policy_snapshots):
        model = policy_models[snap["name"]]
        init_states, transitions = rollout_policy(model, env_id, steps_per_policy, seed + idx)
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
        )


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
            act_low=PENDULUM_ACT_LOW,
            act_high=PENDULUM_ACT_HIGH,
            device=device,
            use_amp=device.type == "cuda",
            log_hook=make_epoch_logger(wandb_run, f"q_train/{name}"),
        )
        q_models[name] = rescaled_q
    return q_models


def save_q_models(q_models: Dict[str, torch.nn.Module], directory: Path) -> Dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, model in q_models.items():
        path = directory / f"{name}.pt"
        torch.save(copy.deepcopy(model).cpu(), path)
        saved[name] = path.as_posix()
    with open(directory / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    return saved


def load_q_models(directory: Path, device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Dict[str, str]]:
    manifest = directory / "manifest.json"
    with open(manifest, "r", encoding="utf-8") as f:
        saved = json.load(f)
    models: Dict[str, torch.nn.Module] = {}
    for name, path_str in saved.items():
        model = torch.load(path_str, map_location=device, weights_only=False)
        model.to(device)
        models[name] = model
    return models, saved


def make_dynamics_net(state_dim: int, act_dim: int) -> DynamicsNet:
    return DynamicsNet(
        state_dim=state_dim,
        act_dim=act_dim,
        state_low=PENDULUM_STATE_LOW,
        state_upper=PENDULUM_STATE_HIGH,
        wrapped_dims=None,
    )


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
    wandb_run: Optional[Any] = None,
) -> Tuple[DynamicsNet, Dict[str, DynamicsNet], DynamicsNet, Dict[str, DynamicsNet]]:
    state_dim = dataset.states.shape[1]
    act_dim = dataset.actions.shape[1]

    sup_model = make_dynamics_net(state_dim, act_dim).to(device)
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
    )

    """q_aware_models: Dict[str, DynamicsNet] = {}
    for name, policy in policies.items():
        model = make_dynamics_net(state_dim, act_dim).to(device)
        model.train_q_aware_model(
            dataset,
            target_policy=policy,
            q_fn=q_models[name],
            gamma=gamma,
            lambda_td=lambda_td,
            epochs=dyn_epochs,
            batch_size=dyn_batch,
            lr=dyn_lr,
            act_low=PENDULUM_ACT_LOW,
            act_high=PENDULUM_ACT_HIGH,
            reward_fn=pendulum_reward_torch,
            device=device,
            log_hook=make_epoch_logger(wandb_run, f"dynamics/qaware/{name}"),
            val_fraction=val_fraction,
            early_stop_patience=early_stop_patience,
            min_epochs=min_epochs,
        )
        q_aware_models[name] = model"""

    policy_q_pairs = [(policies[name], q_models[name]) for name in policies]
    ranking_model = make_dynamics_net(state_dim, act_dim).to(device)
    ranking_model.train_ranking_aware_model(
        dataset,
        policy_q_pairs=policy_q_pairs,
        gamma=gamma,
        lambda_rank=lambda_rank,
        epochs=dyn_epochs,
        batch_size=dyn_batch,
        lr=dyn_lr,
        act_low=PENDULUM_ACT_LOW,
        act_high=PENDULUM_ACT_HIGH,
        reward_fn=pendulum_reward_torch,
        device=device,
        log_hook=make_epoch_logger(wandb_run, "dynamics/ranking"),
        val_fraction=val_fraction,
        early_stop_patience=early_stop_patience,
        min_epochs=min_epochs,
    )

    # Train additional ranking-aware models using the new loss variants.
    ranking_new_models: Dict[str, DynamicsNet] = {}
    for loss_name in ("kendall", "hinge", "listnet"):
        model = make_dynamics_net(state_dim, act_dim).to(device)
        model.train_ranking_aware_model_new(
            dataset,
            policy_q_pairs=policy_q_pairs,
            gamma=gamma,
            lambda_rank=lambda_rank,
            epochs=dyn_epochs,
            batch_size=dyn_batch,
            lr=dyn_lr,
            act_low=PENDULUM_ACT_LOW,
            act_high=PENDULUM_ACT_HIGH,
            reward_fn=pendulum_reward_torch,
            device=device,
            log_hook=make_epoch_logger(wandb_run, f"dynamics/ranking_new/{loss_name}"),
            val_fraction=val_fraction,
            early_stop_patience=early_stop_patience,
            min_epochs=min_epochs,
            ranking_loss_type=loss_name,
        )
        ranking_new_models[loss_name] = model

    q_aware_models = {}
    return sup_model, q_aware_models, ranking_model, ranking_new_models


def save_dynamics_models(
    sup_model: DynamicsNet,
    q_aware_models: Dict[str, DynamicsNet],
    ranking_model: DynamicsNet,
    ranking_new_models: Dict[str, DynamicsNet],
    directory: Path,
) -> Dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, object] = {
        "supervised": (directory / "dynamics_supervised.pt").as_posix(),
        "ranking": (directory / "dynamics_ranking.pt").as_posix(),
        "q_aware": {},
        "ranking_new": {},
    }
    torch.save(copy.deepcopy(sup_model).cpu(), paths["supervised"])
    torch.save(copy.deepcopy(ranking_model).cpu(), paths["ranking"])
    """for name, model in q_aware_models.items():
        path = directory / f"dynamics_q_{name}.pt"
        torch.save(copy.deepcopy(model).cpu(), path)
        paths["q_aware"][name] = path.as_posix()"""

    for loss_name, model in ranking_new_models.items():
        path = directory / f"dynamics_ranking_new_{loss_name}.pt"
        torch.save(copy.deepcopy(model).cpu(), path)
        paths["ranking_new"][loss_name] = path.as_posix()

    with open(directory / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)
    return paths


def load_dynamics_models(
    directory: Path,
    device: torch.device,
) -> Tuple[DynamicsNet, Dict[str, DynamicsNet], DynamicsNet, Dict[str, DynamicsNet], Dict[str, object]]:
    manifest = directory / "manifest.json"
    with open(manifest, "r", encoding="utf-8") as f:
        paths = json.load(f)

    sup_model: DynamicsNet = torch.load(paths["supervised"], map_location=device, weights_only=False)
    sup_model.to(device)

    ranking_model: DynamicsNet = torch.load(paths["ranking"], map_location=device, weights_only=False)
    ranking_model.to(device)

    q_aware_models: Dict[str, DynamicsNet] = {}
    for name, path in paths.get("q_aware", {}).items():
        model: DynamicsNet = torch.load(path, map_location=device, weights_only=False)
        model.to(device)
        q_aware_models[name] = model

    ranking_new_models: Dict[str, DynamicsNet] = {}
    for loss_name, path in paths.get("ranking_new", {}).items():
        model: DynamicsNet = torch.load(path, map_location=device, weights_only=False)
        model.to(device)
        ranking_new_models[loss_name] = model

    return sup_model, q_aware_models, ranking_model, ranking_new_models, paths


def evaluate_sb3_policy(model: PPO, env_id: str, episodes: int, seed: int) -> float:
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            total += pendulum_reward_np(obs, action)
            obs, _, done, truncated, _ = env.step(action)
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def evaluate_sb3_policy_mc(
    model: PPO,
    env_id: str,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
) -> float:
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    returns = []
    for ep in range(rollouts):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total = 0.0
        discount = 1.0
        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            total += discount * pendulum_reward_np(obs, action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            discount *= gamma
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def evaluate_q_estimate(
    q_model: QNet,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    samples: int,
) -> float:
    return float(
        estimate_V_from_Q_on_s0(
            q_model,
            initial_states,
            policy,
            K=samples,
        )
    )


def evaluate_in_dynamics_mc(
    dynamics: DynamicsNet,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    horizon: int,
    gamma: float,
    device: torch.device,
    rollouts: int,
) -> float:
    if initial_states.shape[0] == 0:
        raise ValueError("No initial states available for dynamics evaluation.")
    idx = np.random.choice(initial_states.shape[0], size=rollouts, replace=initial_states.shape[0] < rollouts)
    states = torch.tensor(initial_states[idx], dtype=torch.float32, device=device)
    total = torch.zeros(states.size(0), device=device)
    discount = torch.ones_like(total)
    for _ in range(horizon):
        actions = policy.sample_torch_actions(
            states,
            deterministic=True,
            act_low=PENDULUM_ACT_LOW,
            act_high=PENDULUM_ACT_HIGH,
        )
        rewards = pendulum_reward_torch(states, actions)
        total += discount * rewards
        discount = discount * gamma
        states = dynamics.sample_next(states, actions, deterministic=True)
    return float(total.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end Pendulum offline RL pipeline")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--ppo-fractions", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-steps", type=int, default=5_000, help="steps per policy for dataset collection")
    parser.add_argument("--q-epochs", type=int, default=1000)
    parser.add_argument("--q-batch", type=int, default=1024)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--q-samples", type=int, default=32)
    parser.add_argument("--dyn-epochs", type=int, default=2000)
    parser.add_argument("--dyn-batch", type=int, default=1024)
    parser.add_argument("--dyn-lr", type=float, default=1e-3)
    parser.add_argument("--dyn-val-fraction", type=float, default=0.1)
    parser.add_argument("--dyn-early-stop-patience", type=int, default=200)
    parser.add_argument("--dyn-min-epochs", type=int, default=50)
    parser.add_argument("--lambda-td", type=float, default=0.1)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-rollouts", type=int, default=256)
    parser.add_argument("--eval-horizon", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mse/pendulum_pipeline"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force-policy-training", action="store_true")
    parser.add_argument("--force-q-training", action="store_true")
    parser.add_argument("--force-dynamics-training", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="DT2-pendulum")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    args = parser.parse_args()

    if args.results_only and (
        args.force_policy_training or args.force_q_training or args.force_dynamics_training
    ):
        parser.error("--results-only cannot be combined with force-training flags.")

    device = torch.device(args.device) if args.device != "auto" else default_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(args)

    policy_dir = args.output_dir / "policies"
    snapshots_manifest = policy_dir / "snapshots.json"
    if snapshots_manifest.exists() and not args.force_policy_training:
        print("[Step 1] Using existing PPO checkpoints...")
        snapshots = load_snapshots(snapshots_manifest)
        policy_trained = False
        policy_train_time: Optional[float] = None
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved PPO checkpoints at {snapshots_manifest}. Run without --results-only to generate them."
            )
        print("[Step 1] Training PPO policies with checkpoints...")
        start_time = time.perf_counter()
        snapshots = train_ppo_with_checkpoints(
            env_id=args.env_id,
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
    wandb_log(
        wandb_run,
        {
            "policies/count": len(snapshots),
            "policies/trained": int(policy_trained),
            **({"timing/ppo_training_sec": policy_train_time} if policy_train_time is not None else {}),
        },
    )

    policy_models = load_policy_models(snapshots)

    dataset_path = args.output_dir / "offline_dataset.npz"
    if args.results_only:
        print("[Step 2] Loading offline dataset from disk...")
        dataset = load_offline_dataset(dataset_path)
    else:
        print("[Step 2] Collecting offline dataset...")
        dataset = build_offline_dataset(snapshots, policy_models, args.env_id, args.rollout_steps, args.seed)
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
            "dataset/source_loaded": int(args.results_only),
        },
    )

    torch_policies = make_policy_adapters(policy_models)
    q_dir = args.output_dir / "q_models"
    q_manifest = q_dir / "manifest.json"
    if q_manifest.exists() and not args.force_q_training:
        print("[Step 3] Using existing Q networks...")
        q_models, q_model_paths = load_q_models(q_dir, device)
        q_trained = False
        q_train_time: Optional[float] = None
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved Q networks at {q_manifest}. Run training once without --results-only."
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
    wandb_log(
        wandb_run,
        {
            "q_models/count": len(q_models),
            "q_models/trained": int(q_trained),
            **({"timing/q_training_sec": q_train_time} if q_train_time is not None else {}),
        },
    )

    dynamics_dir = args.output_dir / "dynamics"
    dynamics_manifest = dynamics_dir / "manifest.json"
    if dynamics_manifest.exists() and not args.force_dynamics_training:
        print("[Step 4] Using existing dynamics models...")
        sup_model, q_aware_models, ranking_model, ranking_new_models, dynamics_paths = load_dynamics_models(
            dynamics_dir, device
        )
        dynamics_trained = False
        dynamics_train_time: Optional[float] = None
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved dynamics models at {dynamics_manifest}. Run training once without --results-only."
            )
        print("[Step 4] Training dynamics models...")
        start_time = time.perf_counter()
        sup_model, q_aware_models, ranking_model, ranking_new_models = train_dynamics_models(
            dataset,
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
            wandb_run=wandb_run,
        )
        dynamics_paths = save_dynamics_models(
            sup_model,
            q_aware_models,
            ranking_model,
            ranking_new_models,
            dynamics_dir,
        )
        dynamics_trained = True
        dynamics_train_time = time.perf_counter() - start_time
    wandb_log(
        wandb_run,
        {
            "dynamics/trained": int(dynamics_trained),
            **({"timing/dynamics_training_sec": dynamics_train_time} if dynamics_train_time is not None else {}),
        },
    )

    print("[Step 5-7] Evaluating policies across true env, Q, MC env, and dynamics...")
    results = []
    initial_states = dataset.initial_states
    for snap in snapshots:
        name = snap["name"]
        policy = torch_policies[name]
        ppo_model = policy_models[name]
        q_net = q_models[name]

        true_return = evaluate_sb3_policy(ppo_model, args.env_id, args.eval_episodes, args.seed)
        env_mc = evaluate_sb3_policy_mc(
            ppo_model,
            args.env_id,
            args.eval_horizon,
            args.gamma,
            args.eval_rollouts,
            args.seed,
        )
        q_est = evaluate_q_estimate(q_net, policy, initial_states, args.eval_rollouts)
        dyn_sup = evaluate_in_dynamics_mc(
            sup_model,
            policy,
            initial_states,
            args.eval_horizon,
            args.gamma,
            device,
            args.eval_rollouts,
        )
        #dyn_q = evaluate_in_dynamics_mc(
        #    q_aware_models[name],
        #    policy,
        #    initial_states,
        #    args.eval_horizon,
        #    args.gamma,
        #    device,
        #    args.eval_rollouts,
        #)
        dyn_rank = evaluate_in_dynamics_mc(
            ranking_model,
            policy,
            initial_states,
            args.eval_horizon,
            args.gamma,
            device,
            args.eval_rollouts,
        )
        dyn_rank_new: Dict[str, float] = {}
        for loss_name, dyn_model in ranking_new_models.items():
            dyn_rank_new[loss_name] = evaluate_in_dynamics_mc(
                dyn_model,
                policy,
                initial_states,
                args.eval_horizon,
                args.gamma,
                device,
                args.eval_rollouts,
            )

        wandb_payload = {
            "eval/policy_name": name,
            "eval/true_return": true_return,
            "eval/env_mc": env_mc,
            "eval/q_estimate": q_est,
            "eval/dynamics_supervised": dyn_sup,
            #"eval/dynamics_qaware": dyn_q,
            "eval/dynamics_ranking": dyn_rank,
        }
        for loss_name, val in dyn_rank_new.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}"] = val

        wandb_log(
            wandb_run,
            wandb_payload,
            step=int(snap["timesteps"]),
        )

        results.append(
            {
                "name": name,
                "checkpoint": snap["path"],
                "timesteps": snap["timesteps"],
                "true_return": true_return,
                "env_mc": env_mc,
                "q_estimate": q_est,
                "dynamics": {
                    "supervised": dyn_sup,
                    #"q_aware": dyn_q,
                    "ranking": dyn_rank,
                    "ranking_new": dyn_rank_new,
                },
            }
        )

    summary = {
        "config": args_to_config(args),
        "num_transitions": int(len(dataset.states)),
        "q_model_paths": q_model_paths,
        "dynamics_paths": dynamics_paths,
        "results": results,
    }

    summary_path = args.output_dir / f"summary_{args.seed}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
