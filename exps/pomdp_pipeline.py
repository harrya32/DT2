"""
Pipeline for Partially Observable MDP (POMDP) environment.

This environment tests the hypothesis that ranking-aware dynamics losses
outperform pure NLL when the true dynamics function is outside the hypothesis class.

Key idea:
- The TRUE dynamics depend on hidden state variables that the agent cannot observe
- The agent only sees a partial observation of the full state
- Because the dynamics model only receives observations (not full state),
  it cannot perfectly predict transitions - there's irreducible error
- NLL will try to fit P(s'|obs, a) = ∫ P(s'|obs, hidden, a) P(hidden|obs) d_hidden
  which may be multi-modal or high-variance
- Ranking-aware loss focuses on preserving policy ordering, which may be more
  robust to this model misspecification

Environment:
- Full state: [x, v, hidden_force, hidden_friction]
- Observation: [x, v] (agent cannot see hidden_force or hidden_friction)
- hidden_force: a constant bias force that affects acceleration
- hidden_friction: affects velocity damping
- These hidden variables make transitions unpredictable from observations alone
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import wandb

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor

from src.datasets import OfflineDataset
from src.fqe import estimate_V_from_Q_on_s0
from src.networks import DynamicsNet, QNet
from src.policies import TorchPolicy

_DEFINED_WANDB_METRICS: Set[str] = set()
_DEFINED_WANDB_STEP_KEYS: Set[str] = set()


# =============================================================================
# Custom POMDP Environment
# =============================================================================

# Discrete hidden modes - creates truly multi-modal transitions
# Each mode has distinct dynamics that NLL (unimodal Gaussian) cannot capture
# Moderate parameters: enough variance for model misspecification, but learnable by PPO
HIDDEN_MODES = {
    0: {"force": 1.5, "friction": 0.1, "action_scale": 1.0},    # Moderate positive bias
    1: {"force": -1.5, "friction": 0.1, "action_scale": 1.0},   # Moderate negative bias
    2: {"force": 0.0, "friction": 0.3, "action_scale": 0.5},    # Dampened controls (high friction)
    3: {"force": 0.0, "friction": 0.0, "action_scale": 1.5},    # Slightly amplified controls
}


class POMDPEnv(gym.Env):
    """
    Partially Observable MDP for testing dynamics model misspecification.
    
    This version uses DISCRETE hidden modes to create truly multi-modal
    transition distributions that a unimodal Gaussian (NLL) cannot capture.
    
    Full State: [x, v, hidden_mode]
    - x: position in [-2, 2]
    - v: velocity in [-2, 2]
    - hidden_mode: integer in {0, 1, 2, 3} determining dynamics behavior
    
    Observation: [x, v] - the agent CANNOT see hidden_mode
    
    Action: continuous in [-1, 1], controls acceleration
    
    Hidden Modes create qualitatively different dynamics:
    - Mode 0: Strong rightward force bias
    - Mode 1: Strong leftward force bias  
    - Mode 2: High friction + REVERSED controls (action has opposite effect!)
    - Mode 3: No friction + amplified controls (very responsive)
    
    This creates multi-modal P(s'|s,a):
    - Same (x, v, action) leads to 4 distinct possible outcomes
    - NLL averages over modes → poor predictions
    - Ranking-aware can still preserve policy ordering
    
    Reward: based on reaching goal position (x=0) with low velocity
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        max_steps: int = 200,
        hidden_force_range: Tuple[float, float] = (-0.5, 0.5),  # Legacy, ignored if use_discrete_modes=True
        hidden_friction_range: Tuple[float, float] = (0.0, 0.3),  # Legacy, ignored if use_discrete_modes=True
        goal_position: float = 0.0,
        dt: float = 0.1,
        use_discrete_modes: bool = True,
        num_modes: int = 4,
        mode_probs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.hidden_force_range = hidden_force_range
        self.hidden_friction_range = hidden_friction_range
        self.goal_position = goal_position
        self.dt = dt
        self.use_discrete_modes = use_discrete_modes
        self.num_modes = min(num_modes, len(HIDDEN_MODES))
        self.mode_probs = mode_probs  # If None, uniform distribution
        
        # Full state dim vs observation dim
        self.full_state_dim = 3 if use_discrete_modes else 4  # [x, v, mode] or [x, v, force, friction]
        self.obs_dim = 2  # [x, v]
        
        # Observation space (what the agent sees)
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self._full_state = None
        self._hidden_mode = None
        self._step_count = 0
        self._rng = np.random.default_rng()
    
    def _get_obs(self) -> np.ndarray:
        """Return the observable part of the state."""
        return self._full_state[:2].copy()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Initialize observable state
        x = self._rng.uniform(-1.0, 1.0)
        v = self._rng.uniform(-0.5, 0.5)
        
        if self.use_discrete_modes:
            # Sample a hidden mode for this episode
            if self.mode_probs is not None:
                self._hidden_mode = self._rng.choice(self.num_modes, p=self.mode_probs[:self.num_modes])
            else:
                self._hidden_mode = self._rng.integers(0, self.num_modes)
            self._full_state = np.array([x, v, float(self._hidden_mode)], dtype=np.float32)
        else:
            # Legacy continuous hidden variables
            hidden_force = self._rng.uniform(*self.hidden_force_range)
            hidden_friction = self._rng.uniform(*self.hidden_friction_range)
            self._full_state = np.array([x, v, hidden_force, hidden_friction], dtype=np.float32)
        
        self._step_count = 0
        return self._get_obs(), {}
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).flatten()[0]
        
        x, v = self._full_state[0], self._full_state[1]
        
        if self.use_discrete_modes:
            # Get mode parameters
            mode_params = HIDDEN_MODES[self._hidden_mode]
            hidden_force = mode_params["force"]
            hidden_friction = mode_params["friction"]
            action_scale = mode_params["action_scale"]
            
            # Dynamics with mode-specific behavior
            effective_action = action * action_scale
            acceleration = effective_action + hidden_force
        else:
            # Legacy continuous hidden variables
            hidden_force = self._full_state[2]
            hidden_friction = self._full_state[3]
            acceleration = action + hidden_force
        
        # Velocity update with friction
        v_new = v * (1 - hidden_friction) + acceleration * self.dt
        v_new = np.clip(v_new, -2.0, 2.0)
        
        x_new = x + v_new * self.dt
        x_new = np.clip(x_new, -2.0, 2.0)
        
        # Update state
        if self.use_discrete_modes:
            self._full_state = np.array([x_new, v_new, float(self._hidden_mode)], dtype=np.float32)
        else:
            self._full_state = np.array([x_new, v_new, hidden_force, hidden_friction], dtype=np.float32)
        
        self._step_count += 1
        
        # Reward only depends on observable state
        reward = self._compute_reward(x_new, v_new, action)
        
        # Termination
        terminated = False
        truncated = self._step_count >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _compute_reward(self, x, v, action):
        """Reward for reaching goal with low velocity."""
        goal_dist = abs(x - self.goal_position)
        
        # Reward for being close to goal
        proximity_reward = 1.0 - goal_dist / 2.0  # Max 1.0 at goal
        
        # Velocity penalty
        velocity_penalty = -0.1 * abs(v)
        
        # Action cost
        action_cost = -0.05 * abs(action)
        
        # Bonus for reaching goal region
        goal_bonus = 1.0 if goal_dist < 0.1 else 0.0
        
        return proximity_reward + velocity_penalty + action_cost + goal_bonus
    
    def get_full_state(self) -> np.ndarray:
        """For analysis: return the full state including hidden variables."""
        return self._full_state.copy()
    
    def get_hidden_mode(self) -> int:
        """For analysis: return the current hidden mode."""
        return self._hidden_mode if self.use_discrete_modes else -1


def pomdp_reward_fn(obs: np.ndarray, action: np.ndarray) -> float:
    """Numpy reward function for evaluation (uses observation only)."""
    x = obs[0]
    v = obs[1]
    act = action.flatten()[0] if hasattr(action, 'flatten') else action
    
    goal_dist = abs(x)
    proximity_reward = 1.0 - goal_dist / 2.0
    velocity_penalty = -0.1 * abs(v)
    action_cost = -0.05 * abs(act)
    goal_bonus = 1.0 if goal_dist < 0.1 else 0.0
    
    return proximity_reward + velocity_penalty + action_cost + goal_bonus


def pomdp_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Torch reward function for dynamics training."""
    x = states[:, 0]
    v = states[:, 1]
    act = actions[:, 0] if actions.dim() > 1 else actions
    
    goal_dist = torch.abs(x)
    proximity_reward = 1.0 - goal_dist / 2.0
    velocity_penalty = -0.1 * torch.abs(v)
    action_cost = -0.05 * torch.abs(act)
    goal_bonus = (goal_dist < 0.1).float()
    
    return proximity_reward + velocity_penalty + action_cost + goal_bonus


# Register the environment
gym.register(
    id="POMDPEnv-v0",
    entry_point="pomdp_pipeline:POMDPEnv",
    max_episode_steps=200,
)


# =============================================================================
# Utility functions (adapted from anisotropic_pipeline)
# =============================================================================

def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    """Callback to log PPO training metrics to wandb."""
    def __init__(self, run: Optional[Any], prefix: str = "ppo"):
        super().__init__(verbose=0)
        self.run = run
        self.prefix = prefix
        self._metrics_defined = False
        self._step_key = f"{prefix}/step"

    def _on_training_start(self) -> None:
        """Define metrics with custom step axis on first call."""
        if self.run is None or self._metrics_defined:
            return
        # Define the step metric for PPO
        self.run.define_metric(self._step_key)
        # All ppo/* metrics will use ppo/step as their x-axis
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


class SB3PolicyAdapter:
    def __init__(self, name: str, model: PPO, batch_size: int = 2048, act_low: float = -1.0, act_high: float = 1.0):
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
        act_low: float = -1.0,
        act_high: float = 1.0,
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
    models: Dict[str, PPO] = {}
    for snap in policy_snapshots:
        models[snap["name"]] = PPO.load(snap["path"])
    return models


def make_policy_adapters(policy_models: Dict[str, PPO]) -> Dict[str, TorchPolicy]:
    return {name: SB3PolicyAdapter(name, model) for name, model in policy_models.items()}


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def initialize_wandb(args: argparse.Namespace) -> Optional[Any]:
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
    if run is None:
        return
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


def make_epoch_logger(
    run: Optional[Any],
    metric_key: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Callable[..., None]]:
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


# =============================================================================
# Training functions
# =============================================================================

def make_pomdp_env(
    max_steps: int = 200,
    hidden_force_range: Tuple[float, float] = (-0.5, 0.5),
    hidden_friction_range: Tuple[float, float] = (0.0, 0.3),
    use_discrete_modes: bool = True,
    num_modes: int = 4,
):
    """Factory for creating the POMDP environment."""
    return POMDPEnv(
        max_steps=max_steps,
        hidden_force_range=hidden_force_range,
        hidden_friction_range=hidden_friction_range,
        use_discrete_modes=use_discrete_modes,
        num_modes=num_modes,
    )


def train_ppo_with_checkpoints(
    max_steps: int,
    hidden_force_range: Tuple[float, float],
    hidden_friction_range: Tuple[float, float],
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
    use_discrete_modes: bool = True,
    num_modes: int = 4,
) -> List[Dict[str, object]]:
    set_random_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    save_dir.mkdir(parents=True, exist_ok=True)

    def env_fn():
        return POMDPEnv(
            max_steps=max_steps,
            hidden_force_range=hidden_force_range,
            hidden_friction_range=hidden_friction_range,
            use_discrete_modes=use_discrete_modes,
            num_modes=num_modes,
        )

    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed)
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
    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)

    vec_env.close()
    snapshots.extend(checkpoint_cb.saved)
    return snapshots


def rollout_policy(
    model: PPO,
    max_steps: int,
    hidden_force_range: Tuple[float, float],
    hidden_friction_range: Tuple[float, float],
    total_steps: int,
    seed: int,
    use_discrete_modes: bool = True,
    num_modes: int = 4,
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]]:
    """Rollout policy and collect (observation, action, reward, next_observation, done) tuples."""
    env = POMDPEnv(
        max_steps=max_steps,
        hidden_force_range=hidden_force_range,
        hidden_friction_range=hidden_friction_range,
        use_discrete_modes=use_discrete_modes,
        num_modes=num_modes,
    )
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    initial_states: List[np.ndarray] = [obs.copy()]
    transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
    steps = 0
    while steps < total_steps:
        action, _ = model.predict(obs, deterministic=False)
        action = np.asarray(action, dtype=np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated or truncated)
        transitions.append((obs.copy(), action.copy(), float(reward), next_obs.copy(), done))
        obs = next_obs
        steps += 1
        if done:
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            initial_states.append(obs.copy())
    return initial_states, transitions


def build_offline_dataset(
    policy_snapshots: Sequence[Dict[str, object]],
    policy_models: Dict[str, PPO],
    max_steps: int,
    hidden_force_range: Tuple[float, float],
    hidden_friction_range: Tuple[float, float],
    steps_per_policy: int,
    seed: int,
    use_discrete_modes: bool = True,
    num_modes: int = 4,
) -> OfflineDataset:
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_next_states: List[np.ndarray] = []
    all_dones: List[float] = []
    initial_states: List[np.ndarray] = []

    for idx, snap in enumerate(policy_snapshots):
        model = policy_models[snap["name"]]
        init_states, transitions = rollout_policy(
            model, max_steps, hidden_force_range, hidden_friction_range,
            steps_per_policy, seed + idx,
            use_discrete_modes=use_discrete_modes, num_modes=num_modes
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
    manifest = directory / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
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


def train_dynamics_models(
    dataset: OfflineDataset,
    policies: Dict[str, TorchPolicy],
    q_models: Dict[str, QNet],
    device: torch.device,
    dyn_epochs: int,
    dyn_batch: int,
    dyn_lr: float,
    gamma: float,
    lambda_rank: float,
    val_fraction: float,
    early_stop_patience: int,
    min_epochs: int,
    dynamics_loss: str = "nll",
    wandb_run: Optional[Any] = None,
) -> Tuple[DynamicsNet, Dict[str, DynamicsNet]]:
    state_dim = dataset.states.shape[1]  # 2 (only observables)
    act_dim = dataset.actions.shape[1]

    # State bounds for the POMDP observation space
    # Observation: x in [-2, 2], v in [-2, 2]
    state_low = torch.tensor([-2.0, -2.0], dtype=torch.float32)
    state_high = torch.tensor([2.0, 2.0], dtype=torch.float32)

    # Supervised dynamics
    sup_model = DynamicsNet(
        state_dim=state_dim,
        act_dim=act_dim,
        state_low=state_low,
        state_upper=state_high,
        wrapped_dims=[],  # No wrapped dims in this env
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

    # Ranking-aware dynamics with different loss types
    policy_q_pairs = [(policies[name], q_models[name]) for name in policies]
    ranking_new_models: Dict[str, DynamicsNet] = {}
    
    for loss_name in ("kendall", "hinge", "listnet"):
        model = DynamicsNet(
            state_dim=state_dim,
            act_dim=act_dim,
            state_low=state_low,
            state_upper=state_high,
            wrapped_dims=[],
        ).to(device)
        model.train_ranking_aware_model_new(
            dataset,
            policy_q_pairs=policy_q_pairs,
            gamma=gamma,
            lambda_rank=lambda_rank,
            epochs=dyn_epochs,
            batch_size=dyn_batch,
            lr=dyn_lr,
            reward_fn=pomdp_reward_torch,
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


def save_dynamics_models(
    sup_model: DynamicsNet,
    ranking_new_models: Dict[str, DynamicsNet],
    directory: Path,
) -> Dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, object] = {
        "supervised": (directory / "dynamics_supervised.pt").as_posix(),
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
# Evaluation functions
# =============================================================================

def evaluate_sb3_policy(
    model: PPO,
    max_steps: int,
    hidden_force_range: Tuple[float, float],
    hidden_friction_range: Tuple[float, float],
    episodes: int,
    seed: int,
    use_discrete_modes: bool = True,
    num_modes: int = 4,
) -> float:
    env = POMDPEnv(
        max_steps=max_steps,
        hidden_force_range=hidden_force_range,
        hidden_friction_range=hidden_friction_range,
        use_discrete_modes=use_discrete_modes,
        num_modes=num_modes,
    )
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        total = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            total += pomdp_reward_fn(obs, action)
            obs, _, done, truncated, _ = env.step(action)
        returns.append(total)
    return float(np.mean(returns))


def evaluate_sb3_policy_fixed_horizon(
    model: PPO,
    max_steps: int,
    hidden_force_range: Tuple[float, float],
    hidden_friction_range: Tuple[float, float],
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
    use_discrete_modes: bool = True,
    num_modes: int = 4,
) -> float:
    """Roll out PPO in the true env for a fixed horizon."""
    env = POMDPEnv(
        max_steps=max_steps,
        hidden_force_range=hidden_force_range,
        hidden_friction_range=hidden_friction_range,
        use_discrete_modes=use_discrete_modes,
        num_modes=num_modes,
    )
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    for ep in range(rollouts):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        discount = 1.0
        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            total += discount * pomdp_reward_fn(obs, action)
            discount *= gamma
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        returns.append(total)
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


def evaluate_in_dynamics(
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
        actions = policy.sample_torch_actions(states, deterministic=True)
        rewards = pomdp_reward_torch(states, actions)
        total += discount * rewards
        discount *= gamma
        states = dynamics.sample_next(states, actions, deterministic=True)
    return float(total.mean().item())


# =============================================================================
# Analysis functions for understanding partial observability effects
# =============================================================================

def analyze_transition_variance(
    max_steps: int,
    hidden_force_range: Tuple[float, float],
    hidden_friction_range: Tuple[float, float],
    num_samples: int = 1000,
    seed: int = 42,
    use_discrete_modes: bool = True,
    num_modes: int = 4,
) -> Dict[str, float]:
    """
    Analyze how much variance in transitions is due to hidden variables.
    
    For the same (obs, action), measure variance in next_obs across different
    hidden variable settings. This quantifies the "irreducible" prediction error.
    """
    rng = np.random.default_rng(seed)
    
    # Sample some fixed (obs, action) pairs
    test_obs = rng.uniform([-1.0, -0.5], [1.0, 0.5], size=(100, 2)).astype(np.float32)
    test_actions = rng.uniform(-1.0, 1.0, size=(100, 1)).astype(np.float32)
    
    # For each (obs, action), sample many hidden variable settings and compute next_obs
    all_variances = []
    
    for obs, action in zip(test_obs, test_actions):
        next_obs_samples = []
        
        if use_discrete_modes:
            # With discrete modes, compute outcome for each mode
            for mode in range(num_modes):
                mode_params = HIDDEN_MODES[mode]
                x, v = obs
                effective_action = action[0] * mode_params["action_scale"]
                acceleration = effective_action + mode_params["force"]
                v_new = v * (1 - mode_params["friction"]) + acceleration * 0.1
                v_new = np.clip(v_new, -2.0, 2.0)
                x_new = np.clip(x + v_new * 0.1, -2.0, 2.0)
                next_obs_samples.append([x_new, v_new])
        else:
            # Continuous hidden variables
            for _ in range(num_samples):
                hidden_force = rng.uniform(*hidden_force_range)
                hidden_friction = rng.uniform(*hidden_friction_range)
                
                x, v = obs
                acceleration = action[0] + hidden_force
                v_new = v * (1 - hidden_friction) + acceleration * 0.1
                v_new = np.clip(v_new, -2.0, 2.0)
                x_new = np.clip(x + v_new * 0.1, -2.0, 2.0)
                
                next_obs_samples.append([x_new, v_new])
        
        next_obs_samples = np.array(next_obs_samples)
        variance = np.var(next_obs_samples, axis=0).sum()
        all_variances.append(variance)
    
    return {
        "mean_transition_variance": float(np.mean(all_variances)),
        "std_transition_variance": float(np.std(all_variances)),
        "max_transition_variance": float(np.max(all_variances)),
    }


def analyze_mode_separation(seed: int = 42) -> Dict[str, Any]:
    """
    Analyze how different the outcomes are across hidden modes.
    
    This shows the multi-modality that NLL cannot capture.
    """
    rng = np.random.default_rng(seed)
    
    # Test with a few representative (obs, action) pairs
    test_cases = [
        (np.array([0.0, 0.0]), np.array([0.5])),   # Center, positive action
        (np.array([0.0, 0.0]), np.array([-0.5])),  # Center, negative action
        (np.array([1.0, 0.5]), np.array([0.0])),   # Right side, no action
        (np.array([-1.0, -0.5]), np.array([1.0])), # Left side, strong action
    ]
    
    results = []
    for obs, action in test_cases:
        mode_outcomes = {}
        for mode in range(len(HIDDEN_MODES)):
            mode_params = HIDDEN_MODES[mode]
            x, v = obs
            effective_action = action[0] * mode_params["action_scale"]
            acceleration = effective_action + mode_params["force"]
            v_new = v * (1 - mode_params["friction"]) + acceleration * 0.1
            v_new = np.clip(v_new, -2.0, 2.0)
            x_new = np.clip(x + v_new * 0.1, -2.0, 2.0)
            mode_outcomes[f"mode_{mode}"] = {"x_new": float(x_new), "v_new": float(v_new)}
        
        results.append({
            "obs": obs.tolist(),
            "action": action.tolist(),
            "outcomes": mode_outcomes,
        })
    
    return {"test_cases": results}


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="POMDP environment pipeline for testing dynamics misspecification")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--use-discrete-modes", action="store_true", default=True, help="Use discrete hidden modes (recommended)")
    parser.add_argument("--use-continuous-hidden", action="store_true", help="Use continuous hidden variables instead of discrete modes")
    parser.add_argument("--num-modes", type=int, default=4, help="Number of discrete hidden modes (1-4)")
    parser.add_argument("--hidden-force-min", type=float, default=-1.5, help="Min hidden force bias (continuous mode)")
    parser.add_argument("--hidden-force-max", type=float, default=1.5, help="Max hidden force bias (continuous mode)")
    parser.add_argument("--hidden-friction-min", type=float, default=0.0, help="Min hidden friction (continuous mode)")
    parser.add_argument("--hidden-friction-max", type=float, default=0.6, help="Max hidden friction (continuous mode)")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--ppo-fractions", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-steps", type=int, default=5_000, help="Steps per policy for dataset collection")
    parser.add_argument("--q-epochs", type=int, default=500)
    parser.add_argument("--q-batch", type=int, default=512)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--q-samples", type=int, default=32)
    parser.add_argument("--dyn-epochs", type=int, default=2000)
    parser.add_argument("--dyn-batch", type=int, default=512)
    parser.add_argument("--dyn-lr", type=float, default=3e-4)
    parser.add_argument("--dyn-val-fraction", type=float, default=0.1)
    parser.add_argument("--dyn-early-stop-patience", type=int, default=50)
    parser.add_argument("--dyn-min-epochs", type=int, default=50)
    parser.add_argument("--dynamics-loss", type=str, default="nll", choices=["nll", "mse"], help="Loss function for dynamics training")
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-rollouts", type=int, default=200)
    parser.add_argument("--eval-horizon", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pomdp_pipeline"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force-policy-training", action="store_true")
    parser.add_argument("--force-dataset-collection", action="store_true", help="ignore saved offline dataset and recollect")
    parser.add_argument("--force-q-training", action="store_true")
    parser.add_argument("--force-dynamics-training", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    parser.add_argument("--analyze-variance", action="store_true", help="Analyze transition variance due to hidden variables")
    parser.add_argument("--wandb-project", type=str, default="DT2-pomdp")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.results_only and (
        args.force_policy_training or args.force_dataset_collection or args.force_q_training or args.force_dynamics_training
    ):
        parser.error("--results-only cannot be combined with force-training flags.")

    # Determine if using discrete modes or continuous hidden variables
    use_discrete_modes = not args.use_continuous_hidden
    num_modes = args.num_modes
    
    hidden_force_range = (args.hidden_force_min, args.hidden_force_max)
    hidden_friction_range = (args.hidden_friction_min, args.hidden_friction_max)

    device = torch.device(args.device) if args.device != "auto" else default_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(args)

    # Log environment info
    wandb_log(wandb_run, {
        "env/obs_dim": 2,
        "env/use_discrete_modes": use_discrete_modes,
        "env/num_modes": num_modes if use_discrete_modes else 0,
        "env/hidden_force_range": list(hidden_force_range),
        "env/hidden_friction_range": list(hidden_friction_range),
    })

    # Optional: Analyze transition variance
    if args.analyze_variance:
        print("[Analysis] Computing transition variance due to hidden variables...")
        variance_stats = analyze_transition_variance(
            args.max_steps, hidden_force_range, hidden_friction_range,
            num_samples=1000, seed=args.seed,
            use_discrete_modes=use_discrete_modes, num_modes=num_modes
        )
        print(f"  Mean transition variance: {variance_stats['mean_transition_variance']:.6f}")
        print(f"  Std transition variance: {variance_stats['std_transition_variance']:.6f}")
        print(f"  Max transition variance: {variance_stats['max_transition_variance']:.6f}")
        wandb_log(wandb_run, {f"analysis/{k}": v for k, v in variance_stats.items()})
        
        if use_discrete_modes:
            print("\n[Analysis] Mode separation analysis:")
            mode_analysis = analyze_mode_separation(seed=args.seed)
            for i, case in enumerate(mode_analysis["test_cases"]):
                print(f"  Case {i}: obs={case['obs']}, action={case['action']}")
                for mode_name, outcome in case["outcomes"].items():
                    print(f"    {mode_name}: x'={outcome['x_new']:.3f}, v'={outcome['v_new']:.3f}")

    # Step 1: Train PPO policies
    policy_dir = args.output_dir / "policies"
    snapshots_manifest = policy_dir / "snapshots.json"
    policy_trained = False
    policy_train_time: Optional[float] = None
    
    if snapshots_manifest.exists() and not args.force_policy_training:
        print("[Step 1] Using existing PPO checkpoints...")
        snapshots = load_snapshots(snapshots_manifest)
    else:
        if args.results_only:
            raise FileNotFoundError(f"No saved PPO checkpoints at {snapshots_manifest}.")
        print("[Step 1] Training PPO policies with checkpoints...")
        start_time = time.perf_counter()
        snapshots = train_ppo_with_checkpoints(
            max_steps=args.max_steps,
            hidden_force_range=hidden_force_range,
            hidden_friction_range=hidden_friction_range,
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
            use_discrete_modes=use_discrete_modes,
            num_modes=num_modes,
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

    # Step 2: Build offline dataset
    dataset_path = args.output_dir / "offline_dataset.npz"
    dataset_loaded = False
    if dataset_path.exists() and not args.force_policy_training and not args.force_dataset_collection:
        print("[Step 2] Loading existing offline dataset from disk...")
        dataset = load_offline_dataset(dataset_path)
        dataset_loaded = True
    else:
        if args.results_only:
            raise FileNotFoundError(
                f"No saved offline dataset at {dataset_path}. Run without --results-only to generate it."
            )
        print("[Step 2] Collecting offline dataset...")
        dataset = build_offline_dataset(
            snapshots, policy_models, args.max_steps,
            hidden_force_range, hidden_friction_range,
            args.rollout_steps, args.seed,
            use_discrete_modes=use_discrete_modes, num_modes=num_modes
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

    wandb_log(wandb_run, {
        "dataset/transitions": int(len(dataset.states)),
        "dataset/initial_states": int(len(dataset.initial_states)),
        "dataset/loaded_from_disk": int(dataset_loaded),
    })

    torch_policies = make_policy_adapters(policy_models)

    # Step 3: Train Q networks
    q_dir = args.output_dir / "q_models"
    q_manifest = q_dir / "manifest.json"
    q_trained = False
    q_train_time: Optional[float] = None
    
    if q_manifest.exists() and not args.force_q_training:
        print("[Step 3] Using existing Q networks...")
        q_models, q_model_paths = load_q_models(q_dir, device)
    else:
        if args.results_only:
            raise FileNotFoundError(f"No saved Q networks at {q_manifest}.")
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

    # Step 4: Train dynamics models
    dynamics_dir = args.output_dir / "dynamics"
    dynamics_manifest = dynamics_dir / "manifest.json"
    dynamics_trained = False
    dynamics_train_time: Optional[float] = None
    
    if dynamics_manifest.exists() and not args.force_dynamics_training:
        print("[Step 4] Using existing dynamics models...")
        sup_model, ranking_new_models, dynamics_paths = load_dynamics_models(dynamics_dir, device)
    else:
        if args.results_only:
            raise FileNotFoundError(f"No saved dynamics models at {dynamics_manifest}.")
        print("[Step 4] Training dynamics models...")
        start_time = time.perf_counter()
        sup_model, ranking_new_models = train_dynamics_models(
            dataset,
            torch_policies,
            q_models,
            device=device,
            dyn_epochs=args.dyn_epochs,
            dyn_batch=args.dyn_batch,
            dyn_lr=args.dyn_lr,
            gamma=args.gamma,
            lambda_rank=args.lambda_rank,
            val_fraction=args.dyn_val_fraction,
            early_stop_patience=args.dyn_early_stop_patience,
            min_epochs=args.dyn_min_epochs,
            dynamics_loss=args.dynamics_loss,
            wandb_run=wandb_run,
        )
        dynamics_paths = save_dynamics_models(sup_model, ranking_new_models, dynamics_dir)
        dynamics_trained = True
        dynamics_train_time = time.perf_counter() - start_time

    dyn_log = {"dynamics/trained": int(dynamics_trained)}
    if dynamics_train_time is not None:
        dyn_log["timing/dynamics_training_sec"] = dynamics_train_time
    wandb_log(wandb_run, dyn_log)

    # Step 5-7: Evaluate
    print("[Step 5-7] Evaluating policies across true env, Q, and dynamics...")
    results = []
    initial_states = dataset.initial_states
    
    for snap in snapshots:
        name = snap["name"]
        policy = torch_policies[name]
        ppo_model = policy_models[name]
        q_net = q_models[name]

        q_est = evaluate_q_estimate(q_net, policy, initial_states, args.eval_rollouts)
        dyn_sup = evaluate_in_dynamics(
            sup_model, policy, initial_states, args.eval_horizon, args.gamma, device, args.eval_rollouts
        )
        
        dyn_rank_new: Dict[str, float] = {}
        for loss_name, dyn_model in ranking_new_models.items():
            dyn_rank_new[loss_name] = evaluate_in_dynamics(
                dyn_model, policy, initial_states, args.eval_horizon, args.gamma, device, args.eval_rollouts
            )
        
        fixed_env = evaluate_sb3_policy_fixed_horizon(
            ppo_model, args.max_steps, hidden_force_range, hidden_friction_range,
            args.eval_horizon, args.gamma, args.eval_rollouts, args.seed,
            use_discrete_modes=use_discrete_modes, num_modes=num_modes
        )

        wandb_payload = {
            "eval/policy_name": name,
            "eval/q_estimate": q_est,
            "eval/dynamics_supervised": dyn_sup,
            "eval/env_mc": fixed_env,
        }
        for loss_name, val in dyn_rank_new.items():
            wandb_payload[f"eval/dynamics_ranking_new_{loss_name}"] = val

        wandb_log(wandb_run, wandb_payload, step=int(snap["timesteps"]))

        results.append({
            "name": name,
            "checkpoint": snap["path"],
            "timesteps": snap["timesteps"],
            "q_estimate": q_est,
            "dynamics": {
                "supervised": dyn_sup,
                "ranking_new": dyn_rank_new,
            },
            "env_mc": fixed_env,
        })

    # Save summary
    config = args_to_config(args)
    summary = {
        "config": config,
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
        wandb_run.summary.update({
            "num_transitions": int(len(dataset.states)),
            "results": results,
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
