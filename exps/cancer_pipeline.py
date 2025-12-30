"""
Pipeline for Ghaffari Cancer Treatment Environment from DTRGym.

This environment models combined radiotherapy and chemotherapy for cancer with metastasis.
It's a challenging medical decision-making environment with:
- High-dimensional continuous state (tumor and immune cell populations)
- 2D continuous action space (radiation dose + chemotherapy concentration)
- Realistic pharmacokinetic/pharmacodynamic dynamics via ODEs
- Varying noise levels for observation, state, and PKPD parameters

Reference:
"A mixed radiotherapy and chemotherapy model for treatment of cancer with metastasis"
https://onlinelibrary.wiley.com/doi/full/10.1002/mma.3887

Observation Space (7D, log-transformed):
    T_p: Tumor cell population at primary site
    N_p: NK cells concentration at primary site (cells/L)
    L_p: CD8+T cells concentration at primary site (cells/L)
    C: Lymphocytes concentration in blood (cells/L)
    T_s: Tumor cell population at secondary site (metastasis)
    N_s: NK cells concentration at secondary site
    L_s: CD8+T cells concentration at secondary site

Action Space (2D continuous):
    D: Radiation dose effect (0-10 Gy)
    v_M: Chemotherapy concentration (0-8 mg/L)

Settings (increasing difficulty):
    Setting 1: No noise (deterministic)
    Setting 2: PKPD noise only
    Setting 3: Low observation/state noise + PKPD noise
    Setting 4: High observation/state noise + PKPD noise (default)
    Setting 5: High noise + missing observations
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

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
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

# Try to import DTRGym
try:
    import DTRGym  # This registers the environments with gym
    DTRGYM_AVAILABLE = True
except ImportError:
    DTRGYM_AVAILABLE = False
    print("Warning: DTRGym not installed. Install with: pip install DTRGym")



# =============================================================================
# Environment Constants
# =============================================================================

# Action bounds
CANCER_ACT_LOW = np.array([0.0, 0.0], dtype=np.float32)  # [D_min, v_M_min]
CANCER_ACT_HIGH = np.array([10.0, 8.0], dtype=np.float32)  # [D_max, v_M_max]

# Observation bounds (log-transformed)
# Original: T_p, N_p, L_p, C, T_s, N_s, L_s with ranges up to 1e11
CANCER_OBS_LOW = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
CANCER_OBS_HIGH = torch.tensor(
    [np.log(1e11), np.log(1e10), np.log(1e10), np.log(1e11), np.log(1e11), np.log(1e10), np.log(1e10)],
    dtype=torch.float32
)

# State dimension
OBS_DIM = 7
ACT_DIM = 2

_DEFINED_WANDB_METRICS: Set[str] = set()
_DEFINED_WANDB_STEP_KEYS: Set[str] = set()


# =============================================================================
# Utility Functions
# =============================================================================

def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cancer_reward_np(state: np.ndarray, action: np.ndarray, init_state: np.ndarray) -> float:
    """
    Compute reward for cancer treatment.
    
    Reward is based on tumor reduction relative to initial state.
    Log-transformed states are expected.
    
    Args:
        state: Current observation [T_p, N_p, L_p, C, T_s, N_s, L_s] (log-transformed)
        action: Action [D, v_M]
        init_state: Initial observation (log-transformed)
    
    Returns:
        Reward value
    """
    # Extract tumor populations (already log-transformed in observation)
    T_p, T_s = state[0], state[4]
    T_p0, T_s0 = init_state[0], init_state[4]
    
    # Total tumor (in log space, this is log-sum-exp approximation)
    # For simplicity, use sum of log values as proxy
    T_log = T_p + T_s
    T0_log = T_p0 + T_s0
    
    # Reward: tumor reduction
    # Positive when current tumor < initial tumor
    tumor_reduction = 1 - (T_log / max(T0_log, 1e-6))
    
    return float(tumor_reduction)


def cancer_reward_torch(
    states: torch.Tensor,
    actions: torch.Tensor,
    init_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Torch version of cancer reward matching the environment exactly.
    
    Reward = tumor_reduction + termination_bonus
    - tumor_reduction = 1 - T/T0 (in log space)
    - positive termination (+100): T_p < 1 AND T_s < 1 (tumor eliminated)
    - negative termination (-100): state out of bounds
    
    Args:
        states: Batch of observations [B, 7] (log-transformed)
        actions: Batch of actions [B, 2]
        init_states: Initial states for computing tumor reduction [B, 7]
                    If None, uses a default initial tumor burden
    
    Returns:
        Batch of rewards [B]
    """
    # Extract tumor populations (log-transformed)
    # Observations are log-transformed and clipped to [0, high]
    T_p_log = states[:, 0]  # Primary tumor
    T_s_log = states[:, 4]  # Secondary tumor (metastasis)
    
    if init_states is not None:
        T_p0_log = init_states[:, 0]
        T_s0_log = init_states[:, 4]
    else:
        # Default initial tumor: log(1e7) ≈ 16.1 for primary, 0 for secondary
        T_p0_log = torch.full_like(T_p_log, np.log(1e7))
        T_s0_log = torch.zeros_like(T_s_log)
    
    # Total tumor in log space (use max to avoid division by zero)
    T_log = T_p_log + T_s_log
    T0_log = torch.clamp(T_p0_log + T_s0_log, min=1.0)
    
    # Tumor reduction reward: positive when tumor decreases
    tumor_reduction = 1.0 - (T_log / T0_log)
    
    return tumor_reduction


def check_termination_torch(
    states: torch.Tensor,
    obs_high: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Check for termination conditions in batch of states.
    
    Args:
        states: Batch of observations [B, 7] (log-transformed)
        obs_high: Upper bounds for observations [7]
    
    Returns:
        positive_terminated: [B] bool tensor - tumor eliminated
        negative_terminated: [B] bool tensor - state exploded
    """
    # Extract tumor populations (log-transformed)
    T_p_log = states[:, 0]
    T_s_log = states[:, 4]
    
    # Positive termination: both tumors < 1 cell
    # In log space: log(1) = 0, but observations are clipped to >= 0
    # So we check if tumor is very small (near the lower bound)
    tumor_threshold = 0.1  # Approximately log(1.1)
    positive_terminated = (T_p_log < tumor_threshold) & (T_s_log < tumor_threshold)
    
    # Negative termination: state out of bounds (exploded)
    # Check if any state dimension exceeds bounds
    obs_high_expanded = obs_high.to(states.device).unsqueeze(0)
    negative_terminated = (states >= obs_high_expanded - 0.1).any(dim=1)
    
    return positive_terminated, negative_terminated


def cancer_reward_with_termination_torch(
    states: torch.Tensor,
    actions: torch.Tensor,
    init_states: Optional[torch.Tensor] = None,
    obs_high: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute reward including termination bonuses.
    
    Args:
        states: Current states [B, 7]
        actions: Actions [B, 2]
        init_states: Initial states [B, 7] or None
        obs_high: Upper bounds for checking termination
    
    Returns:
        rewards: [B] rewards including termination bonuses
        done: [B] boolean tensor indicating episode termination
    """
    # Base reward (tumor reduction)
    reward = cancer_reward_torch(states, actions, init_states)
    
    if obs_high is None:
        obs_high = CANCER_OBS_HIGH
    
    # Check termination
    positive_term, negative_term = check_termination_torch(states, obs_high)
    
    # Add termination bonuses
    reward = reward + positive_term.float() * 100.0
    reward = reward + negative_term.float() * (-100.0)
    
    # Episode is done if either termination condition is met
    done = positive_term | negative_term
    
    return reward, done


# =============================================================================
# Environment Factory
# =============================================================================

def create_cancer_env(
    max_t: int = 200,
    setting: int = 4,
    delayed_steps: int = 0,
    use_discrete: bool = False,
    n_act: int = 25,
) -> gym.Env:
    """
    Create a Ghaffari Cancer environment with specified settings.
    
    DTRGym registers environments as:
    - GhaffariCancerEnv-continuous
    - GhaffariCancerEnv-discrete
    
    Settings (passed via kwargs):
        1: No noise (deterministic)
        2: PKPD noise only (10%)
        3: Low obs/state noise + PKPD
        4: High obs/state noise + PKPD (default)
        5: High noise + 50% missing observations
    
    Args:
        max_t: Maximum timesteps (days)
        setting: Noise setting (1-5)
        delayed_steps: Steps between reward updates (0 for immediate)
        use_discrete: Whether to use discrete action wrapper
        n_act: Number of discrete actions (only if use_discrete=True)
    
    Returns:
        Gymnasium environment
    """
    if not DTRGYM_AVAILABLE:
        raise ImportError("DTRGym is required. Install with: pip install DTRGym")
    
    # Settings configuration for noise levels
    settings_config = {
        1: {"obs_noise": 0.0, "state_noise": 0.0, "pkpd_noise": 0.0, "missing_rate": 0.0},
        2: {"obs_noise": 0.0, "state_noise": 0.0, "pkpd_noise": 0.1, "missing_rate": 0.0},
        3: {"obs_noise": 0.1, "state_noise": 0.2, "pkpd_noise": 0.1, "missing_rate": 0.0},
        4: {"obs_noise": 0.2, "state_noise": 0.5, "pkpd_noise": 0.1, "missing_rate": 0.0},
        5: {"obs_noise": 0.2, "state_noise": 0.5, "pkpd_noise": 0.1, "missing_rate": 0.5},
    }
    
    if setting not in settings_config:
        raise ValueError(f"Setting must be 1-5, got {setting}")
    
    config = settings_config[setting]
    
    # Use gym.make with the registered environment name
    env_type = "discrete" if use_discrete else "continuous"
    env_id = f"GhaffariCancerEnv-{env_type}"
    
    # Pass configuration as kwargs
    env = gym.make(
        env_id,
        max_t=max_t,
        delayed_steps=delayed_steps,
        obs_noise=config["obs_noise"],
        state_noise=config["state_noise"],
        pkpd_noise=config["pkpd_noise"],
        missing_rate=config["missing_rate"],
    )
    
    return env


# Register the environment for gym.make compatibility
def _register_cancer_envs():
    """Register cancer environments with gymnasium."""
    if not DTRGYM_AVAILABLE:
        return
    
    for setting in range(1, 6):
        env_id = f"GhaffariCancer-v{setting}"
        if env_id not in gym.envs.registry:
            gym.register(
                id=env_id,
                entry_point="cancer_pipeline:create_cancer_env",
                kwargs={"setting": setting},
                max_episode_steps=200,
            )

_register_cancer_envs()


# =============================================================================
# Callbacks
# =============================================================================

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
    """Adapter to use SB3 PPO models with our TorchPolicy interface."""
    
    def __init__(
        self,
        name: str,
        model: PPO,
        batch_size: int = 2048,
        act_low: np.ndarray = CANCER_ACT_LOW,
        act_high: np.ndarray = CANCER_ACT_HIGH,
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
        act_low: Optional[np.ndarray] = None,
        act_high: Optional[np.ndarray] = None,
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
            
            # Clamp to action bounds
            low_tensor = torch.tensor(act_low, device=states.device, dtype=torch.float32)
            high_tensor = torch.tensor(act_high, device=states.device, dtype=torch.float32)
            return action_tensor.clamp(min=low_tensor, max=high_tensor)


def load_policy_models(policy_snapshots: Sequence[Dict[str, object]]) -> Dict[str, PPO]:
    models: Dict[str, PPO] = {}
    for snap in policy_snapshots:
        models[snap["name"]] = PPO.load(snap["path"])
    return models


def make_policy_adapters(policy_models: Dict[str, PPO]) -> Dict[str, TorchPolicy]:
    return {name: SB3PolicyAdapter(name, model) for name, model in policy_models.items()}


# =============================================================================
# Wandb Utilities
# =============================================================================

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
# Training Functions
# =============================================================================

def train_ppo_with_checkpoints(
    max_t: int,
    setting: int,
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
    """Train PPO on cancer environment with periodic checkpoints."""
    set_random_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    save_dir.mkdir(parents=True, exist_ok=True)

    def env_fn():
        return create_cancer_env(max_t=max_t, setting=setting)

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
    max_t: int,
    setting: int,
    total_steps: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]]:
    """Rollout policy and collect transitions."""
    env = create_cancer_env(max_t=max_t, setting=setting)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    init_obs = obs.copy()
    initial_states: List[np.ndarray] = [obs.copy()]
    transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
    steps = 0
    
    while steps < total_steps:
        action, _ = model.predict(obs, deterministic=False)
        action = np.asarray(action, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)
        
        # Use environment reward directly
        transitions.append((obs.copy(), action.copy(), float(reward), next_obs.copy(), done))
        obs = next_obs
        steps += 1
        
        if done:
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            init_obs = obs.copy()
            initial_states.append(obs.copy())
    
    return initial_states, transitions


def build_offline_dataset(
    policy_snapshots: Sequence[Dict[str, object]],
    policy_models: Dict[str, PPO],
    max_t: int,
    setting: int,
    steps_per_policy: int,
    seed: int,
) -> OfflineDataset:
    """Build offline dataset by rolling out multiple policies."""
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_next_states: List[np.ndarray] = []
    all_dones: List[float] = []
    initial_states: List[np.ndarray] = []

    for idx, snap in enumerate(policy_snapshots):
        model = policy_models[snap["name"]]
        init_states, transitions = rollout_policy(
            model, max_t, setting, steps_per_policy, seed + idx
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
    hidden_dim: int = 256,
    backbone: str = "mlp",
    wandb_run: Optional[Any] = None,
) -> Tuple[DynamicsNet, Dict[str, DynamicsNet]]:
    """Train supervised and ranking-aware dynamics models."""
    state_dim = dataset.states.shape[1]  # 7
    act_dim = dataset.actions.shape[1]   # 2

    # Supervised dynamics
    sup_model = DynamicsNet(
        state_dim=state_dim,
        act_dim=act_dim,
        state_low=CANCER_OBS_LOW,
        state_upper=CANCER_OBS_HIGH,
        wrapped_dims=[],  # No wrapped dims in this env
        hidden=hidden_dim,
        backbone=backbone,
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
            state_low=CANCER_OBS_LOW,
            state_upper=CANCER_OBS_HIGH,
            wrapped_dims=[],
            hidden=hidden_dim,
            backbone=backbone,
        ).to(device)
        model.train_ranking_aware_model(
            dataset,
            policy_q_pairs=policy_q_pairs,
            gamma=gamma,
            lambda_rank=lambda_rank,
            epochs=dyn_epochs,
            batch_size=dyn_batch,
            lr=dyn_lr,
            reward_fn=cancer_reward_torch,
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
# Evaluation Functions
# =============================================================================

def evaluate_sb3_policy_fixed_horizon(
    model: PPO,
    max_t: int,
    setting: int,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
) -> float:
    """
    Evaluate policy in true environment with fixed horizon.
    
    Properly handles early termination by stopping the episode
    (no reset and continue - that would be wrong for value estimation).
    """
    env = create_cancer_env(max_t=max_t, setting=setting)
    returns: List[float] = []
    
    for ep in range(rollouts):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        discount = 1.0
        
        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total += discount * reward
            discount *= gamma
            obs = next_obs
            
            # Stop episode on termination (don't reset and continue)
            if terminated or truncated:
                break
        
        returns.append(total)
    
    return float(np.mean(returns))


def evaluate_q_estimate(
    q_model: QNet,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    samples: int,
) -> float:
    """Estimate value using Q-network."""
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
    """
    Evaluate policy using learned dynamics model.
    
    Properly handles termination conditions to match true environment evaluation:
    - Positive termination (+100): tumor eliminated
    - Negative termination (-100): state exploded
    - Episodes stop accumulating rewards after termination
    """
    if initial_states.shape[0] == 0:
        raise ValueError("No initial states available for dynamics evaluation.")
    
    idx = np.random.choice(initial_states.shape[0], size=rollouts, replace=initial_states.shape[0] < rollouts)
    states = torch.tensor(initial_states[idx], dtype=torch.float32, device=device)
    init_states = states.clone()  # Save for reward computation
    
    total = torch.zeros(states.size(0), device=device)
    discount = torch.ones_like(total)
    active = torch.ones(states.size(0), dtype=torch.bool, device=device)  # Track active episodes
    
    obs_high = CANCER_OBS_HIGH.to(device)
    
    for _ in range(horizon):
        # Only compute for active episodes
        if not active.any():
            break
            
        actions = policy.sample_torch_actions(states, deterministic=True)
        
        # Get rewards with termination bonuses
        rewards, done = cancer_reward_with_termination_torch(
            states, actions, init_states, obs_high
        )
        
        # Only add rewards for still-active episodes
        total += discount * rewards * active.float()
        
        # Update discount for next step
        discount *= gamma
        
        # Predict next states
        next_states = dynamics.sample_next(states, actions, deterministic=True)
        
        # Deactivate terminated episodes
        active = active & ~done
        
        # Update states (only matters for active episodes)
        states = next_states
    
    return float(total.mean().item())


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_dataset_statistics(dataset: OfflineDataset) -> Dict[str, Any]:
    """Compute statistics about the offline dataset."""
    stats = {
        "num_transitions": len(dataset.states),
        "num_initial_states": len(dataset.initial_states),
        "state_dim": dataset.states.shape[1],
        "action_dim": dataset.actions.shape[1],
        "state_mean": dataset.states.mean(axis=0).tolist(),
        "state_std": dataset.states.std(axis=0).tolist(),
        "action_mean": dataset.actions.mean(axis=0).tolist(),
        "action_std": dataset.actions.std(axis=0).tolist(),
        "reward_mean": float(dataset.rewards.mean()),
        "reward_std": float(dataset.rewards.std()),
        "reward_min": float(dataset.rewards.min()),
        "reward_max": float(dataset.rewards.max()),
        "done_rate": float(dataset.dones.mean()),
    }
    return stats


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cancer treatment pipeline for OPE with ranking-aware dynamics"
    )
    
    # Environment settings
    parser.add_argument("--max-t", type=int, default=200, help="Max timesteps per episode (days)")
    parser.add_argument("--setting", type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Noise setting (1=none, 5=max noise+missing)")
    
    # PPO training
    parser.add_argument("--total-steps", type=int, default=10_000)
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
    
    # Dataset collection
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-steps", type=int, default=10_000,
                        help="Steps per policy for dataset collection")
    
    # Q-network training
    parser.add_argument("--q-epochs", type=int, default=500)
    parser.add_argument("--q-batch", type=int, default=512)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--q-samples", type=int, default=32)
    
    # Dynamics training
    parser.add_argument("--dyn-epochs", type=int, default=2000)
    parser.add_argument("--dyn-batch", type=int, default=512)
    parser.add_argument("--dyn-lr", type=float, default=3e-4)
    parser.add_argument("--dyn-hidden-dim", type=int, default=256)
    parser.add_argument("--dyn-val-fraction", type=float, default=0.1)
    parser.add_argument("--dyn-early-stop-patience", type=int, default=50)
    parser.add_argument("--dyn-min-epochs", type=int, default=0)
    parser.add_argument("--dynamics-loss", type=str, default="nll", choices=["nll", "mse"],
                        help="Loss function for dynamics training")
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--backbone", type=str, default="mlp", choices=["mlp", "resnet", "ode", "transformer", "gru"],)
    
    # Evaluation
    parser.add_argument("--eval-rollouts", type=int, default=200)
    parser.add_argument("--eval-horizon", type=int, default=200)
    
    # Output and logging
    parser.add_argument("--output-dir", type=Path, default=Path("results/cancer_pipeline"))
    parser.add_argument("--device", type=str, default="auto")
    
    # Flags to skip/force stages
    parser.add_argument("--force-policy-training", action="store_true")
    parser.add_argument("--force-dataset-collection", action="store_true")
    parser.add_argument("--force-q-training", action="store_true")
    parser.add_argument("--force-dynamics-training", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="DT2-cancer")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    
    args = parser.parse_args()

    if args.results_only and (
        args.force_policy_training or args.force_dataset_collection or
        args.force_q_training or args.force_dynamics_training
    ):
        parser.error("--results-only cannot be combined with force-training flags.")

    if not DTRGYM_AVAILABLE:
        print("ERROR: DTRGym is required for this pipeline.")
        print("Install with: pip install DTRGym")
        return

    device = torch.device(args.device) if args.device != "auto" else default_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(args)

    # Log environment info
    wandb_log(wandb_run, {
        "env/name": "GhaffariCancerEnv",
        "env/obs_dim": OBS_DIM,
        "env/act_dim": ACT_DIM,
        "env/setting": args.setting,
        "env/max_t": args.max_t,
    })

    print(f"=== Cancer Treatment Pipeline ===")
    print(f"Environment: GhaffariCancerEnv (Setting {args.setting})")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print()

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
            max_t=args.max_t,
            setting=args.setting,
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


    # Step 2: Build offline dataset
    dataset_path = args.output_dir / "offline_dataset.npz"
    dataset_loaded = False

    if dataset_path.exists() and not args.force_policy_training and not args.force_dataset_collection:
        print("[Step 2] Loading existing offline dataset from disk...")
        dataset = load_offline_dataset(dataset_path)
        dataset_loaded = True
    else:
        if args.results_only:
            raise FileNotFoundError(f"No saved offline dataset at {dataset_path}.")
        print("[Step 2] Collecting offline dataset...")
        dataset = build_offline_dataset(
            snapshots, policy_models, args.max_t, args.setting,
            args.rollout_steps, args.seed
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

    # Log dataset statistics
    dataset_stats = analyze_dataset_statistics(dataset)
    wandb_log(wandb_run, {
        "dataset/transitions": dataset_stats["num_transitions"],
        "dataset/initial_states": dataset_stats["num_initial_states"],
        "dataset/loaded_from_disk": int(dataset_loaded),
        "dataset/reward_mean": dataset_stats["reward_mean"],
        "dataset/reward_std": dataset_stats["reward_std"],
    })


    # Step 2b: Generate a small test set of transitions for dynamics MSE evaluation
    def generate_dynamics_test_set_with_policies(env_setting, n_transitions_per_policy=200, seed=0):
        env = create_cancer_env(max_t=args.max_t, setting=env_setting)
        rng = np.random.default_rng(seed)
        all_states = []
        all_actions = []
        all_next_states = []
        for policy_name, policy in torch_policies.items():
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            for _ in range(n_transitions_per_policy):
                # Use policy to select action
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_tensor = policy.sample_torch_actions(obs_tensor, deterministic=True)
                action = action_tensor.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                all_states.append(obs.copy())
                all_actions.append(action.copy())
                all_next_states.append(next_obs.copy())
                obs = next_obs
                if terminated or truncated:
                    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        return (
            np.asarray(all_states, dtype=np.float32),
            np.asarray(all_actions, dtype=np.float32),
            np.asarray(all_next_states, dtype=np.float32),
        )

    # torch_policies must be defined before calling this
    torch_policies = make_policy_adapters(policy_models)
    test_states, test_actions, test_next_states = generate_dynamics_test_set_with_policies(args.setting, n_transitions_per_policy=200, seed=args.seed+999)

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
    dynamics_dir = args.output_dir / args.backbone / "dynamics"
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
            hidden_dim=args.dyn_hidden_dim,
            backbone=args.backbone,
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

    # Helper: MSE for dynamics model
    def calc_dynamics_mse(dynamics_model, states, actions, next_states, device):
        with torch.no_grad():
            s = torch.tensor(states, dtype=torch.float32, device=device)
            a = torch.tensor(actions, dtype=torch.float32, device=device)
            s_next_true = torch.tensor(next_states, dtype=torch.float32, device=device)
            s_next_pred = dynamics_model.sample_next(s, a, deterministic=True)
            mse = torch.mean((s_next_pred - s_next_true) ** 2).item()
        return mse

    # Helper: Sample a batch of transitions from the training dataset
    def sample_training_transitions(dataset, n_samples=1000, seed=0):
        rng = np.random.default_rng(seed)
        n_total = dataset.states.shape[0]
        idx = rng.choice(n_total, size=min(n_samples, n_total), replace=n_samples > n_total)
        return dataset.states[idx], dataset.actions[idx], dataset.next_states[idx]

    # Evaluate and log
    # Sample a batch of training transitions for MSE eval
    train_states, train_actions, train_next_states = sample_training_transitions(dataset, n_samples=1000, seed=args.seed+123)

    for snap in snapshots:
        name = snap["name"]
        policy = torch_policies[name]
        ppo_model = policy_models[name]
        q_net = q_models[name]

        # Q-network estimate
        q_est = evaluate_q_estimate(q_net, policy, initial_states, args.eval_rollouts)

        # Supervised dynamics estimate
        dyn_sup = evaluate_in_dynamics(
            sup_model, policy, initial_states, args.eval_horizon, args.gamma, device, args.eval_rollouts
        )

        # Supervised dynamics MSE (test set)
        dyn_sup_mse = calc_dynamics_mse(sup_model, test_states, test_actions, test_next_states, device)
        # Supervised dynamics MSE (training set)
        dyn_sup_train_mse = calc_dynamics_mse(sup_model, train_states, train_actions, train_next_states, device)

        # Ranking-aware dynamics estimates and MSEs
        dyn_rank_new: Dict[str, float] = {}
        dyn_rank_new_mse: Dict[str, float] = {}
        dyn_rank_new_train_mse: Dict[str, float] = {}
        for loss_name, dyn_model in ranking_new_models.items():
            dyn_rank_new[loss_name] = evaluate_in_dynamics(
                dyn_model, policy, initial_states, args.eval_horizon, args.gamma, device, args.eval_rollouts
            )
            dyn_rank_new_mse[loss_name] = calc_dynamics_mse(dyn_model, test_states, test_actions, test_next_states, device)
            dyn_rank_new_train_mse[loss_name] = calc_dynamics_mse(dyn_model, train_states, train_actions, train_next_states, device)

        # True environment estimate
        env_mc = evaluate_sb3_policy_fixed_horizon(
            ppo_model, args.max_t, args.setting,
            args.eval_horizon, args.gamma, args.eval_rollouts, args.seed
        )

        wandb_payload = {
            "eval/policy_name": name,
            "eval/q_estimate": q_est,
            "eval/dynamics_supervised": dyn_sup,
            "eval/dynamics_supervised_mse": dyn_sup_mse,
            "eval/dynamics_supervised_train_mse": dyn_sup_train_mse,
            "eval/env_mc": env_mc,
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
            "env_mc": env_mc,
        })

        print(f"  {name}: env_mc={env_mc:.2f}, q_est={q_est:.2f}, dyn_sup={dyn_sup:.2f}, dyn_sup_mse={dyn_sup_mse:.4f}, dyn_sup_train_mse={dyn_sup_train_mse:.4f}")

    # Save summary
    config = args_to_config(args)
    summary = {
        "config": config,
        "dataset_stats": dataset_stats,
        "num_transitions": int(len(dataset.states)),
        "q_model_paths": q_model_paths,
        "dynamics_paths": dynamics_paths,
        "results": results,
    }

    summary_path = args.output_dir / args.backbone / f"summary_{args.seed}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    if wandb_run is not None:
        wandb_run.summary.update({
            "num_transitions": int(len(dataset.states)),
            "results": results,
        })
        wandb_run.finish()

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
