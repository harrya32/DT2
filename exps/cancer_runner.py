"""
Cancer Treatment (Ghaffari) pipeline using predefined expert clinical policies.

This pipeline evaluates Off-Policy Evaluation (OPE) methods using interpretable
clinical treatment strategies instead of PPO-trained policies.

Expert Policies:
- NoTreatment: Baseline with no intervention
- StandardFractionatedRadio (SFR): 2 Gy/day radiotherapy, 5 days/week
- MetronomicChemo: Continuous low-dose chemotherapy
- AdaptiveTherapy: Treatment holidays based on tumor burden
- AggressiveMTD: Maximum tolerated dose of both modalities

The Cancer environment from DTRGym models combined radiotherapy and chemotherapy
for cancer with metastasis. Key features:
- 7D continuous observation space (log-transformed tumor/immune populations)
- 2D continuous action space (radiation dose + chemotherapy concentration)
- ODE-based dynamics with configurable noise levels

Reference:
"A mixed radiotherapy and chemotherapy model for treatment of cancer with metastasis"
https://onlinelibrary.wiley.com/doi/full/10.1002/mma.3887
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from base_pipeline import (
    add_common_args,
    args_to_config,
    default_device,
    initialize_wandb,
    wandb_log,
    make_epoch_logger,
    FractionCheckpointCallback,
    WandbMetricsCallback,
    SB3PolicyAdapter,
    load_snapshots,
    save_snapshots,
    load_policy_models,
    load_q_models,
    save_q_models,
    load_dynamics_models,
    save_dynamics_models,
    load_offline_dataset,
    make_sequence_dataset,
    train_q_networks,
    evaluate_q_estimate,
    sample_training_transitions,
)
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

# Action bounds for cancer treatment
CANCER_ACT_LOW = np.array([0.0, 0.0], dtype=np.float32)  # [D_min, v_M_min]
CANCER_ACT_HIGH = np.array([10.0, 8.0], dtype=np.float32)  # [D_max, v_M_max]


# =============================================================================
# Expert Clinical Policies
# =============================================================================

class ExpertPolicy:
    """Base class for expert clinical policies."""
    def __init__(self, name: str):
        self.name = name

    def get_action(self, obs: np.ndarray, time_step: int) -> np.ndarray:
        raise NotImplementedError


class NoTreatment(ExpertPolicy):
    """Baseline: No treatment at all."""
    def __init__(self):
        super().__init__("NoTreatment")
    
    def get_action(self, obs: np.ndarray, time_step: int) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=np.float32)


class StandardFractionatedRadio(ExpertPolicy):
    """
    Standard Fractionated Radiotherapy: 2 Gy/day, 5 days/week.
    This is a classic clinical radiotherapy protocol.
    """
    def __init__(self):
        super().__init__("SFR_2Gy_5d")
    
    def get_action(self, obs: np.ndarray, time_step: int) -> np.ndarray:
        # Schedule: 5 days on, 2 days off (weekends)
        day_of_week = time_step % 7
        if day_of_week < 5:
            return np.array([2.0, 0.0], dtype=np.float32)  # 2 Gy Radio, 0 Chemo
        else:
            return np.array([0.0, 0.0], dtype=np.float32)


class MetronomicChemo(ExpertPolicy):
    """
    Metronomic Chemotherapy: Continuous low-dose chemotherapy.
    Known for anti-angiogenic effects and reduced toxicity.
    """
    def __init__(self, dose: float = 2.0):
        super().__init__(f"MetronomicChemo_{dose}")
        self.dose = dose
    
    def get_action(self, obs: np.ndarray, time_step: int) -> np.ndarray:
        return np.array([0.0, self.dose], dtype=np.float32)


class AdaptiveTherapy(ExpertPolicy):
    """
    Adaptive Therapy: Treat only when tumor exceeds threshold.
    Inspired by evolutionary game theory approaches to cancer treatment.
    """
    def __init__(self, threshold_fraction: float = 0.5, init_tumor_size: float = 1e7):
        super().__init__("AdaptiveTherapy")
        # Treat if tumor grows beyond threshold_fraction of initial size
        self.threshold = np.log(init_tumor_size * threshold_fraction)
        
    def get_action(self, obs: np.ndarray, time_step: int) -> np.ndarray:
        # obs[0] is T_p (primary tumor) in log scale
        tumor_size_log = obs[0]
        
        # If tumor is large, use combination therapy
        if tumor_size_log > self.threshold:
            return np.array([2.0, 4.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)


class AggressiveMTD(ExpertPolicy):
    """
    Aggressive Maximum Tolerated Dose: High doses of both modalities.
    Traditional approach aiming for maximum tumor kill.
    """
    def __init__(self):
        super().__init__("AggressiveMTD")
    
    def get_action(self, obs: np.ndarray, time_step: int) -> np.ndarray:
        return np.array([4.0, 6.0], dtype=np.float32)


def get_clinical_policies() -> List[ExpertPolicy]:
    """Get all predefined clinical expert policies."""
    return [
        NoTreatment(),
        StandardFractionatedRadio(),
        MetronomicChemo(dose=2.0),
        AdaptiveTherapy(),
        AggressiveMTD(),
    ]

# Observation bounds (log-transformed)
CANCER_OBS_LOW = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
CANCER_OBS_HIGH = torch.tensor(
    [np.log(1e11), np.log(1e10), np.log(1e10), np.log(1e11), np.log(1e11), np.log(1e10), np.log(1e10)],
    dtype=torch.float32
)

OBS_DIM = 7
ACT_DIM = 2


# =============================================================================
# Environment Factory
# =============================================================================

def create_cancer_env(
    max_t: int = 200,
    setting: int = 4,
    delayed_steps: int = 0,
) -> gym.Env:
    """
    Create a Ghaffari Cancer environment with specified settings.
    
    Settings (noise levels):
        1: No noise (deterministic)
        2: PKPD noise only (10%)
        3: Low obs/state noise + PKPD
        4: High obs/state noise + PKPD
        5: High noise + 50% missing observations
    
    Args:
        max_t: Maximum timesteps per episode (days)
        setting: Noise setting (1-5)
        delayed_steps: Steps between reward updates (0 for immediate)
    
    Returns:
        Gymnasium environment
    """
    if not DTRGYM_AVAILABLE:
        raise ImportError("DTRGym is required. Install with: pip install DTRGym")
    
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
    
    env = gym.make(
        "GhaffariCancerEnv-continuous",
        max_t=max_t,
        delayed_steps=delayed_steps,
        obs_noise=config["obs_noise"],
        state_noise=config["state_noise"],
        pkpd_noise=config["pkpd_noise"],
        missing_rate=config["missing_rate"],
    )
    
    return env


# =============================================================================
# Reward Functions
# =============================================================================

def cancer_reward_np(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute reward for cancer treatment (NumPy version).
    
    Reward is based on tumor reduction. This is a simplified version
    that uses a fixed reference for the initial tumor.
    
    Args:
        state: Current observation [T_p, N_p, L_p, C, T_s, N_s, L_s] (log-transformed)
        action: Action [D, v_M]
    
    Returns:
        Reward value
    """
    # Extract tumor populations (log-transformed)
    T_p, T_s = state[0], state[4]
    
    # Use default initial tumor: log(1e7) ≈ 16.1 for primary
    T_p0_log = np.log(1e7)
    T_s0_log = 0.0
    
    T_log = T_p + T_s
    T0_log = max(T_p0_log + T_s0_log, 1.0)
    
    tumor_reduction = 1.0 - (T_log / T0_log)
    return float(tumor_reduction)


def cancer_reward_torch(
    states: torch.Tensor,
    actions: torch.Tensor,
    init_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Torch version of cancer reward.
    
    Args:
        states: Batch of observations [B, 7] (log-transformed)
        actions: Batch of actions [B, 2]
        init_states: Initial states for computing tumor reduction [B, 7]
                    If None, uses a default initial tumor burden
    
    Returns:
        Batch of rewards [B]
    """
    T_p_log = states[:, 0]
    T_s_log = states[:, 4]
    
    if init_states is not None:
        T_p0_log = init_states[:, 0]
        T_s0_log = init_states[:, 4]
    else:
        T_p0_log = torch.full_like(T_p_log, np.log(1e7))
        T_s0_log = torch.zeros_like(T_s_log)
    
    T_log = T_p_log + T_s_log
    T0_log = torch.clamp(T_p0_log + T_s0_log, min=1.0)
    
    tumor_reduction = 1.0 - (T_log / T0_log)
    return tumor_reduction


def check_termination_torch(
    states: torch.Tensor,
    obs_high: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Check for termination conditions in batch of states."""
    T_p_log = states[:, 0]
    T_s_log = states[:, 4]
    
    # Positive termination: both tumors < 1 cell
    tumor_threshold = 0.1
    positive_terminated = (T_p_log < tumor_threshold) & (T_s_log < tumor_threshold)
    
    # Negative termination: state out of bounds
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
    
    Returns:
        rewards: [B] rewards including termination bonuses
        done: [B] boolean tensor indicating episode termination
    """
    reward = cancer_reward_torch(states, actions, init_states)
    
    if obs_high is None:
        obs_high = CANCER_OBS_HIGH
    
    positive_term, negative_term = check_termination_torch(states, obs_high)
    
    reward = reward + positive_term.float() * 100.0
    reward = reward + negative_term.float() * (-100.0)
    
    done = positive_term | negative_term
    
    return reward, done


# =============================================================================
# Cancer-specific Policy Adapters
# =============================================================================

class ExpertPolicyAdapter(TorchPolicy):
    """
    Adapter that wraps ExpertPolicy to conform to TorchPolicy interface.
    
    Since expert policies may depend on time_step, we track a virtual time
    that resets for each batch of initial states.
    """
    
    def __init__(
        self,
        expert: ExpertPolicy,
        act_low: np.ndarray = CANCER_ACT_LOW,
        act_high: np.ndarray = CANCER_ACT_HIGH,
    ):
        self.expert = expert
        self.name = expert.name
        self.act_low_arr = act_low
        self.act_high_arr = act_high
        self.action_dim = ACT_DIM
        self._time_step = 0  # Track virtual time for time-dependent policies

    def reset_time(self):
        """Reset the virtual time step counter."""
        self._time_step = 0

    def step_time(self):
        """Increment the virtual time step."""
        self._time_step += 1

    def sample_torch_actions(
        self,
        states: torch.Tensor,
        repeats: int = 1,
        deterministic: bool = False,
        act_low: Optional[np.ndarray] = None,
        act_high: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Sample actions for a batch of states.
        
        Note: For time-dependent policies, all states in the batch get the same
        time_step. Use reset_time() and step_time() to manage rollouts.
        """
        if act_low is None:
            act_low = self.act_low_arr
        if act_high is None:
            act_high = self.act_high_arr
            
        base = states
        if repeats > 1:
            base = states.repeat_interleave(repeats, dim=0)
        
        obs_np = base.detach().cpu().numpy()
        batch_size = obs_np.shape[0]
        
        # Get actions for each observation
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        for i in range(batch_size):
            actions[i] = self.expert.get_action(obs_np[i], self._time_step)
        
        action_tensor = torch.tensor(actions, device=states.device, dtype=torch.float32)
        
        low_tensor = torch.tensor(act_low, device=states.device, dtype=torch.float32)
        high_tensor = torch.tensor(act_high, device=states.device, dtype=torch.float32)
        return action_tensor.clamp(min=low_tensor, max=high_tensor)


class CancerPolicyAdapter(SB3PolicyAdapter):
    """Adapter for cancer environment with array-based action bounds (for PPO models)."""
    
    def __init__(
        self,
        name: str,
        model: PPO,
        batch_size: int = 2048,
        act_low: np.ndarray = CANCER_ACT_LOW,
        act_high: np.ndarray = CANCER_ACT_HIGH,
    ):
        # Call parent with scalar bounds (will be overridden)
        super().__init__(name, model, batch_size, act_low=0.0, act_high=1.0)
        self.act_low_arr = act_low
        self.act_high_arr = act_high

    def sample_torch_actions(
        self,
        states: torch.Tensor,
        repeats: int = 1,
        deterministic: bool = False,
        act_low: Optional[np.ndarray] = None,
        act_high: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        if act_low is None:
            act_low = self.act_low_arr
        if act_high is None:
            act_high = self.act_high_arr
            
        with torch.no_grad():
            base = states
            if repeats > 1:
                base = states.repeat_interleave(repeats, dim=0)
            obs = base.detach().to("cpu").to(torch.float32)
            obs_np = obs.numpy()
            np.nan_to_num(obs_np, copy=False)
            
            if obs.shape[0] == 0:
                return torch.zeros((0, self.action_dim), device=states.device)
            
            import math
            chunks = max(1, math.ceil(obs_np.shape[0] / self.batch_size))
            actions = []
            for chunk in np.array_split(obs_np, chunks):
                if chunk.size == 0:
                    continue
                act, _ = self.model.predict(chunk, deterministic=deterministic)
                actions.append(np.asarray(act, dtype=np.float32))
            
            action_np = np.concatenate(actions, axis=0)
            action_tensor = torch.tensor(action_np, device=states.device, dtype=torch.float32)
            
            low_tensor = torch.tensor(act_low, device=states.device, dtype=torch.float32)
            high_tensor = torch.tensor(act_high, device=states.device, dtype=torch.float32)
            return action_tensor.clamp(min=low_tensor, max=high_tensor)


def make_cancer_policy_adapters(policy_models: Dict[str, PPO]) -> Dict[str, TorchPolicy]:
    """Create CancerPolicyAdapter instances for each loaded PPO model."""
    return {name: CancerPolicyAdapter(name, model) for name, model in policy_models.items()}


def make_expert_policy_adapters(experts: List[ExpertPolicy]) -> Dict[str, ExpertPolicyAdapter]:
    """Create ExpertPolicyAdapter instances for each expert policy."""
    return {expert.name: ExpertPolicyAdapter(expert) for expert in experts}


# =============================================================================
# Cancer-specific Training Functions
# =============================================================================

def train_ppo_cancer(
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
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback
    
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


def rollout_cancer_policy(
    model: PPO,
    max_t: int,
    setting: int,
    total_steps: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]]:
    """Rollout PPO policy and collect transitions."""
    env = create_cancer_env(max_t=max_t, setting=setting)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    initial_states: List[np.ndarray] = [obs.copy()]
    transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
    steps = 0
    
    while steps < total_steps:
        action, _ = model.predict(obs, deterministic=False)
        action = np.asarray(action, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)
        
        transitions.append((obs.copy(), action.copy(), float(reward), next_obs.copy(), done))
        obs = next_obs
        steps += 1
        
        if done:
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            initial_states.append(obs.copy())
    
    env.close()
    return initial_states, transitions


def rollout_expert_policy(
    expert: ExpertPolicy,
    max_t: int,
    setting: int,
    total_steps: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]]]:
    """Rollout expert policy and collect transitions."""
    env = create_cancer_env(max_t=max_t, setting=setting)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    initial_states: List[np.ndarray] = [obs.copy()]
    transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
    steps = 0
    time_step = 0
    
    while steps < total_steps:
        action = expert.get_action(obs, time_step)
        action = np.asarray(action, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)
        
        transitions.append((obs.copy(), action.copy(), float(reward), next_obs.copy(), done))
        obs = next_obs
        steps += 1
        time_step += 1
        
        if done:
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            initial_states.append(obs.copy())
            time_step = 0  # Reset time step for new episode
    
    env.close()
    return initial_states, transitions


def build_cancer_offline_dataset(
    policy_snapshots: Sequence[Dict[str, object]],
    policy_models: Dict[str, PPO],
    max_t: int,
    setting: int,
    steps_per_policy: int,
    seed: int,
) -> OfflineDataset:
    """Build offline dataset by rolling out multiple PPO policies."""
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_next_states: List[np.ndarray] = []
    all_dones: List[float] = []
    initial_states: List[np.ndarray] = []

    for idx, snap in enumerate(policy_snapshots):
        model = policy_models[snap["name"]]
        init_states, transitions = rollout_cancer_policy(
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


def build_expert_offline_dataset(
    experts: List[ExpertPolicy],
    max_t: int,
    setting: int,
    steps_per_policy: int,
    seed: int,
) -> OfflineDataset:
    """Build offline dataset by rolling out expert policies."""
    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_next_states: List[np.ndarray] = []
    all_dones: List[float] = []
    initial_states: List[np.ndarray] = []

    for idx, expert in enumerate(experts):
        print(f"  Rolling out {expert.name}...")
        init_states, transitions = rollout_expert_policy(
            expert, max_t, setting, steps_per_policy, seed + idx * 1000
        )
        initial_states.extend(init_states)
        for s, a, r, sn, d in transitions:
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)
            all_next_states.append(sn)
            all_dones.append(d)
        print(f"    Collected {len(transitions)} transitions, {len(init_states)} episodes")

    return OfflineDataset(
        states=np.asarray(all_states, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.float32),
        rewards=np.asarray(all_rewards, dtype=np.float32),
        next_states=np.asarray(all_next_states, dtype=np.float32),
        dones=np.asarray(all_dones, dtype=np.float32),
        initial_states=np.asarray(initial_states, dtype=np.float32),
    )


def train_cancer_dynamics_models(
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
    """Train supervised and ranking-aware dynamics models for cancer."""
    state_dim = dataset.states.shape[-1]
    act_dim = dataset.actions.shape[-1]

    # Supervised dynamics
    sup_model = DynamicsNet(
        state_dim=state_dim,
        act_dim=act_dim,
        state_low=CANCER_OBS_LOW,
        state_upper=CANCER_OBS_HIGH,
        wrapped_dims=[],
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


# =============================================================================
# Cancer-specific Evaluation Functions
# =============================================================================

def evaluate_cancer_in_dynamics(
    dynamics: DynamicsNet,
    policy: TorchPolicy,
    initial_states: np.ndarray,
    horizon: int,
    gamma: float,
    device: torch.device,
    rollouts: int,
) -> float:
    """
    Evaluate policy using learned dynamics model with termination handling.
    
    Properly handles termination conditions to match true environment evaluation:
    - Positive termination (+100): tumor eliminated
    - Negative termination (-100): state exploded
    - Episodes stop accumulating rewards after termination
    """
    if initial_states.shape[0] == 0:
        raise ValueError("No initial states available for dynamics evaluation.")
    
    idx = np.random.choice(initial_states.shape[0], size=rollouts, replace=initial_states.shape[0] < rollouts)
    states = torch.tensor(initial_states[idx], dtype=torch.float32, device=device)
    init_states = states.clone()
    
    total = torch.zeros(states.size(0), device=device)
    discount = torch.ones_like(total)
    active = torch.ones(states.size(0), dtype=torch.bool, device=device)
    
    obs_high = CANCER_OBS_HIGH.to(device)
    
    for _ in range(horizon):
        if not active.any():
            break
            
        actions = policy.sample_torch_actions(states, deterministic=True)
        
        rewards, done = cancer_reward_with_termination_torch(
            states, actions, init_states, obs_high
        )
        
        total += discount * rewards * active.float()
        discount *= gamma
        
        next_states = dynamics.sample_next(states, actions, deterministic=True)
        active = active & ~done
        states = next_states
    
    return float(total.mean().item())


def evaluate_cancer_sb3_policy(
    model: PPO,
    max_t: int,
    setting: int,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
) -> float:
    """Evaluate PPO policy in true environment with fixed horizon."""
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
            
            if terminated or truncated:
                break
        
        returns.append(total)
    
    env.close()
    return float(np.mean(returns))


def evaluate_expert_policy(
    expert: ExpertPolicy,
    max_t: int,
    setting: int,
    horizon: int,
    gamma: float,
    rollouts: int,
    seed: int,
) -> float:
    """Evaluate expert policy in true environment with fixed horizon."""
    env = create_cancer_env(max_t=max_t, setting=setting)
    returns: List[float] = []
    
    for ep in range(rollouts):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        discount = 1.0
        
        for t in range(horizon):
            action = expert.get_action(obs, t)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total += discount * reward
            discount *= gamma
            obs = next_obs
            
            if terminated or truncated:
                break
        
        returns.append(total)
    
    env.close()
    return float(np.mean(returns))


def generate_cancer_test_transitions(
    max_t: int,
    setting: int,
    policies: Dict[str, TorchPolicy],
    n_transitions_per_policy: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test transitions by rolling out policies (TorchPolicy interface)."""
    env = create_cancer_env(max_t=max_t, setting=setting)
    rng = np.random.default_rng(seed)
    all_states, all_actions, all_next_states = [], [], []
    
    for policy_name, policy in policies.items():
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        # Reset time for time-dependent policies
        if hasattr(policy, 'reset_time'):
            policy.reset_time()
        for _ in range(n_transitions_per_policy):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_tensor = policy.sample_torch_actions(obs_tensor, deterministic=True)
            action = action_tensor.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            all_states.append(obs.copy())
            all_actions.append(action.copy())
            all_next_states.append(next_obs.copy())
            obs = next_obs
            # Step time for time-dependent policies
            if hasattr(policy, 'step_time'):
                policy.step_time()
            if terminated or truncated:
                obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
                if hasattr(policy, 'reset_time'):
                    policy.reset_time()
    
    env.close()
    return (
        np.asarray(all_states, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        np.asarray(all_next_states, dtype=np.float32),
    )


def generate_expert_test_transitions(
    max_t: int,
    setting: int,
    experts: List[ExpertPolicy],
    n_transitions_per_policy: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test transitions by rolling out expert policies directly."""
    env = create_cancer_env(max_t=max_t, setting=setting)
    rng = np.random.default_rng(seed)
    all_states, all_actions, all_next_states = [], [], []
    
    for expert in experts:
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        time_step = 0
        for _ in range(n_transitions_per_policy):
            action = expert.get_action(obs, time_step)
            next_obs, reward, terminated, truncated, info = env.step(action)
            all_states.append(obs.copy())
            all_actions.append(action.copy())
            all_next_states.append(next_obs.copy())
            obs = next_obs
            time_step += 1
            if terminated or truncated:
                obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
                time_step = 0
    
    env.close()
    return (
        np.asarray(all_states, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        np.asarray(all_next_states, dtype=np.float32),
    )


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


# =============================================================================
# Main Pipeline
# =============================================================================

def run_cancer_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the complete cancer treatment offline RL pipeline with expert policies.
    
    This version uses predefined clinical expert policies instead of PPO-trained ones
    for more interpretable experiments.
    """
    if args.results_only and (
        args.force_dataset_collection
        or args.force_q_training
        or args.force_dynamics_training
    ):
        raise ValueError("--results-only cannot be combined with force-training flags.")

    if not DTRGYM_AVAILABLE:
        raise ImportError("DTRGym is required for this pipeline. Install with: pip install DTRGym")

    device = torch.device(args.device) if args.device != "auto" else default_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = initialize_wandb(args)

    # Get expert policies
    expert_policies = get_clinical_policies()
    expert_names = [e.name for e in expert_policies]

    # Log environment info
    wandb_log(wandb_run, {
        "env/name": "GhaffariCancerEnv",
        "env/obs_dim": OBS_DIM,
        "env/act_dim": ACT_DIM,
        "env/setting": args.setting,
        "env/max_t": args.max_t,
        "policies/type": "expert",
        "policies/names": expert_names,
    })

    print(f"=== Cancer Treatment Pipeline (Expert Policies) ===")
    print(f"Environment: GhaffariCancerEnv (Setting {args.setting})")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Expert Policies: {expert_names}")
    print()

    # =========================================================================
    # Step 1: Create expert policy adapters (no training needed!)
    # =========================================================================
    print("[Step 1] Creating expert policy adapters...")
    torch_policies = make_expert_policy_adapters(expert_policies)
    print(f"Created {len(torch_policies)} expert policy adapters.")
    
    wandb_log(wandb_run, {"policies/count": len(torch_policies)})

    # =========================================================================
    # Step 2: Collect or load offline dataset
    # =========================================================================
    dataset_path = args.output_dir / "offline_dataset.npz"
    dataset_loaded = False

    if dataset_path.exists() and not args.force_dataset_collection:
        print("[Step 2] Loading existing offline dataset from disk...")
        dataset = load_offline_dataset(dataset_path)
        dataset_loaded = True
    else:
        if args.results_only:
            raise FileNotFoundError(f"No saved offline dataset at {dataset_path}.")
        print("[Step 2] Collecting offline dataset from expert policies...")
        dataset = build_expert_offline_dataset(
            expert_policies, args.max_t, args.setting,
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

    wandb_log(
        wandb_run,
        {
            "dataset/transitions": int(len(dataset.states)),
            "dataset/initial_states": int(len(dataset.initial_states)),
            "dataset/loaded_from_disk": int(dataset_loaded),
        },
    )

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
            raise FileNotFoundError(f"No saved Q networks at {q_manifest}.")
        print("[Step 3] Training Q networks for each expert policy...")
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
            raise FileNotFoundError(f"No saved dynamics models at {dynamics_manifest}.")
        print("[Step 4] Training dynamics models...")
        start_time = time.perf_counter()
        sup_model, ranking_new_models = train_cancer_dynamics_models(
            dyn_dataset,
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

    # =========================================================================
    # Step 5-7: Evaluation
    # =========================================================================
    print("[Step 5-7] Evaluating expert policies across true env, Q, and dynamics...")
    results = []
    initial_states = dataset.initial_states

    # Generate test/train transitions for MSE evaluation
    test_states, test_actions, test_next_states = generate_expert_test_transitions(
        args.max_t, args.setting, expert_policies, n_transitions_per_policy=200, seed=args.seed + 999
    )
    train_states, train_actions, train_next_states = sample_training_transitions(
        dyn_dataset, n_samples=1000, seed=args.seed + 123
    )

    for expert in expert_policies:
        name = expert.name
        policy = torch_policies[name]
        q_net = q_models[name]

        q_est = evaluate_q_estimate(q_net, policy, initial_states, args.eval_rollouts)
        dyn_sup = evaluate_cancer_in_dynamics(
            sup_model, policy, initial_states, args.eval_horizon,
            args.gamma, device, args.eval_rollouts
        )

        # MSE evaluation
        dyn_sup_mse = calc_dynamics_mse(sup_model, test_states, test_actions, test_next_states, device)
        dyn_sup_train_mse = calc_dynamics_mse(sup_model, train_states, train_actions, train_next_states, device)

        dyn_rank_new: Dict[str, float] = {}
        dyn_rank_new_mse: Dict[str, float] = {}
        dyn_rank_new_train_mse: Dict[str, float] = {}
        for loss_name, dyn_model in ranking_new_models.items():
            dyn_rank_new[loss_name] = evaluate_cancer_in_dynamics(
                dyn_model, policy, initial_states, args.eval_horizon,
                args.gamma, device, args.eval_rollouts
            )
            dyn_rank_new_mse[loss_name] = calc_dynamics_mse(
                dyn_model, test_states, test_actions, test_next_states, device
            )
            dyn_rank_new_train_mse[loss_name] = calc_dynamics_mse(
                dyn_model, train_states, train_actions, train_next_states, device
            )

        # Evaluate in true environment
        env_mc = evaluate_expert_policy(
            expert, args.max_t, args.setting,
            args.eval_horizon, args.gamma, args.eval_rollouts, args.seed
        )

        # Log to W&B
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

        wandb_log(wandb_run, wandb_payload)

        results.append({
            "name": name,
            "policy_type": "expert",
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

        print(f"  {name}: env_mc={env_mc:.2f}, q_est={q_est:.2f}, dyn_sup={dyn_sup:.2f}, "
              f"dyn_sup_mse={dyn_sup_mse:.4f}")

    # =========================================================================
    # Save summary
    # =========================================================================
    config = args_to_config(args)
    summary = {
        "config": config,
        "num_transitions": int(len(dataset.states)),
        "expert_policies": expert_names,
        "q_model_paths": q_model_paths,
        "dynamics_paths": dynamics_paths,
        "results": results,
    }

    summary_path = args.output_dir / args.backbone / f"summary_{args.seed}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    if wandb_run is not None:
        wandb_run.summary.update({
            "num_transitions": int(len(dataset.states)),
            "expert_policies": expert_names,
            "q_model_paths": q_model_paths,
            "dynamics_paths": dynamics_paths,
            "results": results,
        })
        wandb_run.finish()

    print("\n=== Pipeline Complete ===")
    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cancer treatment pipeline with expert clinical policies for OPE evaluation"
    )
    
    # Environment-specific arguments
    parser.add_argument("--max-t", type=int, default=200, help="Max timesteps per episode (days)")
    parser.add_argument("--setting", type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Noise setting (1=none, 5=max noise+missing)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cancer_pipeline"),
    )
    
    # Add common arguments from base pipeline
    add_common_args(parser)
    
    # Override some defaults for cancer environment with expert policies
    parser.set_defaults(
        # No PPO training needed, but set reasonable rollout defaults
        total_steps=10_000,  # Not used for PPO, but kept for compatibility
        batch_size=512,
        gamma=0.99,
        q_epochs=500,
        q_batch=512,
        dyn_batch=512,
        dyn_early_stop_patience=20,
        dyn_min_epochs=0,
        eval_rollouts=200,
        eval_horizon=200,
        rollout_steps=1000,  # Steps per expert policy for dataset collection
        wandb_project="DT2-cancer-expert",
    )
    
    args = parser.parse_args()
    
    # Run the cancer-specific pipeline with expert policies
    run_cancer_pipeline(args)


if __name__ == "__main__":
    main()
