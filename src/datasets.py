from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generator, Optional

import numpy as np
import torch

from .env_utils import lunarlander_reward_fn


@dataclass
class OfflineDataset:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    initial_states: np.ndarray
    mask: Optional[np.ndarray] = None  # Optional padding mask for sequence inputs [B, T]

    def as_dict(self) -> Dict[str, np.ndarray]:
        data = {
            "s": self.states,
            "a": self.actions,
            "r": self.rewards,
            "s_next": self.next_states,
            "done": self.dones,
            "s0": self.initial_states,
        }
        if self.mask is not None:
            data["mask"] = self.mask
        return data

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        data = self.as_dict()
        return {k: torch.tensor(v, dtype=_infer_dtype(v), device=device) for k, v in data.items()}

    def __len__(self) -> int:
        return self.states.shape[0]


def _infer_dtype(array: np.ndarray) -> torch.dtype:
    if array.dtype == np.bool_:
        return torch.float32
    if np.issubdtype(array.dtype, np.integer):
        return torch.float32
    return torch.float32


def collect_dataset(
    env,
    behavior_policy,
    n_episodes: int = 100,
    max_steps: int = 500,
    seed: int = 0,
    reward_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = lunarlander_reward_fn,
) -> OfflineDataset:
    """Gather transitions from a behavior policy into an OfflineDataset."""

    rng = np.random.default_rng(seed)
    transitions = []
    initial_states = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        initial_states.append(state.copy())
        for _ in range(max_steps):
            action = behavior_policy.sample(state, rng)
            next_state, reward_env, terminated, truncated, _ = env.step(action.astype(np.float32))
            reward = reward_fn(state, action) if reward_fn is not None else reward_env
            done = bool(terminated or truncated)
            transitions.append((state.copy(), action.copy(), reward, next_state.copy(), done))
            state = next_state
            if done:
                break

    states, actions, rewards, next_states, dones = map(np.array, zip(*transitions))
    return OfflineDataset(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        initial_states=np.asarray(initial_states, dtype=np.float32),
    )
