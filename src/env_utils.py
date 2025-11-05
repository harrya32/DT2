from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import math

State = np.ndarray
Action = np.ndarray
TorchState = torch.Tensor
TorchAction = torch.Tensor


def make_lunarlander_env(**kwargs: Dict[str, Any]) -> gym.Env:
    """Create a LunarLanderContinuous-v3 environment."""
    return gym.make("LunarLanderContinuous-v3", **kwargs)


def lunarlander_reward_fn(state: State, action: Action) -> float:
    """Analytic approximation of the LunarLander reward used by Gym."""
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = state
    shaping = (
        -100.0 * np.sqrt(x * x + y * y)
        - 100.0 * np.sqrt(xdot * xdot + ydot * ydot)
        - 100.0 * abs(theta)
        - 10.0 * abs(thetadot)
        + (leg1 + leg2) * 10.0
    )
    return float(shaping - 0.3 * np.square(action).sum())


def lunarlander_reward_torch(state: TorchState, action: TorchAction) -> TorchAction:
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = state.unbind(-1)
    shaping = (
        -100.0 * torch.sqrt(x * x + y * y)
        - 100.0 * torch.sqrt(xdot * xdot + ydot * ydot)
        - 100.0 * torch.abs(theta)
        - 10.0 * torch.abs(thetadot)
        + (leg1 + leg2) * 10.0
    )
    return shaping - 0.3 * torch.sum(action * action, dim=-1)


def evaluate_policy_in_env(
    env,
    policy,
    episodes: int = 50,
    horizon: int = 500,
    gamma: float = 0.97,
    seed: int = 0,
    reward_fn: Callable[[State, Action], float] = lunarlander_reward_fn,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        discount = 1.0
        for _ in range(horizon):
            action = policy.sample(state, rng)
            next_state, _reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
            reward = reward_fn(state, action)
            total += discount * reward
            discount *= gamma
            state = next_state
            if terminated or truncated:
                break
        returns.append(total)

    mean = float(np.mean(returns))
    stderr = float(np.std(returns) / math.sqrt(len(returns))) if returns else 0.0
    return mean, stderr
