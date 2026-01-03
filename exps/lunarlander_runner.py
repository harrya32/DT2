"""
LunarLander-specific pipeline using the base pipeline infrastructure.

This is a lightweight script that defines environment-specific components
and delegates to the shared base pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from base_pipeline import add_common_args, run_pipeline


# =============================================================================
# Environment Constants
# =============================================================================

ENV_ID = "LunarLanderContinuous-v3"
LUNARLANDER_ACT_LOW = -1.0
LUNARLANDER_ACT_HIGH = 1.0


# =============================================================================
# Reward Functions
# =============================================================================

def lunarlander_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
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


def lunarlander_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch version of the LunarLander reward function."""
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = states.unbind(-1)
    shaping = (
        -100.0 * torch.sqrt(x * x + y * y)
        - 100.0 * torch.sqrt(xdot * xdot + ydot * ydot)
        - 100.0 * torch.abs(theta)
        - 10.0 * torch.abs(thetadot)
        + (leg1 + leg2) * 10.0
    )
    return shaping - 0.3 * torch.sum(actions * actions, dim=-1)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="LunarLander offline RL pipeline")
    
    # Environment-specific arguments
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nll/lunarlander_pipeline"),
    )
    parser.add_argument("--wandb-project", type=str, default="DT2-lunarlander")
    
    # Add common arguments from base pipeline
    add_common_args(parser)
    
    args = parser.parse_args()
    
    # Run the pipeline with environment-specific reward functions
    run_pipeline(
        args=args,
        env_id=args.env_id,
        reward_fn=lunarlander_reward_fn,
        reward_fn_torch=lunarlander_reward_torch,
        act_low=LUNARLANDER_ACT_LOW,
        act_high=LUNARLANDER_ACT_HIGH,
    )


if __name__ == "__main__":
    main()
