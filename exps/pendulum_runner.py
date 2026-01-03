"""
Pendulum-specific pipeline using the base pipeline infrastructure.

This is a lightweight script that defines environment-specific components
and delegates to the shared base pipeline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from base_pipeline import add_common_args, run_pipeline


# =============================================================================
# Environment Constants
# =============================================================================

ENV_ID = "Pendulum-v1"
PENDULUM_ACT_LOW = -2.0
PENDULUM_ACT_HIGH = 2.0


# =============================================================================
# Reward Functions
# =============================================================================

def pendulum_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    """Compute the Pendulum reward from state and action."""
    cos_theta, sin_theta, theta_dot = state
    theta = math.atan2(sin_theta, cos_theta)
    torque = float(np.clip(action, PENDULUM_ACT_LOW, PENDULUM_ACT_HIGH).item())
    return -(
        theta * theta
        + 0.1 * theta_dot * theta_dot
        + 0.001 * torque * torque
    )


def pendulum_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch version of the Pendulum reward function."""
    cos_theta, sin_theta, theta_dot = states.unbind(dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    torque_penalty = torch.sum(actions * actions, dim=-1)
    return -(theta.pow(2) + 0.1 * theta_dot.pow(2) + 0.001 * torque_penalty)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Pendulum offline RL pipeline")
    
    # Environment-specific arguments
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nll/pendulum_pipeline"),
    )
    parser.add_argument("--wandb-project", type=str, default="DT2-pendulum")
    
    # Add common arguments from base pipeline
    add_common_args(parser)
    
    args = parser.parse_args()
    
    # Run the pipeline with environment-specific reward functions
    run_pipeline(
        args=args,
        env_id=args.env_id,
        reward_fn=pendulum_reward_fn,
        reward_fn_torch=pendulum_reward_torch,
        act_low=PENDULUM_ACT_LOW,
        act_high=PENDULUM_ACT_HIGH,
    )


if __name__ == "__main__":
    main()
