"""
HalfCheetah-specific pipeline using the base pipeline infrastructure.

This is a lightweight script that defines environment-specific components
and delegates to the shared base pipeline.

HalfCheetah-v5 details:
- State dimension: 17 (positions and velocities of joints)
- Action dimension: 6 (torques applied to joints)
- Reward: forward_velocity - 0.1 * control_cost
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

ENV_ID = "HalfCheetah-v5"
CHEETAH_ACT_LOW = -1.0
CHEETAH_ACT_HIGH = 1.0

# State bounds for HalfCheetah (approximate, based on typical ranges)
# State: [x_pos (excluded), z_pos, y_angle, joint_angles (6), x_vel, joint_velocities (8)]
# The observation excludes x_position by default in v4
CHEETAH_STATE_LOW = torch.tensor(
    [-np.inf] * 17, dtype=torch.float32
)
CHEETAH_STATE_HIGH = torch.tensor(
    [np.inf] * 17, dtype=torch.float32
)


# =============================================================================
# Reward Functions
# =============================================================================

def cheetah_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute the HalfCheetah reward from state and action.
    
    The reward in HalfCheetah is:
        reward = forward_velocity - ctrl_cost_weight * sum(action^2)
    
    Note: The forward velocity is encoded in the observation at index 8 (x_velocity).
    ctrl_cost_weight = 0.1 by default in HalfCheetah-v4
    """
    # x_velocity is at index 8 in the default observation
    forward_velocity = state[8]
    ctrl_cost = 0.1 * np.sum(np.square(action))
    return float(forward_velocity - ctrl_cost)


def cheetah_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch version of the HalfCheetah reward function."""
    # x_velocity is at index 8
    forward_velocity = states[..., 8]
    ctrl_cost = 0.1 * torch.sum(actions * actions, dim=-1)
    return forward_velocity - ctrl_cost


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="HalfCheetah offline RL pipeline")
    
    # Environment-specific arguments
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/test/cheetah_pipeline"),
    )
    parser.add_argument("--wandb-project", type=str, default="DT2-cheetah")
    
    # Add common arguments from base pipeline
    add_common_args(parser)
    
    args = parser.parse_args()
    
    # Run the pipeline with environment-specific reward functions
    run_pipeline(
        args=args,
        env_id=args.env_id,
        reward_fn=cheetah_reward_fn,
        reward_fn_torch=cheetah_reward_torch,
        act_low=CHEETAH_ACT_LOW,
        act_high=CHEETAH_ACT_HIGH,
        state_low=CHEETAH_STATE_LOW,
        state_upper=CHEETAH_STATE_HIGH,
        wrapped_dims=[],  # No wrapped dimensions in HalfCheetah
        termination_fn_torch=None,  # HalfCheetah has no termination, only truncation
    )


if __name__ == "__main__":
    main()
