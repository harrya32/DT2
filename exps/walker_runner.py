"""
Walker2d-specific pipeline using the base pipeline infrastructure.

This is a lightweight script that defines environment-specific components
and delegates to the shared base pipeline.

Walker2d-v4 details:
- State dimension: 17 (positions and velocities of joints)
- Action dimension: 6 (torques applied to joints)
- Reward: forward_velocity - 0.001 * control_cost + healthy_reward (1.0)
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

ENV_ID = "Walker2d-v4"
WALKER_ACT_LOW = -1.0
WALKER_ACT_HIGH = 1.0

# State bounds for Walker2d (approximate, based on typical ranges)
# State: [z_pos, y_angle, thigh_angle, leg_angle, foot_angle (x2 for each leg),
#         x_vel, z_vel, y_angular_vel, joint_angular_velocities (6)]
WALKER_STATE_LOW = torch.tensor(
    [-np.inf] * 17, dtype=torch.float32
)
WALKER_STATE_HIGH = torch.tensor(
    [np.inf] * 17, dtype=torch.float32
)

# Healthy state bounds for Walker2d (used in termination)
HEALTHY_Z_MIN = 0.8  # Minimum height to be considered healthy
HEALTHY_Z_MAX = 2.0  # Maximum height
HEALTHY_ANGLE_MAX = 1.0  # Maximum angle deviation


# =============================================================================
# Reward Functions
# =============================================================================

def walker_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute the Walker2d reward from state and action.
    
    The reward in Walker2d is:
        reward = forward_velocity - ctrl_cost_weight * sum(action^2) + healthy_reward
    
    Note: The forward velocity is encoded in the observation at index 8 (x_velocity).
    ctrl_cost_weight = 0.001 by default in Walker2d-v4
    healthy_reward = 1.0 by default (given when walker is healthy)
    
    For simplicity, we assume the walker is healthy in the reward function
    since termination would be handled separately by the environment.
    """
    # x_velocity is at index 8 in the default observation
    forward_velocity = state[8]
    ctrl_cost = 0.001 * np.sum(np.square(action))
    healthy_reward = 1.0  # Assuming healthy state
    return float(forward_velocity - ctrl_cost + healthy_reward)


def walker_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch version of the Walker2d reward function."""
    # x_velocity is at index 8
    forward_velocity = states[..., 8]
    ctrl_cost = 0.001 * torch.sum(actions * actions, dim=-1)
    healthy_reward = 1.0  # Assuming healthy state
    return forward_velocity - ctrl_cost + healthy_reward


def walker_termination_torch(states: torch.Tensor) -> torch.Tensor:
    """
    Check termination conditions for Walker2d.
    
    Walker2d terminates when:
    - z_position (state[0]) is not in [0.8, 2.0] (too low or too high)
    - y_angle (state[1]) is not in [-1.0, 1.0] (too tilted)
    
    Returns:
        Boolean tensor indicating which states are terminated.
    """
    z_pos = states[..., 0]
    y_angle = states[..., 1]
    
    # Height check: terminate if z < 0.8 or z > 2.0
    height_ok = (z_pos >= 0.8) & (z_pos <= 2.0)
    
    # Angle check: terminate if |y_angle| > 1.0
    angle_ok = torch.abs(y_angle) <= 1.0
    
    # Terminated if any condition is violated
    terminated = ~(height_ok & angle_ok)
    return terminated


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Walker2d offline RL pipeline")
    
    # Environment-specific arguments
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/test/walker_pipeline"),
    )
    parser.add_argument("--wandb-project", type=str, default="DT2-walker")
    
    # Add common arguments from base pipeline
    add_common_args(parser)
    
    args = parser.parse_args()
    
    # Run the pipeline with environment-specific reward functions
    run_pipeline(
        args=args,
        env_id=args.env_id,
        reward_fn=walker_reward_fn,
        reward_fn_torch=walker_reward_torch,
        act_low=WALKER_ACT_LOW,
        act_high=WALKER_ACT_HIGH,
        state_low=WALKER_STATE_LOW,
        state_upper=WALKER_STATE_HIGH,
        wrapped_dims=[],  # No wrapped dimensions in Walker2d
        termination_fn_torch=walker_termination_torch,
    )


if __name__ == "__main__":
    main()
