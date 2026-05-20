"""
Hopper-specific pipeline using the base pipeline infrastructure.

This is a lightweight script that defines environment-specific components
and delegates to the shared base pipeline.

Hopper-v4 details:
- State dimension: 11 (positions and velocities of joints)
- Action dimension: 3 (torques applied to joints)
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

ENV_ID = "Hopper-v4"
HOPPER_ACT_LOW = -1.0
HOPPER_ACT_HIGH = 1.0

# State bounds for Hopper (approximate, based on typical ranges)
# State: [z_pos, y_angle, thigh_angle, leg_angle, foot_angle, 
#         x_vel, z_vel, y_angular_vel, thigh_angular_vel, leg_angular_vel, foot_angular_vel]
HOPPER_STATE_LOW = torch.tensor(
    [-np.inf] * 11, dtype=torch.float32
)
HOPPER_STATE_HIGH = torch.tensor(
    [np.inf] * 11, dtype=torch.float32
)

# Healthy state bounds for Hopper (used in termination)
HEALTHY_STATE_MIN = -100.0
HEALTHY_STATE_MAX = 100.0
HEALTHY_Z_MIN = 0.7  # Minimum height to be considered healthy
HEALTHY_ANGLE_MAX = 0.2  # Maximum angle deviation


# =============================================================================
# Reward Functions
# =============================================================================

def hopper_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute the Hopper reward from state and action.
    
    The reward in Hopper is:
        reward = forward_velocity - ctrl_cost_weight * sum(action^2) + healthy_reward
    
    Note: The forward velocity is encoded in the observation at index 5 (x_velocity).
    ctrl_cost_weight = 0.001 by default in Hopper-v4
    healthy_reward = 1.0 by default (given when hopper is healthy)
    
    For simplicity, we assume the hopper is healthy in the reward function
    since termination would be handled separately by the environment.
    """
    # x_velocity is at index 5 in the default observation
    forward_velocity = state[5]
    ctrl_cost = 0.001 * np.sum(np.square(action))
    healthy_reward = 1.0  # Assuming healthy state
    return float(forward_velocity - ctrl_cost + healthy_reward)


def hopper_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch version of the Hopper reward function."""
    # x_velocity is at index 5
    forward_velocity = states[..., 5]
    ctrl_cost = 0.001 * torch.sum(actions * actions, dim=-1)
    healthy_reward = 1.0  # Assuming healthy state
    return forward_velocity - ctrl_cost + healthy_reward


def hopper_termination_torch(states: torch.Tensor) -> torch.Tensor:
    """
    Check termination conditions for Hopper.
    
    Hopper terminates when:
    - z_position (state[0]) is not in [0.7, inf) (too low = fallen)
    - Any state element has absolute value > 100 (numerical instability)
    - y_angle (state[1]) is not in [-0.2, 0.2] (too tilted)
    
    Returns:
        Boolean tensor indicating which states are terminated.
    """
    z_pos = states[..., 0]
    y_angle = states[..., 1]
    
    # Height check: terminate if z < 0.7
    height_ok = z_pos >= 0.7
    
    # Angle check: terminate if |y_angle| > 0.2
    angle_ok = torch.abs(y_angle) <= 0.2
    
    # State bounds check: terminate if any |state| > 100
    state_ok = torch.all(torch.abs(states) <= 100.0, dim=-1)
    
    # Terminated if any condition is violated
    terminated = ~(height_ok & angle_ok & state_ok)
    return terminated


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Hopper offline RL pipeline")
    
    # Environment-specific arguments
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/camera_ready/hopper_pipeline"),
    )
    parser.add_argument("--wandb-project", type=str, default="DT2-hopper")
    
    # Add common arguments from base pipeline
    add_common_args(parser)
    
    args = parser.parse_args()
    
    # Run the pipeline with environment-specific reward functions
    run_pipeline(
        args=args,
        env_id=args.env_id,
        reward_fn=hopper_reward_fn,
        reward_fn_torch=hopper_reward_torch,
        act_low=HOPPER_ACT_LOW,
        act_high=HOPPER_ACT_HIGH,
        state_low=HOPPER_STATE_LOW,
        state_upper=HOPPER_STATE_HIGH,
        wrapped_dims=[],  # No wrapped dimensions in Hopper
        termination_fn_torch=hopper_termination_torch,
    )


if __name__ == "__main__":
    main()
