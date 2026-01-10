"""
Ant-specific pipeline using the base pipeline infrastructure.

This is a lightweight script that defines environment-specific components
and delegates to the shared base pipeline.

Ant-v5 details:
- State dimension: 105 (by default, excludes x,y position of torso)
  Observation space (105 dims):
    - qpos (13 elements): z position, orientation quaternion (4), joint angles (8)
    - qvel (14 elements): linear velocity (3), angular velocity (3), joint velocities (8)
    - cfrc_ext (78 elements): contact forces on body parts (13 bodies * 6)
  
  Detailed indices:
    - Index 0: z position of torso
    - Index 1-4: orientation quaternion (4)
    - Index 5-12: joint angles (8)
    - Index 13: x_velocity
    - Index 14: y_velocity
    - Index 15: z_velocity
    - Index 16-18: angular velocities (3)
    - Index 19-26: joint angular velocities (8)
    - Index 27-104: contact forces (78)
    
- Action dimension: 8 (torques applied to hip and ankle joints)
- Reward: forward_velocity + healthy_reward - ctrl_cost - contact_cost
  where:
    - forward_velocity = x_velocity (index 13)
    - healthy_reward = 1.0 (default, when ant is healthy)
    - ctrl_cost = 0.5 * sum(action^2)
    - contact_cost = 5e-4 * sum(clip(contact_forces, -1, 1)^2)
- Termination: z_position not in [0.2, 1.0] or state has NaN/Inf
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

ENV_ID = "Ant-v5"
ANT_ACT_LOW = -1.0
ANT_ACT_HIGH = 1.0

# State bounds for Ant (approximate, based on typical ranges)
# The observation space is technically unbounded, but we use inf
ANT_STATE_LOW = torch.tensor(
    [-np.inf] * 105, dtype=torch.float32
)
ANT_STATE_HIGH = torch.tensor(
    [np.inf] * 105, dtype=torch.float32
)

# Healthy state bounds for Ant (used in termination)
HEALTHY_Z_MIN = 0.2  # Minimum height to be considered healthy
HEALTHY_Z_MAX = 1.0  # Maximum height to be considered healthy


# =============================================================================
# Reward Functions
# =============================================================================

def ant_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute the Ant reward from state and action.
    
    The reward in Ant-v5 is:
        reward = forward_velocity + healthy_reward - ctrl_cost - contact_cost
    
    Observation indices (105 dims):
        0: z_position (height of torso)
        1-4: orientation (quaternion)
        5-12: joint angles (8 joints)
        13: x_velocity
        14: y_velocity
        15: z_velocity
        16-18: angular velocities of torso
        19-26: joint angular velocities
        27-104: contact forces (13 bodies * 6 force/torque components)
    
    Default parameters:
        - ctrl_cost_weight = 0.5
        - healthy_reward = 1.0
        - contact_cost_weight = 5e-4
    
    For simplicity, we assume the ant is healthy in the reward function
    since termination would be handled separately.
    """
    # x_velocity is at index 13 in the default observation
    forward_velocity = state[13]
    ctrl_cost = 0.5 * np.sum(np.square(action))
    healthy_reward = 1.0  # Assuming healthy state
    
    # Contact cost: uses contact forces from indices 27-104
    contact_forces = state[27:105]
    contact_cost = 5e-4 * np.sum(np.square(np.clip(contact_forces, -1.0, 1.0)))
    
    return float(forward_velocity + healthy_reward - ctrl_cost - contact_cost)


def ant_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Vectorized PyTorch version of the Ant reward function."""
    # x_velocity is at index 13
    forward_velocity = states[..., 13]
    ctrl_cost = 0.5 * torch.sum(actions * actions, dim=-1)
    healthy_reward = 1.0  # Assuming healthy state
    
    # Contact cost: uses contact forces from indices 27-104
    contact_forces = states[..., 27:105]
    contact_cost = 5e-4 * torch.sum(torch.clamp(contact_forces, -1.0, 1.0) ** 2, dim=-1)
    
    return forward_velocity + healthy_reward - ctrl_cost - contact_cost


def ant_termination_torch(states: torch.Tensor) -> torch.Tensor:
    """
    Check termination conditions for Ant.
    
    Ant-v5 terminates when:
    - z_position (state[0]) is not in [0.2, 1.0] (torso too low = fallen, too high = flipped)
    - Any state element is NaN or Inf (numerical instability)
    
    Note: By default in Ant-v5, terminate_when_unhealthy=True
    
    Returns:
        Boolean tensor indicating which states are terminated.
    """
    z_pos = states[..., 0]
    
    # Height check: terminate if z not in [0.2, 1.0]
    height_ok = (z_pos >= HEALTHY_Z_MIN) & (z_pos <= HEALTHY_Z_MAX)
    
    # Check for NaN or Inf (numerical stability)
    finite_ok = torch.all(torch.isfinite(states), dim=-1)
    
    # Terminated if any condition is violated
    terminated = ~(height_ok & finite_ok)
    return terminated


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Ant offline RL pipeline")
    
    # Environment-specific arguments
    parser.add_argument("--env-id", default=ENV_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/test/ant_pipeline"),
    )
    parser.add_argument("--wandb-project", type=str, default="DT2-ant")
    
    # Add common arguments from base pipeline
    add_common_args(parser)
    
    args = parser.parse_args()
    
    # Run the pipeline with environment-specific reward functions
    run_pipeline(
        args=args,
        env_id=args.env_id,
        reward_fn=ant_reward_fn,
        reward_fn_torch=ant_reward_torch,
        act_low=ANT_ACT_LOW,
        act_high=ANT_ACT_HIGH,
        state_low=ANT_STATE_LOW,
        state_upper=ANT_STATE_HIGH,
        wrapped_dims=[],  # No wrapped dimensions in Ant
        termination_fn_torch=ant_termination_torch,
    )


if __name__ == "__main__":
    main()
