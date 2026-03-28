#!/usr/bin/env python3
"""Evaluate NSDT Ant simulators by policy-value alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eval_nsdt_alignment_common import AlignmentEnvSpec, create_arg_parser, run_alignment_eval


ENV_ID = "Ant-v5"
ANT_ACT_LOW = -1.0
ANT_ACT_HIGH = 1.0
HEALTHY_Z_MIN = 0.2
HEALTHY_Z_MAX = 1.0


def ant_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    forward_velocity = state[13]
    ctrl_cost = 0.5 * np.sum(np.square(action))
    healthy_reward = 1.0
    return float(forward_velocity + healthy_reward - ctrl_cost)


def ant_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    forward_velocity = states[..., 13]
    ctrl_cost = 0.5 * torch.sum(actions * actions, dim=-1)
    healthy_reward = 1.0
    return forward_velocity + healthy_reward - ctrl_cost


def ant_termination_torch(states: torch.Tensor) -> torch.Tensor:
    z_pos = states[..., 0]
    height_ok = (z_pos >= HEALTHY_Z_MIN) & (z_pos <= HEALTHY_Z_MAX)
    finite_ok = torch.all(torch.isfinite(states), dim=-1)
    return ~(height_ok & finite_ok)


SPEC = AlignmentEnvSpec(
    display_name="Ant",
    env_id=ENV_ID,
    dataset_env_name="Dataset-Ant",
    policy_dir=Path("pend-eval/ant-policies"),
    output_json=Path("pend-eval/results/nsdt_ant_alignment_summary.json"),
    act_low=ANT_ACT_LOW,
    act_high=ANT_ACT_HIGH,
    reward_fn_np=ant_reward_fn,
    reward_fn_torch=ant_reward_torch,
    termination_fn_torch=ant_termination_torch,
    default_env_kwargs={"include_cfrc_ext_in_observation": False},
)


def main() -> None:
    parser = create_arg_parser(SPEC)
    args = parser.parse_args()
    run_alignment_eval(args, SPEC)


if __name__ == "__main__":
    main()
