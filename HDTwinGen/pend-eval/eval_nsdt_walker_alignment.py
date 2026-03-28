#!/usr/bin/env python3
"""Evaluate NSDT Walker2d simulators by policy-value alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eval_nsdt_alignment_common import AlignmentEnvSpec, create_arg_parser, run_alignment_eval


ENV_ID = "Walker2d-v5"
WALKER_ACT_LOW = -1.0
WALKER_ACT_HIGH = 1.0


def walker_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    forward_velocity = state[8]
    ctrl_cost = 0.001 * np.sum(np.square(action))
    healthy_reward = 1.0
    return float(forward_velocity - ctrl_cost + healthy_reward)


def walker_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    forward_velocity = states[..., 8]
    ctrl_cost = 0.001 * torch.sum(actions * actions, dim=-1)
    healthy_reward = 1.0
    return forward_velocity - ctrl_cost + healthy_reward


def walker_termination_torch(states: torch.Tensor) -> torch.Tensor:
    z_pos = states[..., 0]
    y_angle = states[..., 1]
    height_ok = (z_pos >= 0.8) & (z_pos <= 2.0)
    angle_ok = torch.abs(y_angle) <= 1.0
    return ~(height_ok & angle_ok)


SPEC = AlignmentEnvSpec(
    display_name="Walker2d",
    env_id=ENV_ID,
    dataset_env_name="Dataset-Walker",
    policy_dir=Path("pend-eval/walker-policies"),
    output_json=Path("pend-eval/results/nsdt_walker_alignment_summary.json"),
    act_low=WALKER_ACT_LOW,
    act_high=WALKER_ACT_HIGH,
    reward_fn_np=walker_reward_fn,
    reward_fn_torch=walker_reward_torch,
    termination_fn_torch=walker_termination_torch,
)


def main() -> None:
    parser = create_arg_parser(SPEC)
    args = parser.parse_args()
    run_alignment_eval(args, SPEC)


if __name__ == "__main__":
    main()
