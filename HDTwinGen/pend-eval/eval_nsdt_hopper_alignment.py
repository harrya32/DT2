#!/usr/bin/env python3
"""Evaluate NSDT Hopper simulators by policy-value alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eval_nsdt_alignment_common import AlignmentEnvSpec, create_arg_parser, run_alignment_eval


ENV_ID = "Hopper-v4"
HOPPER_ACT_LOW = -1.0
HOPPER_ACT_HIGH = 1.0


def hopper_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    forward_velocity = state[5]
    ctrl_cost = 0.001 * np.sum(np.square(action))
    healthy_reward = 1.0
    return float(forward_velocity - ctrl_cost + healthy_reward)


def hopper_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    forward_velocity = states[..., 5]
    ctrl_cost = 0.001 * torch.sum(actions * actions, dim=-1)
    healthy_reward = 1.0
    return forward_velocity - ctrl_cost + healthy_reward


def hopper_termination_torch(states: torch.Tensor) -> torch.Tensor:
    z_pos = states[..., 0]
    y_angle = states[..., 1]
    height_ok = z_pos >= 0.7
    angle_ok = torch.abs(y_angle) <= 0.2
    state_ok = torch.all(torch.abs(states) <= 100.0, dim=-1)
    return ~(height_ok & angle_ok & state_ok)


SPEC = AlignmentEnvSpec(
    display_name="Hopper",
    env_id=ENV_ID,
    dataset_env_name="Dataset-Hopper",
    policy_dir=Path("pend-eval/hopper-policies"),
    output_json=Path("pend-eval/results/nsdt_hopper_alignment_summary.json"),
    act_low=HOPPER_ACT_LOW,
    act_high=HOPPER_ACT_HIGH,
    reward_fn_np=hopper_reward_fn,
    reward_fn_torch=hopper_reward_torch,
    termination_fn_torch=hopper_termination_torch,
)


def main() -> None:
    parser = create_arg_parser(SPEC)
    args = parser.parse_args()
    run_alignment_eval(args, SPEC)


if __name__ == "__main__":
    main()
