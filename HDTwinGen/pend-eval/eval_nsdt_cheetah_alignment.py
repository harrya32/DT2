#!/usr/bin/env python3
"""Evaluate NSDT HalfCheetah simulators by policy-value alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eval_nsdt_alignment_common import AlignmentEnvSpec, create_arg_parser, run_alignment_eval


ENV_ID = "HalfCheetah-v5"
CHEETAH_ACT_LOW = -1.0
CHEETAH_ACT_HIGH = 1.0


def cheetah_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    forward_velocity = state[8]
    ctrl_cost = 0.1 * np.sum(np.square(action))
    return float(forward_velocity - ctrl_cost)


def cheetah_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    forward_velocity = states[..., 8]
    ctrl_cost = 0.1 * torch.sum(actions * actions, dim=-1)
    return forward_velocity - ctrl_cost


SPEC = AlignmentEnvSpec(
    display_name="HalfCheetah",
    env_id=ENV_ID,
    dataset_env_name="Dataset-Cheetah",
    policy_dir=Path("pend-eval/cheetah-policies"),
    output_json=Path("pend-eval/results/nsdt_cheetah_alignment_summary.json"),
    act_low=CHEETAH_ACT_LOW,
    act_high=CHEETAH_ACT_HIGH,
    reward_fn_np=cheetah_reward_fn,
    reward_fn_torch=cheetah_reward_torch,
    termination_fn_torch=None,
)


def main() -> None:
    parser = create_arg_parser(SPEC)
    args = parser.parse_args()
    run_alignment_eval(args, SPEC)


if __name__ == "__main__":
    main()
