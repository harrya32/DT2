#!/usr/bin/env python3
"""Evaluate NSDT Pendulum simulators by policy-value alignment."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from eval_nsdt_alignment_common import AlignmentEnvSpec, create_arg_parser, run_alignment_eval


ENV_ID = "Pendulum-v1"
PENDULUM_ACT_LOW = -2.0
PENDULUM_ACT_HIGH = 2.0


def pendulum_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    cos_theta, sin_theta, theta_dot = state
    theta = math.atan2(float(sin_theta), float(cos_theta))
    torque = float(np.clip(np.asarray(action).reshape(-1)[0], PENDULUM_ACT_LOW, PENDULUM_ACT_HIGH))
    return -(theta * theta + 0.1 * float(theta_dot) * float(theta_dot) + 0.001 * torque * torque)


def pendulum_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    cos_theta, sin_theta, theta_dot = states.unbind(dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    torque = torch.clamp(actions[..., 0], min=PENDULUM_ACT_LOW, max=PENDULUM_ACT_HIGH)
    return -(theta.pow(2) + 0.1 * theta_dot.pow(2) + 0.001 * torque.pow(2))


SPEC = AlignmentEnvSpec(
    display_name="Pendulum",
    env_id=ENV_ID,
    dataset_env_name="Dataset-Pendulum",
    policy_dir=Path("pend-eval/policies"),
    output_json=Path("pend-eval/results/nsdt_alignment_summary.json"),
    act_low=PENDULUM_ACT_LOW,
    act_high=PENDULUM_ACT_HIGH,
    reward_fn_np=pendulum_reward_fn,
    reward_fn_torch=pendulum_reward_torch,
    termination_fn_torch=None,
)


def main() -> None:
    parser = create_arg_parser(SPEC)
    args = parser.parse_args()
    run_alignment_eval(args, SPEC)


if __name__ == "__main__":
    main()
