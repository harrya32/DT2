#!/usr/bin/env python3
"""Evaluate NSDT LunarLander simulators by policy-value alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eval_nsdt_alignment_common import AlignmentEnvSpec, create_arg_parser, run_alignment_eval


ENV_ID = "LunarLanderContinuous-v3"
LUNARLANDER_ACT_LOW = -1.0
LUNARLANDER_ACT_HIGH = 1.0


def lunarlander_reward_fn(state: np.ndarray, action: np.ndarray) -> float:
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = state
    shaping = (
        -100.0 * np.sqrt(x * x + y * y)
        - 100.0 * np.sqrt(xdot * xdot + ydot * ydot)
        - 100.0 * abs(theta)
        - 10.0 * abs(thetadot)
        + (leg1 + leg2) * 10.0
    )
    return float(shaping - 0.3 * np.square(action).sum())


def lunarlander_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = states.unbind(-1)
    shaping = (
        -100.0 * torch.sqrt(x * x + y * y)
        - 100.0 * torch.sqrt(xdot * xdot + ydot * ydot)
        - 100.0 * torch.abs(theta)
        - 10.0 * torch.abs(thetadot)
        + (leg1 + leg2) * 10.0
    )
    return shaping - 0.3 * torch.sum(actions * actions, dim=-1)


def lunarlander_termination_torch(states: torch.Tensor) -> torch.Tensor:
    x, y, xdot, ydot, theta, thetadot, leg1, leg2 = states.unbind(-1)
    out_of_bounds = torch.abs(x) >= 1.0
    crashed = y < 0.0
    legs_contact = (leg1 > 0.5) & (leg2 > 0.5)
    low_velocity = (torch.abs(xdot) < 0.1) & (torch.abs(ydot) < 0.1)
    landed = legs_contact & low_velocity & (y < 0.5)
    return out_of_bounds | crashed | landed


SPEC = AlignmentEnvSpec(
    display_name="LunarLander",
    env_id=ENV_ID,
    dataset_env_name="Dataset-LunarLander",
    policy_dir=Path("pend-eval/lunarlander-policies"),
    output_json=Path("pend-eval/results/nsdt_lunarlander_alignment_summary.json"),
    act_low=LUNARLANDER_ACT_LOW,
    act_high=LUNARLANDER_ACT_HIGH,
    reward_fn_np=lunarlander_reward_fn,
    reward_fn_torch=lunarlander_reward_torch,
    termination_fn_torch=lunarlander_termination_torch,
)


def main() -> None:
    parser = create_arg_parser(SPEC)
    args = parser.parse_args()
    run_alignment_eval(args, SPEC)


if __name__ == "__main__":
    main()
