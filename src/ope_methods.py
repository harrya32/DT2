from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .datasets import OfflineDataset
from .dynamics import evaluate_with_model_rollouts, train_dynamics_supervised
from .env_utils import evaluate_policy_in_env
from .fqe import estimate_V_from_Q_on_s0, train_q_fqe, train_value_fqe_state
from .policies import GaussianLinearPolicy
from .utils import DEVICE
from .value_aware import train_q_aware_model, train_ranking_aware_model, train_value_aware_model


def run_model_based_mc(
    env,
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policies: Sequence[GaussianLinearPolicy],
    gamma: float,
    horizon: int,
    episodes: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    hidden: int = 128,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    print("\n=== Naive model-based estimate ===")
    dynamics = train_dynamics_supervised(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        hidden=hidden,
    )
    estimates: Dict[str, float] = {}
    for policy in target_policies:
        mean, stderr = evaluate_with_model_rollouts(
            env,
            dynamics,
            policy,
            episodes=episodes,
            horizon=horizon,
            gamma=gamma,
            seed=seed,
        )
        print(f"[{policy.name}] J^pi (model rollouts): {mean:.3f} ± {1.96 * stderr:.3f}")
        estimates[policy.name] = mean
    return dynamics, estimates


def run_value_fqe_block(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policies: Sequence[GaussianLinearPolicy],
    behavior_policy: GaussianLinearPolicy,
    gamma: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    hidden: int = 128,
    report: bool = True,
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, float]]:
    print("\n=== FVE estimate ===")

    value_networks: Dict[str, torch.nn.Module] = {}
    estimates: Dict[str, float] = {}
    tensors = dataset.as_dict() if isinstance(dataset, OfflineDataset) else dataset
    initial_states = tensors["s0"]

    for policy in target_policies:
        value_net = train_value_fqe_state(
            dataset,
            target_policy=policy,
            behavior_policy=behavior_policy,
            gamma=gamma,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            hidden=hidden,
        )
        value_networks[policy.name] = value_net
        estimate = float(value_net(torch.tensor(initial_states, dtype=torch.float32, device=DEVICE)).mean().item())
        estimates[policy.name] = estimate
        if report:
            print(f"[{policy.name}] J^pi (FVE on s0): {estimate:.3f}")
    return value_networks, estimates


def run_q_fqe_block(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policies: Sequence[GaussianLinearPolicy],
    gamma: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    action_samples: int,
    eval_samples: int,
    hidden: int = 128,
    report: bool = True,
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, float]]:

    print("\n=== FQE estimate ===")

    q_networks: Dict[str, torch.nn.Module] = {}
    estimates: Dict[str, float] = {}
    tensors = dataset.as_dict() if isinstance(dataset, OfflineDataset) else dataset
    initial_states = tensors["s0"]

    for policy in target_policies:
        q_net = train_q_fqe(
            dataset,
            target_policy=policy,
            gamma=gamma,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            samples=action_samples,
            hidden=hidden,
        )
        q_networks[policy.name] = q_net
        estimate = estimate_V_from_Q_on_s0(q_net, initial_states, policy, K=eval_samples, device=DEVICE)
        estimates[policy.name] = estimate
        if report:
            print(f"[{policy.name}] J^pi (FQE on s0): {estimate:.3f}")
    return q_networks, estimates


def run_value_aware_block(
    env,
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policies: Sequence[GaussianLinearPolicy],
    value_networks: Mapping[str, torch.nn.Module],
    gamma: float,
    lambda_td: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    horizon: int,
    episodes: int,
    hidden: int = 128,
) -> Dict[str, float]:
    print("\n=== FVE-aware model-based estimate ===")
    estimates: Dict[str, float] = {}
    for policy in target_policies:
        value_net = value_networks[policy.name]
        dynamics = train_value_aware_model(
            dataset,
            target_policy=policy,
            value_fn=value_net,
            gamma=gamma,
            lambda_td=lambda_td,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            hidden=hidden,
        )
        mean, stderr = evaluate_with_model_rollouts(
            env,
            dynamics,
            policy,
            episodes=episodes,
            horizon=horizon,
            gamma=gamma,
            seed=seed,
        )
        print(f"[{policy.name}] J^pi (FVE-aware model): {mean:.3f} ± {1.96 * stderr:.3f}")
        estimates[policy.name] = mean
    return estimates


def run_q_aware_block(
    env,
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policies: Sequence[GaussianLinearPolicy],
    q_networks: Mapping[str, torch.nn.Module],
    gamma: float,
    lambda_td: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    horizon: int,
    episodes: int,
    action_samples: int,
    hidden: int = 128,
) -> Dict[str, float]:
    print("\n=== FQE-aware model-based estimate ===")
    estimates: Dict[str, float] = {}
    for policy in target_policies:
        q_net = q_networks[policy.name]
        dynamics = train_q_aware_model(
            dataset,
            target_policy=policy,
            q_fn=q_net,
            gamma=gamma,
            lambda_td=lambda_td,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            samples=action_samples,
            hidden=hidden,
        )
        mean, stderr = evaluate_with_model_rollouts(
            env,
            dynamics,
            policy,
            episodes=episodes,
            horizon=horizon,
            gamma=gamma,
            seed=seed,
        )
        print(f"[{policy.name}] J^pi (FQE-aware model): {mean:.3f} ± {1.96 * stderr:.3f}")
        estimates[policy.name] = mean
    return estimates


def run_ranking_aware_block(
    env,
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policies: Sequence[GaussianLinearPolicy],
    q_networks: Mapping[str, torch.nn.Module],
    gamma: float,
    lambda_rank: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    horizon: int,
    episodes: int,
    rollout_horizon: int,
    rollout_episodes: int,
    hidden: int = 128,
) -> Dict[str, float]:
    print("\n=== Ranking-aware model-based estimate ===")
    policy_q_pairs = [(policy, q_networks[policy.name]) for policy in target_policies]
    dynamics = train_ranking_aware_model(
        dataset,
        policy_q_pairs,
        gamma=gamma,
        lambda_rank=lambda_rank,
        rollout_horizon=rollout_horizon,
        rollout_episodes=rollout_episodes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        hidden=hidden,
    )

    estimates: Dict[str, float] = {}
    for policy in target_policies:
        mean, stderr = evaluate_with_model_rollouts(
            env,
            dynamics,
            policy,
            episodes=episodes,
            horizon=horizon,
            gamma=gamma,
            seed=seed,
        )
        print(f"[{policy.name}] J^pi (ranking-aware model): {mean:.3f} ± {1.96 * stderr:.3f}")
        estimates[policy.name] = mean
    return estimates


def run_ground_truth_block(
    env,
    target_policies: Sequence[GaussianLinearPolicy],
    gamma: float,
    horizon: int,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    print("\n=== Ground-truth ===")
    estimates: Dict[str, float] = {}
    for policy in target_policies:
        mean, stderr = evaluate_policy_in_env(
            env,
            policy,
            episodes=episodes,
            horizon=horizon,
            gamma=gamma,
            seed=seed,
        )
        print(f"[{policy.name}] True J^pi: {mean:.3f} ± {1.96 * stderr:.3f}")
        estimates[policy.name] = mean
    return estimates


def compare_estimates_to_ground_truth(
    ground_truth: Mapping[str, float],
    method_estimates: Mapping[str, Mapping[str, float]],
) -> None:
    if not ground_truth:
        print("\n[Compare] Ground-truth estimates unavailable; skipping comparison.")
        return

    keys = set(ground_truth.keys())
    print("\n=== Method Comparison vs. Ground Truth ===")
    for method, estimates in method_estimates.items():
        overlap = keys.intersection(estimates.keys())
        if not overlap:
            print(f"[Compare] {method}: no overlapping policies with ground truth.")
            continue
        diffs = [abs(estimates[name] - ground_truth[name]) for name in overlap]
        avg_diff = float(np.mean(diffs)) if diffs else float("nan")
        print(f"[Compare] {method}: avg |estimate - ground_truth| = {avg_diff:.3f} over {len(overlap)} policies")
