from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

# Ensure project root is importable when running from the exps/ directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import collect_dataset
from src.env_utils import make_lunarlander_env
from src.ope_methods import (
    compare_estimates_to_ground_truth,
    run_ground_truth_block,
    run_model_based_mc,
    run_q_aware_block,
    run_q_fqe_block,
    run_ranking_aware_block,
    run_value_aware_block,
    run_value_fqe_block,
)
from src.policies import GaussianLinearPolicy
from src.utils import set_seed

METHOD_CHOICES = (
    "model",
    "value",
    "qvalue",
    "value-aware",
    "q-aware",
    "ranking-aware",
    "ground-truth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline OPE proof-of-concept runner for LunarLanderContinuous-v3."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=list(METHOD_CHOICES),
        help="Subset of evaluation methods to run. Defaults to all methods.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed for reproducibility.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of consecutive seeds to run (starting from --seed).",
    )
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Rollout horizon for model rollouts and ground truth.",
    )
    parser.add_argument(
        "--dataset-episodes",
        type=int,
        default=100,
        help="Number of behavior-policy episodes collected for the offline dataset.",
    )
    parser.add_argument(
        "--dataset-max-steps",
        type=int,
        default=500,
        help="Maximum number of steps per behavior-policy episode.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes for model-based and ground-truth estimates.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size used across training routines.",
    )
    parser.add_argument(
        "--dyn-epochs",
        type=int,
        default=200,
        help="Training epochs for the supervised dynamics model.",
    )
    parser.add_argument(
        "--dyn-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the supervised dynamics model.",
    )
    parser.add_argument(
        "--value-epochs",
        type=int,
        default=500,
        help="Training epochs for state-value FQE.",
    )
    parser.add_argument(
        "--value-lr",
        type=float,
        default=1e-3,
        help="Learning rate for state-value FQE.",
    )
    parser.add_argument(
        "--q-epochs",
        type=int,
        default=500,
        help="Training epochs for Q-function FQE.",
    )
    parser.add_argument(
        "--q-lr",
        type=float,
        default=1e-3,
        help="Learning rate for Q-function FQE.",
    )
    parser.add_argument(
        "--q-action-samples",
        type=int,
        default=10,
        help="Number of action samples per next-state when building Q targets.",
    )
    parser.add_argument(
        "--q-eval-samples",
        type=int,
        default=64,
        help="Number of action samples per initial state when estimating values from Q.",
    )
    parser.add_argument(
        "--value-aware-epochs",
        type=int,
        default=200,
        help="Training epochs for the value-aware dynamics model.",
    )
    parser.add_argument(
        "--value-aware-lambda",
        type=float,
        default=0.1,
        help="Weight on the TD consistency term for the value-aware dynamics model.",
    )
    parser.add_argument(
        "--value-aware-lr",
        type=float,
        default=5e-4,
        help="Learning rate for the value-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-epochs",
        type=int,
        default=200,
        help="Training epochs for the Q-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-lr",
        type=float,
        default=5e-4,
        help="Learning rate for the Q-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-lambda",
        type=float,
        default=0.1,
        help="Weight on the TD consistency term for the Q-aware dynamics model.",
    )
    parser.add_argument(
        "--q-aware-action-samples",
        type=int,
        default=4,
        help="Number of target-policy action samples when computing Q-aware TD targets.",
    )
    parser.add_argument(
        "--ranking-aware-epochs",
        type=int,
        default=200,
        help="Training epochs for the ranking-aware dynamics model.",
    )
    parser.add_argument(
        "--ranking-aware-lr",
        type=float,
        default=5e-4,
        help="Learning rate for the ranking-aware dynamics model.",
    )
    parser.add_argument(
        "--ranking-aware-lambda",
        type=float,
        default=0.1,
        help="Weight on the ranking consistency term for the ranking-aware model.",
    )
    parser.add_argument(
        "--ranking-aware-rollout-horizon",
        type=int,
        default=50,
        help="Rollout horizon used inside the ranking-aware objective.",
    )
    parser.add_argument(
        "--ranking-aware-rollout-episodes",
        type=int,
        default=32,
        help="Number of model rollout episodes sampled per ranking-aware update.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    print(f"Selected methods: {', '.join(methods)}")

    seeds = [args.seed + i for i in range(args.num_seeds)]
    method_estimates_acc: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ground_truth_acc: Dict[str, list[float]] = defaultdict(list)

    for run_seed in seeds:
        print(f"\n===== Seed {run_seed} =====")
        set_seed(run_seed)
        env = make_lunarlander_env()
        gamma = args.gamma
        horizon = args.horizon

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        beta = GaussianLinearPolicy(W=0.3 * np.random.randn(act_dim, state_dim), std=0.3, name="behavior")
        targets = [
            GaussianLinearPolicy(W=0.1 * np.random.randn(act_dim, state_dim), std=0.6, name="pi_noisy"),
            GaussianLinearPolicy(W=0.5 * np.random.randn(act_dim, state_dim), std=0.3, name="pi_moderate"),
            GaussianLinearPolicy(W=1.0 * np.random.randn(act_dim, state_dim), std=0.2, name="pi_strong"),
        ]

        print("Behavior:", beta)
        print("Targets:", targets)

        dataset = collect_dataset(
            env,
            beta,
            n_episodes=args.dataset_episodes,
            max_steps=args.dataset_max_steps,
            seed=run_seed,
        )
        dataset_info = dataset.as_dict()
        print(
            "Dataset sizes:",
            {k: v.shape if hasattr(v, "shape") else len(v) for k, v in dataset_info.items()},
        )

        run_method_estimates: Dict[str, Dict[str, float]] = {}
        value_nets = None
        q_nets = None

        if "model" in methods:
            _, model_estimates = run_model_based_mc(
                env,
                dataset,
                targets,
                gamma,
                horizon,
                args.eval_episodes,
                args.batch_size,
                args.dyn_epochs,
                args.dyn_lr,
                run_seed,
            )
            run_method_estimates["model"] = model_estimates

        if "value" in methods:
            value_nets, value_estimates = run_value_fqe_block(
                dataset,
                targets,
                beta,
                gamma,
                args.value_epochs,
                args.batch_size,
                args.value_lr,
                run_seed,
                report=True,
            )
            run_method_estimates["value"] = value_estimates

        if "qvalue" in methods:
            q_nets, qvalue_estimates = run_q_fqe_block(
                dataset,
                targets,
                gamma,
                args.q_epochs,
                args.batch_size,
                args.q_lr,
                run_seed,
                args.q_action_samples,
                args.q_eval_samples,
                report=True,
            )
            run_method_estimates["qvalue"] = qvalue_estimates

        if "value-aware" in methods:
            if value_nets is None:
                value_nets, _ = run_value_fqe_block(
                    dataset,
                    targets,
                    beta,
                    gamma,
                    args.value_epochs,
                    args.batch_size,
                    args.value_lr,
                    run_seed,
                    report=False,
                )
                print("\nPrepared state-value networks for value-aware modeling.")
            value_aware_estimates = run_value_aware_block(
                env,
                dataset,
                targets,
                value_nets,
                gamma,
                args.value_aware_lambda,
                args.value_aware_epochs,
                args.batch_size,
                args.value_aware_lr,
                run_seed,
                horizon,
                args.eval_episodes,
            )
            run_method_estimates["value-aware"] = value_aware_estimates

        if "q-aware" in methods:
            if q_nets is None:
                q_nets, _ = run_q_fqe_block(
                    dataset,
                    targets,
                    gamma,
                    args.q_epochs,
                    args.batch_size,
                    args.q_lr,
                    run_seed,
                    args.q_action_samples,
                    args.q_eval_samples,
                    report=False,
                )
                print("\nPrepared Q networks for Q-aware modeling.")
            q_aware_estimates = run_q_aware_block(
                env,
                dataset,
                targets,
                q_nets,
                gamma,
                args.q_aware_lambda,
                args.q_aware_epochs,
                args.batch_size,
                args.q_aware_lr,
                run_seed,
                horizon,
                args.eval_episodes,
                args.q_aware_action_samples,
            )
            run_method_estimates["q-aware"] = q_aware_estimates

        if "ranking-aware" in methods:
            if q_nets is None:
                q_nets, _ = run_q_fqe_block(
                    dataset,
                    targets,
                    gamma,
                    args.q_epochs,
                    args.batch_size,
                    args.q_lr,
                    run_seed,
                    args.q_action_samples,
                    args.q_eval_samples,
                    report=False,
                )
                print("\nPrepared Q networks for ranking-aware modeling.")
            ranking_estimates = run_ranking_aware_block(
                env,
                dataset,
                targets,
                q_nets,
                gamma,
                args.ranking_aware_lambda,
                args.ranking_aware_epochs,
                args.batch_size,
                args.ranking_aware_lr,
                run_seed,
                horizon,
                args.eval_episodes,
                args.ranking_aware_rollout_horizon,
                args.ranking_aware_rollout_episodes,
            )
            run_method_estimates["ranking-aware"] = ranking_estimates

        ground_truth_estimates: Dict[str, float] = {}
        if "ground-truth" in methods:
            ground_truth_estimates = run_ground_truth_block(
                env,
                targets,
                gamma,
                horizon,
                args.eval_episodes,
                run_seed,
            )

        for method_name, estimates in run_method_estimates.items():
            for policy_name, value in estimates.items():
                method_estimates_acc[method_name][policy_name].append(value)

        for policy_name, value in ground_truth_estimates.items():
            ground_truth_acc[policy_name].append(value)

        env.close()

    method_estimates_avg = {
        method: {policy: float(np.mean(vals)) for policy, vals in policy_dict.items()}
        for method, policy_dict in method_estimates_acc.items()
    }
    ground_truth_avg = {policy: float(np.mean(vals)) for policy, vals in ground_truth_acc.items()}

    if method_estimates_avg:
        print("\n=== Average Estimates Across Seeds ===")
        for method, pol_dict in method_estimates_avg.items():
            print(f"{method}:")
            for policy, val in pol_dict.items():
                print(f"  {policy}: {val:.3f}")

    if ground_truth_avg:
        print("\n=== Ground Truth (Average Across Seeds) ===")
        for policy, val in ground_truth_avg.items():
            print(f"{policy}: {val:.3f}")

    compare_estimates_to_ground_truth(ground_truth_avg, method_estimates_avg)


if __name__ == "__main__":
    main()
