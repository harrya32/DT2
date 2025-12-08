from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

EXPS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPS_DIR.parent
if str(EXPS_DIR) not in sys.path:
    sys.path.append(str(EXPS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import lunarlander_pipeline as pipeline
from lunarlander_pipeline import (
    default_device,
    load_dynamics_models,
    load_policy_models,
    load_snapshots,
    make_policy_adapters,
)
from src.networks import DynamicsNet
from src.policies import TorchPolicy


def get_start_state(env_id: str, seed: int) -> np.ndarray:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    env.close()
    return np.asarray(obs, dtype=np.float32)


def rollout_true_env(
    policy_model,
    env_id: str,
    steps: int,
    seed: int,
    deterministic: bool,
    expected_state: np.ndarray,
) -> np.ndarray:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    if not np.allclose(obs, expected_state, atol=1e-5):
        print("[warn] true env reset state differed from reference; using env state instead")
    states = [obs.copy()]
    for _ in range(steps):
        action, _ = policy_model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        states.append(obs.copy())
        if terminated or truncated:
            break
    env.close()
    return np.stack(states)


def rollout_in_dynamics(
    policy: TorchPolicy,
    dynamics: DynamicsNet,
    start_state: np.ndarray,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    states = [start_state.copy()]
    state = torch.tensor(start_state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            action = policy.sample_torch_actions(state, deterministic=True)
            next_state = dynamics.sample_next(state, action)
            state = next_state
            states.append(next_state.squeeze(0).cpu().numpy())
    return np.stack(states)


def save_state_plot(states: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(states.shape[0])
    state_dim = states.shape[1]
    fig, axes = plt.subplots(state_dim, 1, figsize=(10, 2.2 * state_dim), sharex=True)
    if state_dim == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.plot(steps, states[:, idx], label=f"state[{idx}]")
        ax.set_ylabel(f"s{idx}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Environment Step")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def visualize_trajectories(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device != "auto" else default_device()

    policy_manifest = args.output_dir / "policies" / "snapshots.json"
    if not policy_manifest.exists():
        raise FileNotFoundError(f"Missing policy checkpoints at {policy_manifest}")
    dynamics_dir = args.output_dir / "dynamics"
    manifest = dynamics_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing dynamics manifest at {manifest}")

    snapshots = load_snapshots(policy_manifest)
    policy_models = load_policy_models(snapshots)
    torch_policies = make_policy_adapters(policy_models)
    sup_model, q_aware_models, ranking_model, _ = load_dynamics_models(dynamics_dir, device)

    start_state = get_start_state(args.env_id, args.seed)
    print(f"Using start state from env seed {args.seed}: {start_state}")

    plot_dir = args.output_dir / "trajectory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for snap in snapshots:
        name = snap["name"]
        print(f"Generating trajectories for policy {name}...")
        sb3_model = policy_models[name]
        torch_policy = torch_policies[name]

        true_states = rollout_true_env(
            sb3_model,
            env_id=args.env_id,
            steps=args.rollout_steps,
            seed=args.seed,
            deterministic=not args.stochastic,
            expected_state=start_state,
        )
        save_state_plot(true_states, f"{name} - True Env", plot_dir / f"{name}_true_env.png")

        sup_states = rollout_in_dynamics(torch_policy, sup_model, start_state, args.rollout_steps, device)
        save_state_plot(sup_states, f"{name} - Dynamics Supervised", plot_dir / f"{name}_dyn_supervised.png")

        if name in q_aware_models:
            q_dyn_states = rollout_in_dynamics(torch_policy, q_aware_models[name], start_state, args.rollout_steps, device)
            save_state_plot(q_dyn_states, f"{name} - Dynamics Q-aware", plot_dir / f"{name}_dyn_qaware.png")
        else:
            print(f"[warn] No q-aware dynamics found for policy {name}")

        rank_states = rollout_in_dynamics(torch_policy, ranking_model, start_state, args.rollout_steps, device)
        save_state_plot(rank_states, f"{name} - Dynamics Ranking", plot_dir / f"{name}_dyn_ranking.png")

    print(f"Saved trajectory plots to {plot_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize policy trajectories across real and learned dynamics")
    parser.add_argument("--env-id", default="LunarLanderContinuous-v3")
    parser.add_argument("--output-dir", type=Path, default=Path("results/lunarlander_pipeline"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1234, help="seed for environment reset determining start state")
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--stochastic", action="store_true", help="sample stochastic actions instead of deterministic ones")
    return parser.parse_args(argv)


if __name__ == "__main__":
    visualize_trajectories(parse_args())
