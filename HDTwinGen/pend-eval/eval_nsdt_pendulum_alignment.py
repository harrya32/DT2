#!/usr/bin/env python3
"""Evaluate NSDT Pendulum simulators by policy-value alignment.

This script compares policy values from:
1) true Pendulum-v1 rollouts, and
2) rollouts in a learned NSDT state-differential simulator.

It loads PPO checkpoints from `pend-eval/policies` (or a snapshots manifest),
loads `StateDifferential` + `best_state_dict.pt` from an NSDT log run,
and reports how well estimated policy values align.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # type: ignore[assignment]


PENDULUM_ACT_LOW = -2.0
PENDULUM_ACT_HIGH = 2.0


@dataclass(frozen=True)
class PolicySnapshot:
    name: str
    path: Path
    timesteps: int


def default_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pendulum_reward_np(state: np.ndarray, action: np.ndarray) -> float:
    cos_theta, sin_theta, theta_dot = state
    theta = math.atan2(float(sin_theta), float(cos_theta))
    torque = float(np.clip(np.asarray(action).reshape(-1)[0], PENDULUM_ACT_LOW, PENDULUM_ACT_HIGH))
    return -(
        theta * theta
        + 0.1 * float(theta_dot) * float(theta_dot)
        + 0.001 * torque * torque
    )


def pendulum_reward_torch(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    cos_theta, sin_theta, theta_dot = states.unbind(dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    torque = torch.clamp(actions[..., 0], min=PENDULUM_ACT_LOW, max=PENDULUM_ACT_HIGH)
    return -(theta.pow(2) + 0.1 * theta_dot.pow(2) + 0.001 * torque.pow(2))


def repair_non_finite_with_fallback_np(values: np.ndarray, fallback: np.ndarray) -> Tuple[np.ndarray, int]:
    """Replace only non-finite entries with fallback values (minimal perturbation)."""
    x = np.asarray(values, dtype=np.float32).copy()
    fb = np.asarray(fallback, dtype=np.float32)
    bad = ~np.isfinite(x)
    repaired = int(bad.sum())
    if repaired == 0:
        return x, 0
    x[bad] = fb[bad]
    still_bad = ~np.isfinite(x)
    x[still_bad] = 0.0
    return x, repaired + int(still_bad.sum())


def infer_timesteps_from_name(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_snapshot_path(raw_path: Optional[str], name: str, policy_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    if raw_path:
        raw = Path(raw_path)
        candidates.append(raw)
        if raw.suffix == "":
            candidates.append(raw.with_suffix(".zip"))
        if not raw.is_absolute():
            rel = policy_dir / raw
            candidates.append(rel)
            if rel.suffix == "":
                candidates.append(rel.with_suffix(".zip"))

    if name:
        candidates.append(policy_dir / name)
        candidates.append(policy_dir / f"{name}.zip")

    for path in _dedupe_paths(candidates):
        if path.exists() and path.is_file():
            return path
    return None


def load_policy_snapshots(policy_dir: Path, snapshots_manifest: Optional[Path]) -> List[PolicySnapshot]:
    snapshots: List[PolicySnapshot] = []

    manifest_path = snapshots_manifest
    if manifest_path is None:
        default_manifest = policy_dir / "snapshots.json"
        if default_manifest.exists():
            manifest_path = default_manifest

    if manifest_path is not None and manifest_path.exists():
        raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in raw_data:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            resolved = resolve_snapshot_path(item.get("path"), name, policy_dir)
            if resolved is None:
                continue
            timesteps_raw = item.get("timesteps")
            timesteps = int(timesteps_raw) if timesteps_raw is not None else infer_timesteps_from_name(name)
            snapshots.append(PolicySnapshot(name=name, path=resolved, timesteps=timesteps))

    if snapshots:
        snapshots.sort(key=lambda s: (s.timesteps, s.name))
        return snapshots

    for file_path in sorted(policy_dir.glob("ppo_frac_*.zip")):
        name = file_path.stem
        snapshots.append(
            PolicySnapshot(
                name=name,
                path=file_path,
                timesteps=infer_timesteps_from_name(name),
            )
        )

    snapshots.sort(key=lambda s: (s.timesteps, s.name))
    return snapshots


def discover_latest_nsdt_run(logs_dir: Path, env_name: str, env_seed: int) -> Path:
    candidates: List[Path] = []
    for run_dir in sorted(logs_dir.glob("run-*")):
        candidate = run_dir / env_name / str(env_seed)
        if (candidate / "best_state_differential.py").exists():
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No NSDT run directories with best_state_differential.py found under {logs_dir} "
            f"for env_name={env_name!r}, env_seed={env_seed}."
        )

    # Timestamp is part of the run folder name; lexicographic order works for run-YYYYmmdd-HHMMSS_...
    candidates.sort(key=lambda p: p.parts[-3])
    return candidates[-1]


def load_state_differential_class(code_path: Path):
    prelude = "\n".join(
        [
            "from typing import Tuple",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
        ]
    )
    code = code_path.read_text(encoding="utf-8")
    module = types.ModuleType("nsdt_state_differential")
    exec(prelude + code, module.__dict__)
    if not hasattr(module, "StateDifferential"):
        raise AttributeError(f"StateDifferential class not found in {code_path}")
    return module.StateDifferential


def load_nsdt_model(run_dir: Path, state_diff_file: str, state_dict_file: str, device: torch.device) -> torch.nn.Module:
    code_path = run_dir / state_diff_file
    weights_path = run_dir / state_dict_file
    if not code_path.exists():
        raise FileNotFoundError(f"Missing state differential source: {code_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing state dict checkpoint: {weights_path}")

    model_cls = load_state_differential_class(code_path)
    model = model_cls().to(device)

    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict):
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "[WARN] Non-strict state-dict load: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
    else:
        raise TypeError(f"Expected a state_dict dict in {weights_path}, got {type(state_dict)}")

    model.eval()
    return model


def predict_actions_batch(
    model: Any,
    observations: np.ndarray,
    deterministic: bool,
    batch_size: int,
) -> np.ndarray:
    if observations.ndim != 2:
        raise ValueError(f"Expected observations shape [B, obs_dim], got {observations.shape}")

    actions: List[np.ndarray] = []
    for chunk in np.array_split(observations, max(1, math.ceil(len(observations) / max(1, batch_size)))):
        if chunk.size == 0:
            continue
        act, _ = model.predict(chunk, deterministic=deterministic)
        act = np.asarray(act, dtype=np.float32)
        if act.ndim == 1:
            act = act[:, None]
        actions.append(act)

    if not actions:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(actions, axis=0)


def rollout_true_env(
    policy_model: Any,
    env_id: str,
    reset_seeds: np.ndarray,
    horizon: int,
    gamma: float,
    deterministic_policy: bool,
    respect_env_done: bool,
    env_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    env = gym.make(env_id, **env_kwargs)
    initial_states: List[np.ndarray] = []
    returns: List[float] = []

    try:
        for seed in reset_seeds:
            obs, _ = env.reset(seed=int(seed))
            obs = np.asarray(obs, dtype=np.float32)
            initial_states.append(obs.copy())

            total = 0.0
            discount = 1.0
            for _ in range(horizon):
                action, _ = policy_model.predict(obs, deterministic=deterministic_policy)
                action = np.asarray(action, dtype=np.float32)
                if action.ndim == 0:
                    action = action[None]
                action = np.clip(action, PENDULUM_ACT_LOW, PENDULUM_ACT_HIGH)

                total += discount * pendulum_reward_np(obs, action)
                discount *= gamma

                obs, _, terminated, truncated, _ = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                if respect_env_done and (terminated or truncated):
                    break

            returns.append(total)
    finally:
        env.close()

    return np.asarray(initial_states, dtype=np.float32), np.asarray(returns, dtype=np.float64)


def rollout_learned_model(
    policy_model: Any,
    nsdt_model: torch.nn.Module,
    initial_states: np.ndarray,
    horizon: int,
    gamma: float,
    deterministic_policy: bool,
    sim_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if initial_states.ndim != 2 or initial_states.shape[1] != 3:
        raise ValueError(f"Expected initial_states shape [B, 3], got {initial_states.shape}")

    states = torch.tensor(initial_states, dtype=torch.float32, device=device)
    fallback_states = states.clone()
    total = torch.zeros(states.shape[0], dtype=torch.float32, device=device)
    discount = torch.ones_like(total)
    repaired_state_entries = 0
    repaired_dx_entries = 0

    with torch.no_grad():
        for _ in range(horizon):
            obs_np_raw = states.detach().cpu().numpy()
            fallback_np = fallback_states.detach().cpu().numpy()
            obs_np, repaired = repair_non_finite_with_fallback_np(obs_np_raw, fallback_np)
            repaired_state_entries += repaired
            states = torch.tensor(obs_np, dtype=torch.float32, device=device)

            actions_np = predict_actions_batch(
                policy_model,
                observations=obs_np,
                deterministic=deterministic_policy,
                batch_size=sim_batch_size,
            )
            actions_np = np.clip(actions_np, PENDULUM_ACT_LOW, PENDULUM_ACT_HIGH)
            actions = torch.tensor(actions_np, dtype=torch.float32, device=device)

            rewards = pendulum_reward_torch(states, actions)
            total += discount * rewards
            discount *= gamma

            dx = nsdt_model(states[:, 0], states[:, 1], states[:, 2], actions[:, 0])
            if isinstance(dx, (tuple, list)):
                dx = torch.stack(list(dx), dim=-1)
            elif isinstance(dx, torch.Tensor):
                if dx.ndim == 1:
                    dx = dx.unsqueeze(-1)
            else:
                raise TypeError(f"StateDifferential forward returned unsupported type: {type(dx)}")

            if dx.shape != states.shape:
                raise ValueError(
                    "StateDifferential output shape mismatch: "
                    f"expected {tuple(states.shape)}, got {tuple(dx.shape)}"
                )

            bad_dx = ~torch.isfinite(dx)
            repaired_dx_entries += int(bad_dx.sum().item())
            if bad_dx.any():
                dx = torch.where(bad_dx, torch.zeros_like(dx), dx)

            fallback_states = states
            states = states + dx

    if repaired_state_entries > 0 or repaired_dx_entries > 0:
        print(
            "[WARN] Non-finite simulator outputs repaired minimally: "
            f"state_entries={repaired_state_entries}, dx_entries={repaired_dx_entries}"
        )

    return total.detach().cpu().numpy().astype(np.float64)


def summarize_returns(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "stderr": float("nan"), "n": 0}

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    stderr = float(std / math.sqrt(arr.size)) if arr.size > 0 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "n": int(arr.size),
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    return pearson_corr(rankdata(x), rankdata(y))


def make_policy_metrics(
    policy_names: Sequence[str],
    env_means: np.ndarray,
    sim_means: np.ndarray,
) -> Dict[str, Any]:
    if env_means.size == 0 or sim_means.size == 0:
        raise ValueError("Need at least one policy result to compute metrics.")

    env_ranks = rankdata(-env_means)
    sim_ranks = rankdata(-sim_means)
    env_best_idx = int(np.argmax(env_means))
    sim_best_idx = int(np.argmax(sim_means))
    # Regret of simulator-selected policy measured in the true environment.
    regret = float(env_means[env_best_idx] - env_means[sim_best_idx])
    return {
        "spearman_rank_corr": spearman_corr(env_ranks, sim_ranks),
        "regret": regret,
    }


def get_policy_selection_summary(
    policy_names: Sequence[str],
    env_means: np.ndarray,
    sim_means: np.ndarray,
) -> Dict[str, Any]:
    env_best_idx = int(np.argmax(env_means))
    sim_best_idx = int(np.argmax(sim_means))
    env_ranks = rankdata(-env_means)
    sim_ranks = rankdata(-sim_means)
    return {
        "real_optimal_policy": str(policy_names[env_best_idx]),
        "simulator_optimal_policy": str(policy_names[sim_best_idx]),
        "real_optimal_true_value": float(env_means[env_best_idx]),
        "simulator_optimal_true_value": float(env_means[sim_best_idx]),
        "simulator_optimal_sim_value": float(sim_means[sim_best_idx]),
        "env_rank_vector": {name: float(rank) for name, rank in zip(policy_names, env_ranks)},
        "sim_rank_vector": {name: float(rank) for name, rank in zip(policy_names, sim_ranks)},
    }


def print_results(
    results: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    selection: Dict[str, Any],
) -> None:
    print("\nPolicy value comparison (true env vs NSDT simulator):")
    print(f"{'policy':<20} {'steps':>10} {'env_mean':>12} {'nsdt_mean':>12} {'delta':>12}")
    for row in results:
        env_mean = row["true_env"]["mean"]
        sim_mean = row["nsdt_sim"]["mean"]
        delta = sim_mean - env_mean
        print(
            f"{row['name']:<20} {row['timesteps']:>10d} "
            f"{env_mean:>12.4f} {sim_mean:>12.4f} {delta:>12.4f}"
        )

    spearman_txt = "n/a" if metrics["spearman_rank_corr"] is None else f"{metrics['spearman_rank_corr']:.4f}"
    print("\nRequested metrics:")
    print(f"  Spearman(rank_env, rank_sim): {spearman_txt}")
    print(
        "  Regret (V_env(pi*_env) - V_env(pi*_sim)): "
        f"{metrics['regret']:.4f}"
    )
    print(
        f"  pi*_env={selection['real_optimal_policy']} | "
        f"pi*_sim={selection['simulator_optimal_policy']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NSDT Pendulum simulator value alignment")

    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--policy-dir", type=Path, default=Path("pend-eval/policies"))
    parser.add_argument("--policy-snapshots", type=Path, default=None)
    parser.add_argument("--max-policies", type=int, default=0, help="If >0, keep only the first N checkpoints")

    parser.add_argument(
        "--nsdt-run-dir",
        type=Path,
        default=None,
        help="Path to NSDT run env folder containing best_state_differential.py and best_state_dict.pt",
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--env-name", type=str, default="Dataset-Pendulum")
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--state-diff-file", type=str, default="best_state_differential.py")
    parser.add_argument("--state-dict-file", type=str, default="best_state_dict.pt")

    parser.add_argument("--eval-rollouts", type=int, default=500)
    parser.add_argument("--eval-horizon", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic-policy", action="store_true", help="Use stochastic actions instead of deterministic")
    parser.add_argument("--ignore-env-done", action="store_true", help="Do not stop true-env rollouts on terminated/truncated")
    parser.add_argument("--sim-batch-size", type=int, default=2048)
    parser.add_argument("--env-kwargs", type=json.loads, default={}, help="JSON dict passed to gym.make")

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("pend-eval/results/nsdt_alignment_summary.json"),
    )
    parser.add_argument("--save-rollout-returns", action="store_true", help="Include per-rollout return arrays in output JSON")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if PPO is None:
        raise ImportError(
            "stable_baselines3 is required. Install dependencies from setup/requirements.txt first."
        )

    device = default_device(args.device)

    if args.nsdt_run_dir is None:
        nsdt_run_dir = discover_latest_nsdt_run(args.logs_dir, args.env_name, args.env_seed)
    else:
        nsdt_run_dir = args.nsdt_run_dir

    if not nsdt_run_dir.exists():
        raise FileNotFoundError(f"NSDT run directory not found: {nsdt_run_dir}")

    snapshots = load_policy_snapshots(args.policy_dir, args.policy_snapshots)
    if not snapshots:
        raise FileNotFoundError(
            f"No policy checkpoints found in {args.policy_dir}. "
            "Expected ppo_frac_*.zip files or a valid snapshots manifest."
        )

    if args.max_policies > 0:
        snapshots = snapshots[: args.max_policies]

    nsdt_model = load_nsdt_model(
        run_dir=nsdt_run_dir,
        state_diff_file=args.state_diff_file,
        state_dict_file=args.state_dict_file,
        device=device,
    )

    rng = np.random.default_rng(args.seed)
    reset_seeds = rng.integers(0, 1_000_000_000, size=args.eval_rollouts, dtype=np.int64)

    deterministic_policy = not args.stochastic_policy
    respect_env_done = not args.ignore_env_done

    results: List[Dict[str, Any]] = []
    env_means: List[float] = []
    sim_means: List[float] = []

    for snapshot in snapshots:
        print(f"[Eval] {snapshot.name} ({snapshot.path})")
        policy = PPO.load(snapshot.path.as_posix())

        initial_states, env_returns = rollout_true_env(
            policy_model=policy,
            env_id=args.env_id,
            reset_seeds=reset_seeds,
            horizon=args.eval_horizon,
            gamma=args.gamma,
            deterministic_policy=deterministic_policy,
            respect_env_done=respect_env_done,
            env_kwargs=args.env_kwargs,
        )

        sim_returns = rollout_learned_model(
            policy_model=policy,
            nsdt_model=nsdt_model,
            initial_states=initial_states,
            horizon=args.eval_horizon,
            gamma=args.gamma,
            deterministic_policy=deterministic_policy,
            sim_batch_size=args.sim_batch_size,
            device=device,
        )

        true_stats = summarize_returns(env_returns)
        sim_stats = summarize_returns(sim_returns)
        env_means.append(true_stats["mean"])
        sim_means.append(sim_stats["mean"])

        row: Dict[str, Any] = {
            "name": snapshot.name,
            "checkpoint": snapshot.path.as_posix(),
            "timesteps": int(snapshot.timesteps),
            "true_env": true_stats,
            "nsdt_sim": sim_stats,
            "delta_mean": float(sim_stats["mean"] - true_stats["mean"]),
        }
        if args.save_rollout_returns:
            row["true_env_returns"] = env_returns.tolist()
            row["nsdt_sim_returns"] = sim_returns.tolist()
        results.append(row)

    env_means_arr = np.asarray(env_means, dtype=np.float64)
    sim_means_arr = np.asarray(sim_means, dtype=np.float64)
    policy_names = [row["name"] for row in results]
    metrics = make_policy_metrics(policy_names, env_means_arr, sim_means_arr)
    selection = get_policy_selection_summary(policy_names, env_means_arr, sim_means_arr)

    summary = {
        "config": {
            "env_id": args.env_id,
            "eval_rollouts": args.eval_rollouts,
            "eval_horizon": args.eval_horizon,
            "gamma": args.gamma,
            "seed": args.seed,
            "device": str(device),
            "deterministic_policy": deterministic_policy,
            "respect_env_done": respect_env_done,
            "env_kwargs": args.env_kwargs,
        },
        "artifacts": {
            "policy_dir": args.policy_dir.as_posix(),
            "policy_snapshots": args.policy_snapshots.as_posix() if args.policy_snapshots else None,
            "nsdt_run_dir": nsdt_run_dir.as_posix(),
            "state_diff_file": args.state_diff_file,
            "state_dict_file": args.state_dict_file,
        },
        "metrics": metrics,
        "policy_selection": selection,
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_results(results, metrics, selection)
    print(f"\nSaved summary to {args.output_json}")


if __name__ == "__main__":
    main()
