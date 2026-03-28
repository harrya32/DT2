#!/usr/bin/env python3
"""Shared utilities for NSDT policy-value alignment evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # type: ignore[assignment]


RewardFnNp = Callable[[np.ndarray, np.ndarray], float]
RewardFnTorch = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TerminationFnTorch = Callable[[torch.Tensor], torch.Tensor]

INVALID_REWARD_PENALTY = -1e6
REWARD_ABS_CLIP = 1e6


@dataclass(frozen=True)
class PolicySnapshot:
    name: str
    path: Path
    timesteps: int


@dataclass(frozen=True)
class AlignmentEnvSpec:
    display_name: str
    env_id: str
    dataset_env_name: str
    policy_dir: Path
    output_json: Path
    act_low: float
    act_high: float
    reward_fn_np: RewardFnNp
    reward_fn_torch: RewardFnTorch
    termination_fn_torch: Optional[TerminationFnTorch] = None
    default_env_kwargs: Mapping[str, Any] = field(default_factory=dict)


def default_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Keep auto mode conservative on Apple Silicon: some NSDT models use
    # spectral-norm internals that call ops not implemented on MPS.
    # Users can still force MPS explicitly via --device mps.
    return torch.device("cpu")


def repair_non_finite_with_fallback_np(values: np.ndarray, fallback: np.ndarray) -> Tuple[np.ndarray, int]:
    """Replace only non-finite entries with fallback values."""
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
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dict in {weights_path}, got {type(state_dict)}")

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "[WARN] Non-strict state-dict load: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

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
    num_chunks = max(1, math.ceil(len(observations) / max(1, batch_size)))
    for chunk in np.array_split(observations, num_chunks):
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
    reward_fn_np: RewardFnNp,
    act_low: float,
    act_high: float,
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
                action = np.clip(action, act_low, act_high)

                total += discount * float(reward_fn_np(obs, action))
                discount *= gamma

                obs, _, terminated, truncated, _ = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                if respect_env_done and (terminated or truncated):
                    break

            returns.append(total)
    finally:
        env.close()

    return np.asarray(initial_states, dtype=np.float32), np.asarray(returns, dtype=np.float64)


def _model_forward_arity(nsdt_model: torch.nn.Module) -> Optional[int]:
    try:
        signature = inspect.signature(nsdt_model.forward)
    except (TypeError, ValueError):
        return None

    arity = 0
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            arity += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            return None
    return arity


def _to_delta_tensor(raw_dx: Any, batch_size: int, state_dim: int, device: torch.device) -> torch.Tensor:
    if isinstance(raw_dx, (tuple, list)):
        if len(raw_dx) == 1:
            return _to_delta_tensor(raw_dx[0], batch_size=batch_size, state_dim=state_dim, device=device)
        pieces: List[torch.Tensor] = []
        for item in raw_dx:
            t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=device)
            t = t.to(device=device, dtype=torch.float32)
            if t.ndim == 1:
                if t.shape[0] != batch_size:
                    raise ValueError(f"Differential piece batch mismatch: got {tuple(t.shape)}")
                t = t[:, None]
            elif t.ndim == 2:
                if t.shape[0] != batch_size:
                    raise ValueError(f"Differential piece batch mismatch: got {tuple(t.shape)}")
            else:
                raise ValueError(f"Unsupported differential piece rank: {tuple(t.shape)}")
            pieces.append(t)
        dx = torch.cat(pieces, dim=-1)
        if dx.shape[1] != state_dim:
            raise ValueError(f"Differential output width mismatch: expected {state_dim}, got {dx.shape[1]}")
        return dx

    t = raw_dx if isinstance(raw_dx, torch.Tensor) else torch.as_tensor(raw_dx, device=device)
    t = t.to(device=device, dtype=torch.float32)

    if t.ndim == 1:
        if state_dim == 1 and t.shape[0] == batch_size:
            return t[:, None]
        if batch_size == 1 and t.shape[0] == state_dim:
            return t[None, :]
        raise ValueError(f"Unsupported 1D differential output shape: {tuple(t.shape)}")

    if t.ndim == 2:
        if t.shape == (batch_size, state_dim):
            return t
        if t.shape == (state_dim, batch_size):
            return t.transpose(0, 1).contiguous()
        raise ValueError(
            f"Unsupported 2D differential output shape: got {tuple(t.shape)}, "
            f"expected {(batch_size, state_dim)}"
        )

    if t.shape[0] == batch_size:
        t2 = t.reshape(batch_size, -1)
        if t2.shape[1] == state_dim:
            return t2

    raise ValueError(f"Unsupported differential output shape: {tuple(t.shape)}")


def _call_state_differential(nsdt_model: torch.nn.Module, states: torch.Tensor, actions: torch.Tensor) -> Any:
    state_dim = int(states.shape[1])
    action_dim = int(actions.shape[1])
    state_cols = tuple(states[:, i] for i in range(state_dim))
    action_cols = tuple(actions[:, j] for j in range(action_dim))

    def call_state_action():
        return nsdt_model(states, actions)

    def call_flat_state_action_cols():
        return nsdt_model(*state_cols, *action_cols)

    def call_flat_state_plus_vector_action():
        return nsdt_model(*state_cols, actions)

    def call_flat_state_plus_scalar_action():
        return nsdt_model(*state_cols, actions[:, 0])

    arity = _model_forward_arity(nsdt_model)
    ordered_calls: List[Callable[[], Any]] = []

    if arity == 2:
        ordered_calls.append(call_state_action)
    elif arity == state_dim + action_dim:
        ordered_calls.append(call_flat_state_action_cols)
    elif arity == state_dim + 1 and action_dim == 1:
        ordered_calls.append(call_flat_state_plus_scalar_action)
    elif arity == state_dim + 1:
        ordered_calls.append(call_flat_state_plus_vector_action)

    ordered_calls.extend(
        [
            call_state_action,
            call_flat_state_action_cols,
            call_flat_state_plus_vector_action,
        ]
    )
    if action_dim == 1:
        ordered_calls.append(call_flat_state_plus_scalar_action)

    seen = set()
    unique_calls: List[Callable[[], Any]] = []
    for call in ordered_calls:
        name = call.__name__
        if name in seen:
            continue
        seen.add(name)
        unique_calls.append(call)

    errors: List[str] = []
    for call in unique_calls:
        try:
            return call()
        except TypeError as exc:
            errors.append(f"{call.__name__}: {exc}")

    error_text = "; ".join(errors) if errors else "no callable signatures attempted"
    raise TypeError(f"Unable to call StateDifferential.forward with observed tensors ({error_text})")


def _ensure_bool_mask(values: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, device=device)
    if mask.ndim == 0:
        mask = mask.repeat(batch_size)
    elif mask.ndim > 1:
        mask = mask.reshape(batch_size, -1).any(dim=1)
    if mask.shape[0] != batch_size:
        raise ValueError(f"Termination function returned invalid shape {tuple(mask.shape)} for batch={batch_size}")
    return mask.to(device=device, dtype=torch.bool)


def rollout_learned_model(
    policy_model: Any,
    nsdt_model: torch.nn.Module,
    initial_states: np.ndarray,
    horizon: int,
    gamma: float,
    deterministic_policy: bool,
    sim_batch_size: int,
    device: torch.device,
    reward_fn_torch: RewardFnTorch,
    act_low: float,
    act_high: float,
    termination_fn_torch: Optional[TerminationFnTorch],
    respect_termination: bool,
) -> np.ndarray:
    if initial_states.ndim != 2:
        raise ValueError(f"Expected initial_states shape [B, obs_dim], got {initial_states.shape}")

    states = torch.tensor(initial_states, dtype=torch.float32, device=device)
    fallback_states = states.clone()
    total = torch.zeros(states.shape[0], dtype=torch.float32, device=device)
    discount = torch.ones_like(total)
    alive = torch.ones_like(total, dtype=torch.bool)
    repaired_state_entries = 0
    repaired_dx_entries = 0
    repaired_reward_entries = 0

    with torch.no_grad():
        for _ in range(horizon):
            if respect_termination and not alive.any():
                break

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
            actions_np = np.clip(actions_np, act_low, act_high)
            actions = torch.tensor(actions_np, dtype=torch.float32, device=device)

            rewards = reward_fn_torch(states, actions)
            bad_rewards = ~torch.isfinite(rewards)
            repaired_reward_entries += int(bad_rewards.sum().item())
            if bad_rewards.any():
                rewards = torch.where(
                    bad_rewards,
                    torch.full_like(rewards, INVALID_REWARD_PENALTY),
                    rewards,
                )
                if respect_termination:
                    alive = alive & ~bad_rewards
            rewards = torch.clamp(rewards, min=-REWARD_ABS_CLIP, max=REWARD_ABS_CLIP)
            total += discount * rewards * alive.float()
            discount *= gamma

            raw_dx = _call_state_differential(nsdt_model, states, actions)
            dx = _to_delta_tensor(raw_dx, batch_size=states.shape[0], state_dim=states.shape[1], device=device)

            bad_dx = ~torch.isfinite(dx)
            repaired_dx_entries += int(bad_dx.sum().item())
            if bad_dx.any():
                dx = torch.where(bad_dx, torch.zeros_like(dx), dx)

            fallback_states = states
            states = states + dx

            if respect_termination and termination_fn_torch is not None:
                terminated = _ensure_bool_mask(
                    termination_fn_torch(states),
                    batch_size=states.shape[0],
                    device=device,
                )
                alive = alive & ~terminated

    if repaired_state_entries > 0 or repaired_dx_entries > 0 or repaired_reward_entries > 0:
        print(
            "[WARN] Non-finite simulator outputs repaired minimally: "
            f"state_entries={repaired_state_entries}, "
            f"dx_entries={repaired_dx_entries}, "
            f"reward_entries={repaired_reward_entries}"
        )

    total_np = total.detach().cpu().numpy().astype(np.float64)
    bad_total = ~np.isfinite(total_np)
    if bad_total.any():
        repaired_total = int(bad_total.sum())
        total_np[bad_total] = INVALID_REWARD_PENALTY
        print(
            "[WARN] Non-finite rollout returns repaired: "
            f"returns={repaired_total}"
        )
    return total_np


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
    env_means: np.ndarray,
    sim_means: np.ndarray,
) -> Dict[str, Any]:
    if env_means.size == 0 or sim_means.size == 0:
        raise ValueError("Need at least one policy result to compute metrics.")

    finite_env = np.isfinite(env_means)
    finite_sim = np.isfinite(sim_means)

    env_for_select = np.where(finite_env, env_means, -np.inf)
    sim_for_select = np.where(finite_sim, sim_means, -np.inf)

    env_best_idx = int(np.argmax(env_for_select))
    sim_best_idx = int(np.argmax(sim_for_select))

    if np.isfinite(env_means[env_best_idx]) and np.isfinite(env_means[sim_best_idx]):
        regret = float(env_means[env_best_idx] - env_means[sim_best_idx])
    else:
        regret = float("nan")

    valid_pairs = finite_env & finite_sim
    if int(valid_pairs.sum()) >= 2:
        env_ranks = rankdata(-env_means[valid_pairs])
        sim_ranks = rankdata(-sim_means[valid_pairs])
        spearman = spearman_corr(env_ranks, sim_ranks)
    else:
        spearman = None

    return {
        "spearman_rank_corr": spearman,
        "regret": regret,
    }


def get_policy_selection_summary(
    policy_names: Sequence[str],
    env_means: np.ndarray,
    sim_means: np.ndarray,
) -> Dict[str, Any]:
    finite_env = np.isfinite(env_means)
    finite_sim = np.isfinite(sim_means)

    env_for_select = np.where(finite_env, env_means, -np.inf)
    sim_for_select = np.where(finite_sim, sim_means, -np.inf)

    env_best_idx = int(np.argmax(env_for_select))
    sim_best_idx = int(np.argmax(sim_for_select))

    env_for_rank = np.where(finite_env, env_means, -np.inf)
    sim_for_rank = np.where(finite_sim, sim_means, -np.inf)
    env_ranks = rankdata(-env_for_rank)
    sim_ranks = rankdata(-sim_for_rank)
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
    print(f"  Regret (V_env(pi*_env) - V_env(pi*_sim)): {metrics['regret']:.4f}")
    print(
        f"  pi*_env={selection['real_optimal_policy']} | "
        f"pi*_sim={selection['simulator_optimal_policy']}"
    )


def parse_env_kwargs(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON for --env-kwargs: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--env-kwargs must decode to a JSON object")
    return parsed


def _normalize_env_kwargs(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return parse_env_kwargs(value)
    raise TypeError(f"Unsupported env_kwargs value type: {type(value)}")


def create_arg_parser(spec: AlignmentEnvSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Evaluate NSDT {spec.display_name} simulator value alignment")

    parser.add_argument("--env-id", type=str, default=spec.env_id)
    parser.add_argument("--policy-dir", type=Path, default=spec.policy_dir)
    parser.add_argument("--policy-snapshots", type=Path, default=None)
    parser.add_argument("--max-policies", type=int, default=0, help="If >0, keep only the first N checkpoints")

    parser.add_argument(
        "--nsdt-run-dir",
        type=Path,
        default=None,
        help="Path to NSDT run env folder containing best_state_differential.py and best_state_dict.pt",
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--env-name", type=str, default=spec.dataset_env_name)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--state-diff-file", type=str, default="best_state_differential.py")
    parser.add_argument("--state-dict-file", type=str, default="best_state_dict.pt")

    parser.add_argument("--eval-rollouts", type=int, default=500)
    parser.add_argument("--eval-horizon", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stochastic-policy", action="store_true", help="Use stochastic actions instead of deterministic")
    parser.add_argument("--ignore-env-done", action="store_true", help="Do not stop rollouts on terminated/truncated")
    parser.add_argument("--sim-batch-size", type=int, default=2048)
    parser.add_argument(
        "--env-kwargs",
        type=parse_env_kwargs,
        default=dict(spec.default_env_kwargs),
        help="JSON dict passed to gym.make",
    )

    parser.add_argument("--output-json", type=Path, default=spec.output_json)
    parser.add_argument("--save-rollout-returns", action="store_true", help="Include per-rollout return arrays in output JSON")
    return parser


def run_alignment_eval(args: argparse.Namespace, spec: AlignmentEnvSpec) -> None:
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
    env_kwargs = _normalize_env_kwargs(args.env_kwargs)

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
            reward_fn_np=spec.reward_fn_np,
            act_low=spec.act_low,
            act_high=spec.act_high,
            env_kwargs=env_kwargs,
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
            reward_fn_torch=spec.reward_fn_torch,
            act_low=spec.act_low,
            act_high=spec.act_high,
            termination_fn_torch=spec.termination_fn_torch,
            respect_termination=respect_env_done,
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
    metrics = make_policy_metrics(env_means_arr, sim_means_arr)
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
            "env_kwargs": env_kwargs,
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
