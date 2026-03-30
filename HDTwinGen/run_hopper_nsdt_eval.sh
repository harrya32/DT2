#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

ENV_ID="${ENV_ID:-Hopper-v4}"
POLICY_DIR="${POLICY_DIR:-pend-eval/hopper-policies}"
POLICY_SNAPSHOTS="${POLICY_SNAPSHOTS:-}"
MAX_POLICIES="${MAX_POLICIES:-0}"

NSDT_RUN_DIR="${NSDT_RUN_DIR:-}"
LOGS_DIR="${LOGS_DIR:-logs}"
ENV_NAME="${ENV_NAME:-Dataset-Hopper}"
ENV_SEED="${ENV_SEED:-0}"
STATE_DIFF_FILE="${STATE_DIFF_FILE:-best_state_differential.py}"
STATE_DICT_FILE="${STATE_DICT_FILE:-best_state_dict.pt}"

EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-20}"
EVAL_HORIZON="${EVAL_HORIZON:-500}"
GAMMA="${GAMMA:-0.97}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
SIM_BATCH_SIZE="${SIM_BATCH_SIZE:-2048}"
ENV_KWARGS="${ENV_KWARGS:-{}}"

OUTPUT_JSON="${OUTPUT_JSON:-pend-eval/results/nsdt_hopper_alignment_summary.json}"
SAVE_ROLLOUT_RETURNS="${SAVE_ROLLOUT_RETURNS:-false}"
STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-false}"
IGNORE_ENV_DONE="${IGNORE_ENV_DONE:-false}"

ALL_SEEDS="${ALL_SEEDS:-true}"
SEED_START="${SEED_START:-0}"
SEED_RUNS="${SEED_RUNS:-10}"
PER_SEED_OUTPUT_DIR="${PER_SEED_OUTPUT_DIR:-pend-eval/results}"
AGG_OUTPUT_JSON="${AGG_OUTPUT_JSON:-pend-eval/results/nsdt_hopper_alignment_summary_all_seeds.json}"

EXTRA_ARGS=("$@")

build_cmd() {
  local seed="$1"
  local output_json="$2"
  local -a cmd=(
    "$PYTHON_BIN" "pend-eval/eval_nsdt_hopper_alignment.py"
    --env-id "$ENV_ID"
    --policy-dir "$POLICY_DIR"
    --max-policies "$MAX_POLICIES"
    --logs-dir "$LOGS_DIR"
    --env-name "$ENV_NAME"
    --env-seed "$seed"
    --state-diff-file "$STATE_DIFF_FILE"
    --state-dict-file "$STATE_DICT_FILE"
    --eval-rollouts "$EVAL_ROLLOUTS"
    --eval-horizon "$EVAL_HORIZON"
    --gamma "$GAMMA"
    --seed "$SEED"
    --device "$DEVICE"
    --sim-batch-size "$SIM_BATCH_SIZE"
    --env-kwargs "$ENV_KWARGS"
    --output-json "$output_json"
  )

  if [[ -n "$POLICY_SNAPSHOTS" ]]; then
    cmd+=(--policy-snapshots "$POLICY_SNAPSHOTS")
  fi

  if [[ -n "$NSDT_RUN_DIR" ]]; then
    cmd+=(--nsdt-run-dir "$NSDT_RUN_DIR")
  fi

  if [[ "$SAVE_ROLLOUT_RETURNS" == "true" ]]; then
    cmd+=(--save-rollout-returns)
  fi

  if [[ "$STOCHASTIC_POLICY" == "true" ]]; then
    cmd+=(--stochastic-policy)
  fi

  if [[ "$IGNORE_ENV_DONE" == "true" ]]; then
    cmd+=(--ignore-env-done)
  fi

  cmd+=("${EXTRA_ARGS[@]}")
  printf '%s\n' "${cmd[@]}"
}

run_cmd_lines() {
  mapfile -t cmd < <(build_cmd "$1" "$2")
  printf 'Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

if [[ "$ALL_SEEDS" != "true" ]]; then
  run_cmd_lines "$ENV_SEED" "$OUTPUT_JSON"
  exit 0
fi

if [[ -n "$NSDT_RUN_DIR" ]]; then
  echo "ERROR: ALL_SEEDS=true cannot be combined with NSDT_RUN_DIR (single explicit run)." >&2
  exit 1
fi

if (( SEED_RUNS <= 0 )); then
  echo "ERROR: SEED_RUNS must be > 0 when ALL_SEEDS=true." >&2
  exit 1
fi

mkdir -p "$PER_SEED_OUTPUT_DIR"
seed_files=()
for ((offset=0; offset<SEED_RUNS; offset++)); do
  seed=$((SEED_START + offset))
  per_seed_json="${PER_SEED_OUTPUT_DIR}/nsdt_hopper_alignment_seed${seed}.json"
  if run_cmd_lines "$seed" "$per_seed_json"; then
    seed_files+=("$per_seed_json")
  else
    echo "[WARN] Seed ${seed} failed; skipping." >&2
  fi
done

if (( ${#seed_files[@]} == 0 )); then
  echo "[WARN] No seed evaluations succeeded; writing empty aggregate summary." >&2
fi

"$PYTHON_BIN" - "${seed_files[@]}" "$AGG_OUTPUT_JSON" <<'PY'
import json
import math
import re
import statistics
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("Usage: aggregate [seed_json...] <output_json>")

*seed_jsons, output_json = sys.argv[1:]

per_seed = []
for path_str in seed_jsons:
    path = Path(path_str)
    data = json.loads(path.read_text(encoding="utf-8"))
    match = re.search(r"_seed(\d+)\.json$", path.name)
    env_seed = int(match.group(1)) if match else None
    metrics = data.get("metrics", {})
    per_seed.append(
        {
            "env_seed": env_seed,
            "file": path.as_posix(),
            "spearman_rank_corr": metrics.get("spearman_rank_corr"),
            "regret": metrics.get("regret"),
            "policy_selection": data.get("policy_selection"),
        }
    )

def mean_or_none(values):
    vals = [float(v) for v in values if v is not None]
    return None if not vals else float(sum(vals) / len(vals))

def stderr_or_none(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return float(statistics.stdev(vals) / math.sqrt(len(vals)))

spearman_values = [row["spearman_rank_corr"] for row in per_seed]
regret_values = [row["regret"] for row in per_seed]

summary = {
    "aggregate_over_env_seeds": {
        "num_seeds": len(per_seed),
        "spearman_rank_corr_mean": mean_or_none(spearman_values),
        "spearman_rank_corr_stderr": stderr_or_none(spearman_values),
        "regret_mean": mean_or_none(regret_values),
        "regret_stderr": stderr_or_none(regret_values),
    },
    "per_seed": per_seed,
}

out_path = Path(output_json)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Saved all-seeds aggregate to {out_path}")
PY
