#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(5000 5001 5002 5003 5004)
SEEDS=("${DEFAULT_SEEDS[@]}")
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--seeds)
            if [[ $# -lt 2 ]]; then
                echo "Error: --seeds option requires a comma-separated list" >&2
                exit 1
            fi
            IFS=',' read -r -a SEEDS <<< "$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "Error: no seeds provided" >&2
    exit 1
fi

for seed in "${SEEDS[@]}"; do
    echo "[omd] Running seed ${seed}..."

    COMMON_ARGS=(
        --dynamics-models omd
        --force-dynamics-training
        --eval-rollouts 20
        --omd-inner-updates 1
        --omd-action-samples 8
        --seed "$seed"
    )

    python exps/pendulum_runner.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    python exps/hopper_runner.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    python exps/walker_runner.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    python exps/cheetah_runner.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
    python exps/ant_runner.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

    echo "[omd] Completed seed ${seed}"
    echo
done

