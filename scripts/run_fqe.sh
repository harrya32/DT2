#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
SEEDS=("${DEFAULT_SEEDS[@]}")
EXTRA_ARGS=()
OUTPUT_ROOT="results/fqe-only"

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
    echo "Running seed ${seed}..."
    python exps/pendulum_runner.py --fqe-only --force-q-training --q-epochs 200 --eval-rollouts 100 --seed "$seed" --output-dir "${OUTPUT_ROOT}/pendulum_pipeline" "${EXTRA_ARGS[@]}"
    #python exps/lunarlander_runner.py --fqe-only --force-q-training --q-epochs 500 --eval-rollouts 100 --seed "$seed" --output-dir "${OUTPUT_ROOT}/lunarlander_pipeline" "${EXTRA_ARGS[@]}"
    #python exps/walker_runner.py --fqe-only --force-q-training --q-epochs 200 --eval-rollouts 20 --seed "$seed" --output-dir "${OUTPUT_ROOT}/walker_pipeline" "${EXTRA_ARGS[@]}"
    #python exps/cheetah_runner.py --fqe-only --force-q-training --q-epochs 200 --eval-rollouts 20 --seed "$seed" --output-dir "${OUTPUT_ROOT}/cheetah_pipeline" "${EXTRA_ARGS[@]}"

    echo "Completed seed ${seed}"
    echo
done
