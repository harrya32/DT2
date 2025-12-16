#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(0 1 2 3 4)
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
    echo "Running seed ${seed}..."
    #python exps/lunarlander_pipeline.py --force-dynamics-training --dyn-early-stop-patience 50 --lambda-rank 0.5 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/pendulum_pipeline.py --force-q-training --force-dynamics-training --skip-sup-model --dyn-early-stop-patience 100 --lambda-rank 0.1 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/anisotropic_pipeline.py --force-q-training --force-dynamics-training --dyn-early-stop-patience 100 --lambda-rank 0.1 --seed "$seed" "${EXTRA_ARGS[@]}"
    echo "Completed seed ${seed}"
    echo
done
