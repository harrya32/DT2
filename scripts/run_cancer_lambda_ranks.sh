#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(200 201 202 203 204)
SEEDS=("${DEFAULT_SEEDS[@]}")
LAMBDA_RANKS=(1)
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
        -l|--lambda-ranks)
            if [[ $# -lt 2 ]]; then
                echo "Error: --lambda-ranks option requires a comma-separated list" >&2
                exit 1
            fi
            IFS=',' read -r -a LAMBDA_RANKS <<< "$2"
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

if [[ ${#LAMBDA_RANKS[@]} -eq 0 ]]; then
    echo "Error: no lambda-rank values provided" >&2
    exit 1
fi

for lambda_rank in "${LAMBDA_RANKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "[cancer_pipeline] Running seed ${seed} with --lambda-rank ${lambda_rank}..."
        python exps/cancer_pipeline.py --dyn-hidden-dim 64 --dynamics-loss "mse" --rollout-steps 1000 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 10 --seed "$seed" --lambda-rank "$lambda_rank" "${EXTRA_ARGS[@]}"
        echo "[cancer_pipeline] Completed seed ${seed} with --lambda-rank ${lambda_rank}"
        echo
    done
done
