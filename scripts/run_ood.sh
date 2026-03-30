#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(0 1 2 3 4)
SEEDS=("${DEFAULT_SEEDS[@]}")
EXTRA_ARGS=()
OUTPUT_DIR="results/ood/walker_pipeline"

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
    python exps/walker_runner.py --backbone "resnet" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --eval-ood-policies --ood-policy-types random const_zero const_min const_max const_mid --ood-eval-rollouts 50 --seed "$seed"  --output-dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
    echo "Completed seed ${seed}"
    echo
done
