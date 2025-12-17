#!/bin/bash

# Run POMDP pipeline experiments with multiple seeds
# This tests the hypothesis that ranking-aware loss outperforms NLL
# when dynamics are partially observable (model misspecification)

set -e

cd "$(dirname "$0")/.."

# Default parameters
SEEDS=(0 1 2 3 4)
OUTPUT_BASE="results/pomdp_pipeline"
WANDB_PROJECT="DT2-pomdp"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-disabled)
            WANDB_MODE="disabled"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

WANDB_MODE=${WANDB_MODE:-online}

echo "Running POMDP pipeline with seeds: ${SEEDS[*]}"
echo "Output base: $OUTPUT_BASE"
echo "W&B project: $WANDB_PROJECT (mode: $WANDB_MODE)"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "============================================"
    echo "Running seed $seed"
    echo "============================================"
    
    python exps/pomdp_pipeline.py \
        --seed "$seed" \
        --output-dir "${OUTPUT_BASE}/seed_${seed}" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-mode "$WANDB_MODE" \
        --wandb-run-name "pomdp_seed_${seed}" \
        --analyze-variance
    
    echo "Completed seed $seed"
    echo ""
done

echo "All seeds completed!"
