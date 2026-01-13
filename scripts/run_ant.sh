#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(5 6 7 8 9 10 11 12 13 14)
SEEDS=("${DEFAULT_SEEDS[@]}")

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
        *)
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "Error: no seeds provided" >&2
    exit 1
fi

for seed in "${SEEDS[@]}"; do
    echo "[ant_pipeline] Running seed ${seed}..."
    python exps/ant_runner.py --backbone "resnet" --lambda-rank 0.1 --rank-rollout-horizon 50 --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    
    python exps/ant_runner.py --backbone "ode" --lambda-rank 0.1 --rank-rollout-horizon 50 --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"

    python exps/ant_runner.py --backbone "mlp" --lambda-rank 0.1 --rank-rollout-horizon 50 --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"

    python exps/ant_runner.py --backbone "transformer" --lambda-rank 0.1 --rank-rollout-horizon 50 --dynamics-models supervised kendall --q-epochs 200 --total-steps 1000000 --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" 

    python exps/ant_runner.py --backbone "gru" --lambda-rank 0.1 --rank-rollout-horizon 50 --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    
    echo "[ant_pipeline] Completed seed ${seed}"
    echo
done
