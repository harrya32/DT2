#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(1 2 3 4 5 6 7 8 9)
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
    echo "[walker_pipeline] Running seed ${seed}..."
    python exps/walker_runner.py --backbone "transformer" --force-q-training --force-policy-training --q-epochs 200 --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" 
    
    python exps/walker_runner.py --backbone "gru" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"

    #python exps/walker_runner.py --backbone "mlp" --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"#

    #python exps/walker_runner.py --backbone "ode" --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"

    #python exps/walker_runner.py --backbone "resnet" --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"

    echo "[walker_pipeline] Completed seed ${seed}"
    echo
done
