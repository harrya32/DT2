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

LAMBDA_VALS=(0.25 0.5 0.75)

for lambda in "${LAMBDA_VALS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running lambda=${lambda}, seed=${seed}..."
    
    python exps/lunarlander_runner.py --lambda-rank "$lambda" --force-q-training --q-epochs 200 --record-lambda-val --backbone "resnet" --dynamics-models kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/hopper_runner.py --lambda-rank "$lambda" --record-lambda-val --backbone "resnet" --dynamics-models kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/walker_runner.py --lambda-rank "$lambda" --record-lambda-val --backbone "resnet" --dynamics-models kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/cheetah_runner.py --lambda-rank "$lambda" --record-lambda-val --backbone "resnet" --dynamics-models kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    #python exps/pendulum_runner.py --lambda-rank "$lambda" --record-lambda-val --backbone "resnet" --dynamics-models kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/ant_runner.py --lambda-rank "$lambda" --record-lambda-val --backbone "resnet" --rank-rollout-horizon 50 --dynamics-models kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    echo "Completed lambda=${lambda}, seed=${seed}"
    echo
  done
done
