#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)
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
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "mlp" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "ode" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "resnet" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "transformer" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "gru" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    python exps/pendulum_runner.py --dynamics-loss "mse" --backbone "gru" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/pendulum_runner.py --dynamics-loss "mse" --backbone "mlp" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/pendulum_runner.py --dynamics-loss "mse" --backbone "resnet" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/pendulum_runner.py --dynamics-loss "mse" --backbone "transformer" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/pendulum_runner.py --dynamics-loss "mse" --backbone "ode" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    python exps/hopper_runner.py --dynamics-loss "mse" --backbone "gru" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/hopper_runner.py --dynamics-loss "mse" --backbone "mlp" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/hopper_runner.py --dynamics-loss "mse" --backbone "resnet" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/hopper_runner.py --dynamics-loss "mse" --backbone "transformer" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/hopper_runner.py --dynamics-loss "mse" --backbone "ode" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    echo "Completed seed ${seed}"
    echo
done
