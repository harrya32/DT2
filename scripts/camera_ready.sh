#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
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
    echo "[camera_ready] Running seed ${seed}..."

    python exps/lunarlander_runner.py --force-q-training --q-epochs 200 --dynamics-loss "mse" --backbone "gru" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "transformer" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "mlp" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "ode" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --dynamics-loss "mse" --backbone "resnet" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    python exps/lunarlander_runner.py --backbone "gru" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models hinge listnet --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --backbone "transformer" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models hinge listnet --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --backbone "mlp" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models hinge listnet --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --backbone "ode" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models hinge listnet --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --backbone "resnet" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models hinge listnet --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"


    #python exps/lunarlander_runner.py --force-q-training --q-epochs 200 --backbone "gru" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/lunarlander_runner.py --backbone "transformer" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/lunarlander_runner.py --backbone "mlp" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/lunarlander_runner.py --backbone "ode" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/lunarlander_runner.py --backbone "resnet" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    #python exps/pendulum_runner.py --gamma 0.95 --force-q-training --q-epochs 200 --backbone "gru" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 200 --eval-rollouts 20 --eval-horizon 200 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/pendulum_runner.py --gamma 0.95 --backbone "mlp" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 200 --eval-rollouts 20 --eval-horizon 200 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/pendulum_runner.py --gamma 0.95 --backbone "resnet" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 200 --eval-rollouts 20 --eval-horizon 200 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/pendulum_runner.py --gamma 0.95 --backbone "transformer" --dynamics-models supervised kendall --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 200 --eval-rollouts 20 --eval-horizon 200 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/pendulum_runner.py --gamma 0.95 --backbone "ode" --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 200 --eval-rollouts 20 --eval-horizon 200 --seed "$seed" "${EXTRA_ARGS[@]}"

    #python exps/hopper_runner.py --backbone "gru" --force-q-training --q-epochs 200 --dyn-seq-len 8 --dynamics-models supervised kendall --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/hopper_runner.py --backbone "mlp" --dyn-hidden-dim 64 --dynamics-models supervised kendall --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/hopper_runner.py --backbone "resnet" --dyn-hidden-dim 64 --dynamics-models supervised kendall --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/hopper_runner.py --backbone "transformer" --dyn-seq-len 8 --dyn-hidden-dim 64 --dynamics-models supervised kendall --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/hopper_runner.py --backbone "ode" --dyn-hidden-dim 64 --dynamics-models supervised kendall --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    #python exps/cheetah_runner.py --backbone "gru" --force-q-training --q-epochs 200 --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/cheetah_runner.py --backbone "mlp" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/cheetah_runner.py --backbone "resnet" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/cheetah_runner.py --backbone "transformer" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"
    #python exps/cheetah_runner.py --backbone "ode" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" "${EXTRA_ARGS[@]}"

    #python exps/walker_runner.py --backbone "transformer" --force-q-training --q-epochs 200 --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" 
    #python exps/walker_runner.py --backbone "gru" --dyn-seq-len 8 --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    #python exps/walker_runner.py --backbone "mlp" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    #python exps/walker_runner.py --backbone "ode" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    #python exps/walker_runner.py --backbone "resnet" --dyn-hidden-dim 64 --force-dynamics-training --dynamics-models supervised kendall --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"

    #python exps/ant_runner.py --backbone "resnet" --force-q-training --q-epochs 200 --dynamics-models supervised kendall --rank-rollout-horizon 50 --dyn-hidden-dim 64 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"
    #python exps/ant_runner.py --backbone "ode" --dynamics-models supervised kendall -—rank-rollout-horizon 50 -—dyn-hidden-dim 64 -—force-dynamics-training -—dyn-early-stop-patience 20 -—eval-rollouts 20 —seed "$seed"
    #python exps/ant_runner.py -—backbone "mlp" -—dynamics-models supervised kendall -—rank-rollout-horizon 50 -—dyn-hidden-dim 64 -—force-dynamics-training -—dyn-early-stop-patience 20 -—eval-rollouts 20 —seed "$seed"
    #python exps/ant_runner.py --backbone "transformer" --dynamics-models supervised kendall --rank-rollout-horizon 50 --dyn-seq-len 8 --dyn-hidden-dim 256 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed" 
    #python exps/ant_runner.py --backbone "gru" --dynamics-models supervised kendall --rank-rollout-horizon 50 --dyn-seq-len 8 --dyn-hidden-dim 256 --force-dynamics-training --dyn-early-stop-patience 20 --eval-rollouts 20 --seed "$seed"

    echo "[camera_ready] Completed seed ${seed}"
    echo
done
