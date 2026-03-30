#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(4007 4008 4009)
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
    echo "[romi] Running seed ${seed}..."
    
    python exps/pendulum_runner.py --dynamics-models romi --force-dynamics-training --eval-rollouts 20 --romi-epochs 500 --romi-uncertainty-scale 0.1 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/lunarlander_runner.py --dynamics-models romi --force-dynamics-training --eval-rollouts 20 --romi-epochs 500 --romi-uncertainty-scale 0.1 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/hopper_runner.py --dynamics-models romi --force-dynamics-training --eval-rollouts 20 --romi-uncertainty-scale 1.0 --romi-epochs 500 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/walker_runner.py --dynamics-models romi --force-dynamics-training --eval-rollouts 20 --romi-uncertainty-scale 0.01 --romi-epochs 500 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/cheetah_runner.py --dynamics-models romi --force-dynamics-training --eval-rollouts 20 --romi-uncertainty-scale 0.1 --romi-epochs 500 --seed "$seed" "${EXTRA_ARGS[@]}"
    python exps/ant_runner.py --dynamics-models romi --force-dynamics-training --eval-rollouts 20 --romi-epochs 500 --romi-uncertainty-scale 0.1 --seed "$seed" "${EXTRA_ARGS[@]}"

    echo "[romi] Completed seed ${seed}"
    echo
done
