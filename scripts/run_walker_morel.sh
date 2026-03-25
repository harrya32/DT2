#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(2000 2001 2002)
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
    python exps/walker_runner.py --dynamics-models morel --force-dynamics-training --eval-rollouts 20 --seed "$seed" 

    echo "[walker_pipeline] Completed seed ${seed}"
    echo
done
