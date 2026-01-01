#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_SEEDS=(0 1 2)
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
    echo "[lunarlander_pipeline] Running seed ${seed}..."
    python exps/lunarlander_pipeline.py --backbone "transformer" --dyn-seq-len 8 --force-dynamics-training --seed "$seed" "${EXTRA_ARGS[@]}"
    echo "[lunarlander_pipeline] Completed seed ${seed}"
    echo
done
