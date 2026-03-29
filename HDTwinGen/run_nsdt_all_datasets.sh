#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Runs NSDT on all six offline-control datasets using the same settings as run_nsdt.sh.
ALL_ENVS="${ALL_ENVS:-[Dataset-Hopper,Dataset-Cheetah,Dataset-Walker,Dataset-Ant,Dataset-Pendulum,Dataset-LunarLander]}"

echo "Running NSDT for datasets: ${ALL_ENVS}"
ENVS="${ALL_ENVS}" ./run_nsdt.sh "$@"
