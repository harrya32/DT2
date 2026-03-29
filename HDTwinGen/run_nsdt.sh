#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults as needed.
MODEL="${MODEL:-gpt5-azure}"
METHODS="${METHODS:-[NSDT]}"
# Examples:
# ENVS='[Dataset-Pendulum]'
# ENVS='[Dataset-LunarLander,Dataset-Walker,Dataset-Hopper,Dataset-Cheetah,Dataset-Ant]'
ENVS="${ENVS:-[Dataset-Pendulum]}"
SEED_START="${SEED_START:-0}"
SEED_RUNS="${SEED_RUNS:-1}"
GENERATIONS="${GENERATIONS:-2}"
NSDT_PATIENCE="${NSDT_PATIENCE:-2}"
PYTORCH_EPOCHS="${PYTORCH_EPOCHS:-2000}"
PYTORCH_BATCH_SIZE="${PYTORCH_BATCH_SIZE:-1000}"
PYTORCH_LOG_INTERVAL="${PYTORCH_LOG_INTERVAL:-10}"
OPTIMIZATION_PATIENCE="${OPTIMIZATION_PATIENCE:-3}"
OPTIMIZATION_MIN_DELTA="${OPTIMIZATION_MIN_DELTA:-1e-6}"
CUDA="${CUDA:-true}"
DATASET_MAX_WINDOWS="${DATASET_MAX_WINDOWS:-0}"
DATASET_WINDOW_LENGTH="${DATASET_WINDOW_LENGTH:-${PENDULUM_WINDOW_LENGTH:-200}}"
DATASET_USE_WINDOWS="${DATASET_USE_WINDOWS:-false}"

python run.py \
  "run.model=${MODEL}" \
  "setup.methods_to_evaluate=${METHODS}" \
  "setup.envs_to_evaluate=${ENVS}" \
  "setup.seed_start=${SEED_START}" \
  "setup.seed_runs=${SEED_RUNS}" \
  "run.generations=${GENERATIONS}" \
  "run.nsdt_patience=${NSDT_PATIENCE}" \
  "run.pytorch_as_optimizer.epochs=${PYTORCH_EPOCHS}" \
  "run.pytorch_as_optimizer.batch_size=${PYTORCH_BATCH_SIZE}" \
  "run.pytorch_as_optimizer.log_interval=${PYTORCH_LOG_INTERVAL}" \
  "run.optimization.patience=${OPTIMIZATION_PATIENCE}" \
  "run.optimization.min_delta=${OPTIMIZATION_MIN_DELTA}" \
  "run.dataset_max_windows=${DATASET_MAX_WINDOWS}" \
  "run.dataset_use_windows=${DATASET_USE_WINDOWS}" \
  "run.dataset_window_length=${DATASET_WINDOW_LENGTH}" \
  "run.pendulum_window_length=${DATASET_WINDOW_LENGTH}" \
  "setup.cuda=${CUDA}" \
  "$@"
