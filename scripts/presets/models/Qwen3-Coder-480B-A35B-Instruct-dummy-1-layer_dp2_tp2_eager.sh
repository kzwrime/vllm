#!/bin/bash
# Preset: Qwen3-30B-A3B-Instruct-2507
# Configuration: DP=2, TP=2, PP=1, enforce-eager mode
# MPI Processes: 4 (DP * TP * PP = 2 * 2 * 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../"

# 在加载模板前设置独立配置项
export PD_MODE="MIXED"

# 加载基础模板配置
source "$SCRIPT_DIR/user_env_template.sh"

# 覆盖必要配置
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="../models/Qwen/Qwen3-Coder-480B-A35B-Instruct-dummy-1-layer"
export USER_VLLM_MAX_MODEL_LEN=2048
export VLLM_CPU_KVCACHE_SPACE=8
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_ALL2ALL_BACKEND_XCPU="mpi_alltoallv"

export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS} --load-format dummy "

echo "========================================="
echo "  Preset: Qwen3-30B-A3B_dp2_tp2_eager"
echo "  DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
echo "========================================="
