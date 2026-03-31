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
export TORCH_XCPU_ENABLE_CHECK=0
# export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_ALL2ALL_BACKEND_XCPU="mpi_alltoallv"

echo "========================================="
echo "  Preset: qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager"
echo "  DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
echo "========================================="
