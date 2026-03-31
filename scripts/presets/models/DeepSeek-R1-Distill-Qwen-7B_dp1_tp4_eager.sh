#!/bin/bash
# Preset: DeepSeek-R1-Distill-Qwen-7B
# Configuration: DP=1, TP=4, PP=1, enforce-eager mode
# MPI Processes: 4 (DP * TP * PP = 1 * 4 * 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../"

# 在加载模板前设置独立配置项
export PD_MODE="NOT_MOE"

# 加载基础模板配置
source "$SCRIPT_DIR/user_env_template.sh"

# 覆盖必要配置
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=4
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1

echo "========================================="
echo "  Preset: DeepSeek-R1-Distill-Qwen-7B_dp1_tp4_eager"
echo "  DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
echo "========================================="
