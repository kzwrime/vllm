#!/bin/bash
# Preset: Qwen3-30B-A3B-Instruct-2507
# Configuration: DP=2, TP=2, PP=1, real EPLB with post-profiling rebalance
# MPI Processes: 4 (DP * TP * PP = 2 * 2 * 1)
#
# This preset enables EPLB in real mode (weight rearrangement ON):
# - step_interval is set very large to avoid auto-triggering during bench runs
# - After the profiler dumps statistics, rebalance_after_statistics triggers
#   an immediate rearrangement so the second bench run benefits from it
# - num_redundant_experts=32 for better load distribution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../"

# 在加载模板前设置独立配置项
export PD_MODE="MIXED"

# 加载基础模板配置
source "$SCRIPT_DIR/user_env_template.sh"

# 覆盖必要配置
export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
export USER_VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export VLLM_USE_MPI_COORD=1
export VLLM_CPU_USE_MPI=1
export VLLM_ALL2ALL_BACKEND_XCPU="mpi_alltoallv"

# EPLB Real Mode with Rebalance-After-Statistics
_VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"
_VLLM_OPTIONAL_ARGS+=" --enable-eplb"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.window_size 200"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.step_interval 100000"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.num_redundant_experts 32"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.log_balancedness true"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.log_balancedness_interval 20"
_VLLM_OPTIONAL_ARGS+=" --eplb-config.rebalance_after_profiler_stop true"

export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

# Use mpi4py for EPLB weight-exchange P2P (requires VLLM_CPU_USE_MPI=1, already set above)
export VLLM_EPLB_COMM_BACKEND="mpi"

echo "========================================="
echo "  Preset: qwen30b_a3b_dp2_tp2_ep_eplb_rebalance"
echo "  DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
echo "  EPLB Mode: REAL (weight rearrangement enabled)"
echo "  - step_interval=100000 (no auto-trigger during bench)"
echo "  - rebalance_after_profiler_stop=true (rebalance after profiler stops)"
echo "  - num_redundant_experts=32"
echo "  - VLLM_EPLB_COMM_BACKEND=mpi"
echo "========================================="
