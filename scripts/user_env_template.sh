export USER_VLLM_LOGGING_LEVEL="DEBUG"

export VLLM_TORCH_PROFILER_DIR="./profile"
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_WITH_FLOPS=1

export USER_VLLM_MODEL="/home/mpiuser/.cache/modelscope/hub/models/facebook/opt-125m"
export USER_VLLM_MAX_MODEL_LEN=2048
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=2048
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_DATA_PARALLEL_ADDRESS=172.33.0.11  # DP 0
export USER_VLLM_DATA_PARALLEL_RPC_IP=172.33.0.10   # HEAD
export USER_VLLM_DATA_PARALLEL_RPC_PORT=13345
export USER_VLLM_PORT=14800
# export VLLM_CPU_KVCACHE_SPACE=4

export USER_VLLM_MPC_SIZE=2
export USER_VLLM_MP_RPC_WORKER_PER_NODE=1
export ExecutorIP=172.33.0.11 # IP addresses vary across different TP groups.

export USER_VLLM_ENFORCE_EAGER=1
export USER_VLLM_ENABLE_EXPERT_PARALLEL=0

export VLLM_CPU_USE_MPI=0
if [[ -z "$VLLM_CPU_USE_MPI" || "$VLLM_CPU_USE_MPI" != "1" ]]; then
    export VLLM_ALL2ALL_BACKEND=naive
fi

export VLLM_LOOPBACK_IP=$(hostname -I | awk '{print $1}')
# export VLLM_LOOPBACK_IP=$(ifconfig eth0 | grep "inet " | awk '{print ^C}')

_VLLM_OPTIONAL_ARGS=""
# _VLLM_OPTIONAL_ARGS+=" --load-format dummy"

# 检查是否启用 --enforce-eager
# 将变量转换为小写，然后检查是否等于 "true", "1", "yes", 或 "on"
EAGER_VAR_LOWER=${USER_VLLM_ENFORCE_EAGER,,}
if [[ "$EAGER_VAR_LOWER" == "true" || "$EAGER_VAR_LOWER" == "1" || "$EAGER_VAR_LOWER" == "yes" || "$EAGER_VAR_LOWER" == "on" ]]; then
    _VLLM_OPTIONAL_ARGS+=" --enforce-eager"
    echo "✅ 启用参数: --enforce-eager"
fi

# 检查是否启用 --enable-expert-parallel
EXPERT_VAR_LOWER=${USER_VLLM_ENABLE_EXPERT_PARALLEL,,}
if [[ "$EXPERT_VAR_LOWER" == "true" || "$EXPERT_VAR_LOWER" == "1" || "$EXPERT_VAR_LOWER" == "yes" || "$EXPERT_VAR_LOWER" == "on" ]]; then
    # 检查USER_VLLM_DATA_PARALLEL_SIZE是否>=1
    if [[ "$USER_VLLM_DATA_PARALLEL_SIZE" -ge 1 ]]; then
        _VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"
        echo "✅ 启用参数: --enable-expert-parallel"
    else
        echo "❌ 错误: USER_VLLM_ENABLE_EXPERT_PARALLEL 设置为1，但 USER_VLLM_DATA_PARALLEL_SIZE 小于1，无法启用专家并行功能"
        exit 1
    fi
else
    echo "✅ USER_VLLM_ENABLE_EXPERT_PARALLEL 未启用"
fi

export VLLM_OPTIONAL_ARGS=${_VLLM_OPTIONAL_ARGS}


