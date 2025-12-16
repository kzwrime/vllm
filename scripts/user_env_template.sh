export VLLM_LOGGING_LEVEL="DEBUG"
export VLLM_MODEL="facebook/opt-125m"
export VLLM_MAX_MODEL_LEN=4096
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VLLM_DATA_PARALLEL_SIZE=2
export VLLM_DATA_PARALLEL_ADDRESS=172.33.0.10
export VLLM_DATA_PARALLEL_RPC_PORT=13345

export USER_VLLM_ENFORCE_EAGER=1
export USER_VLLM_ENABLE_EXPERT_PARALLEL=0

_VLLM_OPTIONAL_ARGS=""

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
    _VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"
    echo "✅ 启用参数: --enable-expert-parallel"
fi

export VLLM_OPTIONAL_ARGS=${_VLLM_OPTIONAL_ARGS}
