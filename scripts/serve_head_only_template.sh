#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="$SCRIPT_DIR/common.sh"
if [ -f "$ENV_FILE" ]; then
    echo "loading env file: $ENV_FILE"
    source "$ENV_FILE"
else
    echo "ERROR ! Could not find $ENV_FILE"
    exit 1
fi

load_env_file "$SCRIPT_DIR/env.sh"
load_env_file "$SCRIPT_DIR/user_env.sh"


echo "--- 📝 vLLM 服务配置参数检查与设置 ---"

echo "--- 必需参数 ---"
check_and_print_env "USER_VLLM_MODEL"
check_and_print_env "USER_VLLM_LOGGING_LEVEL"
check_and_print_env "USER_VLLM_MAX_MODEL_LEN"
check_and_print_env "USER_VLLM_MAX_NUM_BATCHED_TOKENS"
check_and_print_env "USER_VLLM_DATA_PARALLEL_SIZE"
check_and_print_env "USER_VLLM_DATA_PARALLEL_ADDRESS"
check_and_print_env "USER_VLLM_DATA_PARALLEL_RPC_IP"
check_and_print_env "USER_VLLM_DATA_PARALLEL_RPC_PORT"
check_and_print_env "USER_VLLM_PORT"
check_and_print_env "VLLM_LOOPBACK_IP"

echo ""
echo "--- 🚀 正在启动 vLLM 服务... ---"

# --- C. 执行 vLLM 命令 ---

# 启动 vLLM 服务，使用参数化的环境变量
VLLM_LOGGING_LEVEL=${USER_VLLM_LOGGING_LEVEL} vllm serve ${USER_VLLM_MODEL} \
  --max-model-len ${USER_VLLM_MAX_MODEL_LEN} \
  --max-num-batched-tokens ${USER_VLLM_MAX_NUM_BATCHED_TOKENS} \
  -tp=${USER_VLLM_TP_SIZE} \
  -pp=${USER_VLLM_PP_SIZE} \
  --distributed-executor-backend mp \
  --port ${USER_VLLM_PORT} \
  ${VLLM_OPTIONAL_ARGS} \
  --data-parallel-size ${USER_VLLM_DATA_PARALLEL_SIZE} \
  --data-parallel-size-local 0 \
  --data-parallel-address ${USER_VLLM_DATA_PARALLEL_ADDRESS} \
  --data-parallel-rpc-ip ${USER_VLLM_DATA_PARALLEL_RPC_IP} \
  --data-parallel-rpc-port ${USER_VLLM_DATA_PARALLEL_RPC_PORT}

# 检查 vLLM 命令的退出状态
if [ $? -ne 0 ]; then
    echo "❌ 错误：vllm serve 命令执行失败。" >&2
    exit 1
fi