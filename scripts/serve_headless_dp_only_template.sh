#!/bin/bash

# USAGE: RANK=x bash scripts/serve_headless_dp_only_template.sh

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
check_and_print_env "RANK"
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

VLLM_LOGGING_LEVEL=${USER_VLLM_LOGGING_LEVEL} vllm serve ${USER_VLLM_MODEL} \
  --headless \
  --max-model-len ${USER_VLLM_MAX_MODEL_LEN} \
  --max-num-batched-tokens ${USER_VLLM_MAX_NUM_BATCHED_TOKENS} \
  -tp=${USER_VLLM_TP_SIZE} \
  -pp=${USER_VLLM_PP_SIZE} \
  --distributed-executor-backend mp \
  --port ${USER_VLLM_PORT} \
  --enforce-eager \
  --data-parallel-size ${USER_VLLM_DATA_PARALLEL_SIZE} \
  --data-parallel-size-local 1  \
  --data-parallel-start-rank $RANK \
  --data-parallel-address ${USER_VLLM_DATA_PARALLEL_ADDRESS} \
  --data-parallel-rpc-ip ${USER_VLLM_DATA_PARALLEL_RPC_IP} \
  --data-parallel-rpc-port ${USER_VLLM_DATA_PARALLEL_RPC_PORT} 2>&1 | tee logs/vllm_serve_log_rank${RANK}.txt

