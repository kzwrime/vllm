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
check_and_print_env "USER_VLLM_MODEL"
check_and_print_env "USER_VLLM_LOGGING_LEVEL"
check_and_print_env "USER_VLLM_MAX_MODEL_LEN"
check_and_print_env "USER_VLLM_MAX_NUM_BATCHED_TOKENS"
check_and_print_env "USER_VLLM_PORT"
check_and_print_env "USER_VLLM_TP_SIZE"
check_and_print_env "VLLM_OPTIONAL_ARGS"

echo ""
echo "--- 🚀 正在启动 vLLM 服务... ---"

VLLM_LOGGING_LEVEL=${USER_VLLM_LOGGING_LEVEL} vllm serve ${USER_VLLM_MODEL} \
  --max-model-len ${USER_VLLM_MAX_MODEL_LEN} \
  --max-num-batched-tokens ${USER_VLLM_MAX_NUM_BATCHED_TOKENS} \
  --port ${USER_VLLM_PORT} \
  -tp=${USER_VLLM_TP_SIZE} \
  ${VLLM_OPTIONAL_ARGS} | tee logs/vllm_serve_log.txt

