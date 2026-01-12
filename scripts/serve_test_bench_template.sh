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

# Prefill 测试
# vllm bench serve --port ${USER_VLLM_PORT} \
#     --model ${USER_VLLM_MODEL} \
#     --backend vllm \
#     --endpoint /v1/completions \
#     --dataset-name random \
#     --random-input-len 5500 \
#     --random-output-len 1 \
#     --random-range-ratio 0.0 \
#     --profile \
#     --num-prompts 1

# Batch Decode 测试
vllm bench serve --port ${USER_VLLM_PORT} \
    --model ${USER_VLLM_MODEL} \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 103 \
    --random-output-len 30 \
    --random-range-ratio 0.9 \
    --profile \
    --num-prompts 13

# Single Decode 测试
# vllm bench serve --port ${USER_VLLM_PORT} \
#     --model ${USER_VLLM_MODEL} \
#     --backend vllm \
#     --endpoint /v1/completions \
#     --dataset-name random \
#     --random-input-len 103 \
#     --random-output-len 200 \
#     --random-range-ratio 0.0 \
#     --profile \
#     --num-prompts 1