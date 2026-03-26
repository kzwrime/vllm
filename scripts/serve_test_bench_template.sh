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
load_user_config "$SCRIPT_DIR"

# Batch Decode 测试
vllm bench serve --port ${USER_VLLM_PORT} \
    --model ${USER_VLLM_MODEL} \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 30 \
    --random-output-len 30 \
    --random-range-ratio 0.0 \
    --profile \
    --num-prompts 4

vllm bench serve --port ${USER_VLLM_PORT} \
    --model ${USER_VLLM_MODEL} \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 30 \
    --random-output-len 30 \
    --random-range-ratio 0.0 \
    --profile \
    --num-prompts 4
