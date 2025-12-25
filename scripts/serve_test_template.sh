#!/bin/bash

set -x

# 查看可用模型
# curl http://localhost:8000/v1/models

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

curl http://localhost:${USER_VLLM_PORT}/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"${USER_VLLM_MODEL}"'",
"prompt": "San Francisco is a",
"max_tokens": 8,
"temperature": 0.1
}'