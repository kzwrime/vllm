#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"
if [ -f "$ENV_FILE" ]; then
    echo "loading env file: $ENV_FILE"
    source "$ENV_FILE"
else
    echo "ERROR ! Could not find $ENV_FILE"
    exit 1
fi

vllm serve "facebook/opt-125m" \
  --data-parallel-size 2 \
  --data-parallel-size-local 0 \
  --data-parallel-address 0.0.0.0 \
  --data-parallel-rpc-port 13345

# vllm serve "/data/Qwen1.5-MoE-A2.7B-Chat" \
#   --max-model-len 2048 \
#   --enable-expert-parallel \
#   --data-parallel-size 2 \
#   --data-parallel-size-local 0 \
#   --data-parallel-address 0.0.0.0 \
#   --data-parallel-rpc-port 13345

