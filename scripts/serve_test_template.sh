#!/bin/bash

# 查看可用模型
# curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "facebook/opt-125m",
"prompt": "San Francisco is a",
"max_tokens": 40,
"temperature": 0
}'

# curl http://localhost:8000/v1/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "/data/Qwen1.5-MoE-A2.7B-Chat",
# "prompt": "San Francisco is a",
# "max_tokens": 40,
# "temperature": 0
# }'

# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "/data/Qwen1.5-MoE-A2.7B-Chat",
#     "max_tokens": 40,
#     "temperature": 0,
#     "messages": [
#       {"role": "user", "content": "Hello, introduce yourself."}
#     ]
#   }'