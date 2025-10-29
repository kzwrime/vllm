#!/bin/bash

# USAGE: mpirun -n 2 bash scripts/serve_headless_mpi_template.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"
if [ -f "$ENV_FILE" ]; then
    echo "loading env file: $ENV_FILE"
    source "$ENV_FILE"
else
    echo "ERROR ! Could not find $ENV_FILE"
    exit 1
fi

source "$SCRIPT_DIR/mpi_get_rank_size.sh"

RANK="$MPI_RANK_DETECT"
SIZE="$MPI_SIZE_DETECT"

# export LOCAL_RANK=${RANK}

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

vllm serve "facebook/opt-125m" \
  --headless \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-size-local 1  \
  --data-parallel-start-rank $RANK \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 13345 2>&1 | tee vllm_serve_log_rank${RANK}.txt

# vllm serve "/data/Qwen1.5-MoE-A2.7B-Chat" \
#   --headless \
#   --enforce-eager \
#   --max-model-len 2048 \
#   --tensor-parallel-size 2 \
#   --enable-expert-parallel \
#   --data-parallel-size 2 \
#   --data-parallel-size-local 1  \
#   --data-parallel-start-rank $RANK \
#   --data-parallel-address 127.0.0.1 \
#   --data-parallel-rpc-port 13345 2>&1 | tee vllm_serve_log_rank${RANK}.txt

  # --distributed-executor-backend external_launcher \
