#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <InnerRankBase> <InnerSize> <ExecutorIP>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../"

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

# Get values from parameters
InnerRankBase=$1
InnerSize=$2
ExecutorIP=$3


check_and_print_env "VLLM_LOOPBACK_IP"
check_and_print_env "InnerRankBase"
check_and_print_env "InnerSize"
check_and_print_env "ExecutorIP"

# Start workers using for loop
for ((i=0; i<InnerSize; i++)); do
    INNER_RANK=$((InnerRankBase + i))
    INNER_LOCAL_RANK=$i
    
    echo "Starting worker: rank=$INNER_RANK, local_rank=$INNER_LOCAL_RANK"
    
    python3 "$SCRIPT_DIR/../vllm/v1/executor/run_mp_rpc_worker.py" \
        --rank $INNER_RANK \
        --local-rank $INNER_LOCAL_RANK \
        --executor-ip "$ExecutorIP" | tee "logs/vllm_worker_log_rank${INNER_RANK}.txt" &
done

echo "Started $InnerSize worker processes"
echo "Waiting for all processes to complete..."

# Wait for all background processes to complete
wait

echo "All worker processes have completed"