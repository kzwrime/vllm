#!/bin/bash

set -e

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
load_user_config "$SCRIPT_DIR"
load_env_file "$SCRIPT_DIR/mpi_get_rank_size.sh"

RANK="$MPI_RANK_DETECT"
SIZE="$MPI_SIZE_DETECT"

# --- MPI Coordination Setup ---
if [ "${VLLM_USE_MPI_COORD:-0}" == "1" ]; then
    COORD_PORT=${VLLM_MPI_COORD_PORT:-15555}
    COORD_SCRIPT="$SCRIPT_DIR/mpi_coord_setup.py"
    export VLLM_MPI_ENV_EXPORT_FILE="/tmp/vllm_mpi_env_rank_${RANK}.sh"

    echo "--- 🔗 MPI Coordination (Rank $RANK) ---"

    # Run client to send/receive topology info
    python3 "$COORD_SCRIPT" \
        --client \
        --port $COORD_PORT \
        --rank $RANK \
        --ip "$VLLM_LOOPBACK_IP"

    # Source the exported environment variables
    if [ -f "$VLLM_MPI_ENV_EXPORT_FILE" ]; then
        echo "Loading environment variables from $VLLM_MPI_ENV_EXPORT_FILE"
        source "$VLLM_MPI_ENV_EXPORT_FILE"
    else
        echo "ERROR: Environment export file not found!"
        exit 1
    fi
    echo ""
fi

echo "--- 📝 vLLM 服务配置参数检查与设置 ---"

echo "--- 必需参数 ---"
check_and_print_env "RANK"
check_and_print_env "SIZE"

check_and_print_env "USER_VLLM_MPC_SIZE"
check_and_print_env "USER_VLLM_MP_RPC_WORKER_PER_NODE"

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
check_and_print_env "VLLM_OPTIONAL_ARGS"

# TODO 检查 PP_SIZE * TP_SIZE = USER_VLLM_MPC_SIZE

if (( USER_VLLM_PP_SIZE * USER_VLLM_TP_SIZE != USER_VLLM_MPC_SIZE )); then
    echo "USER_VLLM_PP_SIZE($USER_VLLM_PP_SIZE) * USER_VLLM_TP_SIZE($USER_VLLM_TP_SIZE) != USER_VLLM_MPC_SIZE($USER_VLLM_MPC_SIZE)"
    exit 1
fi

# Calculate DP_RANK and MPC_RANK (keep original static calculation)
DP_RANK=$((RANK / USER_VLLM_MPC_SIZE))
MPC_RANK=$((RANK % USER_VLLM_MPC_SIZE))
MPC_INNER_RANK=$((RANK % USER_VLLM_MP_RPC_WORKER_PER_NODE))

# ExecutorIP is set by coordination script if enabled, otherwise use original calculation
# if [ "${VLLM_USE_MPI_COORD:-0}" != "1" ]; then
#     TMP_IP_END=$((11 + DP_RANK * USER_VLLM_MPC_SIZE))
#     export ExecutorIP="172.33.0.${TMP_IP_END}"
# fi
check_and_print_env "ExecutorIP"

export VLLM_MP_RPC_READY_BASE_PORT=$((28888 + DP_RANK * USER_VLLM_MPC_SIZE))

# 如果 RANK 是 TP*PP 组的第一个，启动 VLLM 服务
if [ $MPC_RANK -eq 0 ]; then
    (
        sleep 10
        echo "[RANK=$RANK][DP_RANK=$DP_RANK] Starting vLLM serve"
        export VLLM_USE_MP_RPC_WORKERS=1
        VLLM_LOGGING_LEVEL=${USER_VLLM_LOGGING_LEVEL} vllm serve ${USER_VLLM_MODEL} \
          --headless \
          --max-model-len ${USER_VLLM_MAX_MODEL_LEN} \
          --max-num-batched-tokens ${USER_VLLM_MAX_NUM_BATCHED_TOKENS} \
          -tp=${USER_VLLM_TP_SIZE} \
          -pp=${USER_VLLM_PP_SIZE} \
          --distributed-executor-backend mp \
          --data-parallel-size ${USER_VLLM_DATA_PARALLEL_SIZE} \
          --data-parallel-size-local 1 \
          ${USER_VLLM_EAGER_OR_NOT} \
          ${VLLM_OPTIONAL_ARGS} \
          --data-parallel-start-rank ${DP_RANK} \
          --data-parallel-address ${USER_VLLM_DATA_PARALLEL_ADDRESS} \
          --data-parallel-rpc-ip ${USER_VLLM_DATA_PARALLEL_RPC_IP} \
          --data-parallel-rpc-port ${USER_VLLM_DATA_PARALLEL_RPC_PORT} 2>&1 | tee logs/vllm_serve_log_dp_rank${DP_RANK}.txt
        #   --data-parallel-rpc-ip ${USER_VLLM_DATA_PARALLEL_RPC_IP} \
    ) &
    # 保存后台进程的PID，以便后续管理
    SERVE_PID=$!
fi

# 启动 MP RPC Worker
echo "[RANK=$RANK][DP_RANK=$DP_RANK][MPC_RANK=$MPC_RANK] Starting vLLM mp_rpc_worker"
(
    VLLM_LOGGING_LEVEL=${USER_VLLM_LOGGING_LEVEL} python3 "$SCRIPT_DIR/../vllm/v1/executor/run_mp_rpc_worker.py" \
      --rank $MPC_RANK \
      --local-rank $MPC_INNER_RANK \
      --executor-ip ${ExecutorIP} 2>&1 | tee logs/vllm_worker_log_rank${RANK}.txt
)

# sleep 20

wait

echo "All worker processes have completed"
