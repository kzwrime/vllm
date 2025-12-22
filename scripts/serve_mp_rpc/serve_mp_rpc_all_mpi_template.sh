#!/bin/bash

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
load_env_file "$SCRIPT_DIR/mpi_get_rank_size.sh"

RANK="$MPI_RANK_DETECT"
SIZE="$MPI_SIZE_DETECT"

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

# TODO 检查 PP_SIZE * TP_SIZE = USER_VLLM_MPC_SIZE

if (( USER_VLLM_PP_SIZE * USER_VLLM_TP_SIZE != USER_VLLM_MPC_SIZE )); then
    echo "USER_VLLM_PP_SIZE($USER_VLLM_PP_SIZE) * USER_VLLM_TP_SIZE($USER_VLLM_TP_SIZE) != USER_VLLM_MPC_SIZE($USER_VLLM_MPC_SIZE)"
    exit 1
fi

DP_RANK=$((RANK / USER_VLLM_MPC_SIZE))
MPC_RANK=$((RANK % USER_VLLM_MPC_SIZE))
MPC_INNER_RANK=$((RANK % USER_VLLM_MP_RPC_WORKER_PER_NODE))

check_and_print_env "ExecutorIP"

export VLLM_MP_RPC_READY_BASE_PORT=$((28888 + DP_RANK * USER_VLLM_MPC_SIZE))

# 如果 RANK 是 TP*PP 组的第一个，启动 VLLM 服务
if [ $MPC_RANK -eq 0 ]; then
    echo "[RANK=$RANK][DP_RANK=$DP_RANK] Starting vLLM serve"
    (
        VLLM_LOGGING_LEVEL=${USER_VLLM_LOGGING_LEVEL} vllm serve ${USER_VLLM_MODEL} \
          --headless \
          --max-model-len ${USER_VLLM_MAX_MODEL_LEN} \
          --max-num-batched-tokens ${USER_VLLM_MAX_NUM_BATCHED_TOKENS} \
          --data-parallel-size ${USER_VLLM_DATA_PARALLEL_SIZE} \
          --data-parallel-size-local 1 \
          -tp=${USER_VLLM_TP_SIZE} \
          -pp=${USER_VLLM_PP_SIZE} \
          --distributed-executor-backend mp_rpc \
          ${VLLM_OPTIONAL_ARGS} \
          --data-parallel-start-rank ${DP_RANK} \
          --data-parallel-address ${USER_VLLM_DATA_PARALLEL_ADDRESS} \
          --data-parallel-rpc-ip ${USER_VLLM_DATA_PARALLEL_RPC_IP} \
          --data-parallel-rpc-port ${USER_VLLM_DATA_PARALLEL_RPC_PORT} 2>&1 | tee logs/vllm_serve_log_dp_rank${DP_RANK}.txt
    ) &
    # 保存后台进程的PID，以便后续管理
    SERVE_PID=$!
fi

# sleep 20

# 启动 MP RPC Worker
echo "[RANK=$RANK][DP_RANK=$DP_RANK][MPC_RANK=$MPC_RANK] Starting vLLM mp_rpc_worker"
python3 "$SCRIPT_DIR/../vllm/v1/executor/run_mp_rpc_worker.py" \
  --rank $MPC_RANK \
  --local-rank $MPC_INNER_RANK \
  --executor-ip ${ExecutorIP} | tee logs/vllm_worker_log_rank${RANK}.txt

wait

echo "All worker processes have completed"
