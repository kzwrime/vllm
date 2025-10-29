#!/bin/bash

# USAGE: mpirun -n 4 bash scripts/serve_headless_mpi_template.sh

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

# 设置张量并行大小
TP_SIZE=2

# 计算数据并行秩和张量并行内部秩
DP_RANK=$((RANK / TP_SIZE))
TP_INNER_RANK=$((RANK % TP_SIZE))
TP_INNER_LOCAL_RANK=0

# 设置VLLM MP RPC就绪端口，每个 TP 域内应该一致
export VLLM_MP_RPC_READY_BASE_PORT=$((28888 + DP_RANK * TP_SIZE))

# 如果RANK是TP组的第一个，启动VLLM服务
if [ $TP_INNER_RANK -eq 0 ]; then
    echo "[RANK=$RANK][DP_RANK=$DP_RANK] Starting vLLM serve"
    (
        vllm serve "facebook/opt-125m" \
          --headless \
          --enforce-eager \
          --data-parallel-size 2 \
          --tensor-parallel-size $TP_SIZE \
          --data-parallel-size-local 1 \
          --distributed-executor-backend mp_rpc \
          --data-parallel-start-rank $DP_RANK \
          --data-parallel-address 127.0.0.1 \
          --data-parallel-rpc-port 13345 2>&1 | tee vllm_serve_log_dp_rank${DP_RANK}.txt
    ) &
    # 保存后台进程的PID，以便后续管理
    SERVE_PID=$!
fi

sleep 20

# 启动MP RPC工作进程
echo "[RANK=$RANK][DP_RANK=$DP_RANK][TP_RANK=$TP_INNER_RANK] Starting vLLM mp_rpc_worker"
python3 "$SCRIPT_DIR/../vllm/v1/executor/run_mp_rpc_worker.py" \
  --rank $TP_INNER_RANK \
  --local-rank $TP_INNER_LOCAL_RANK \
  --executor-ip 127.0.0.1 | tee vllm_worker_log_rank${RANK}.txt

# 等待后台服务进程结束（如果有的话）
if [ $((RANK % TP_SIZE)) -eq 0 ]; then
    wait $SERVE_PID
fi

  # --distributed-executor-backend external_launcher \
