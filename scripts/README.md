# New Version

Setup your own env.sh first (ref env_template.sh).

## Auto test qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager

```bash
./scripts/run_vllm_test.sh qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager
```

### 日志位置说明

测试运行后，所有日志文件都保存在 `./logs/` 目录下：

| 日志文件 | 说明 |
|---------|------|
| `logs/test_YYYYMMDD_HHMMSS.log` | 测试结果日志（包含完整的 API 响应） |
| `logs/vllm_head_log.txt` | Head Server 运行日志 |
| `logs/mpi_workers_YYYYMMDD_HHMMSS.log` | MPI Workers 启动日志 |
| `logs/mpi_cleanup_YYYYMMDD_HHMMSS.log` | MPI 进程清理日志（含 MPI 清理消息） |
| `logs/vllm_worker_log_rank*.txt` | 各个 Worker 的运行日志（本轮运行） |
| `logs/vllm_serve_log_dp_rank*.txt` | DP rank 相关日志（本轮运行） |

**注意**：每轮测试运行前，旧的 `vllm_*.txt` 日志会被自动备份为 `*.txt.old`，确保只显示当前运行的日志。

### 判断测试是否成功

**成功标志：**
1. 脚本输出显示 `[SUCCESS] 服务启动成功！`
2. 脚本输出显示 `[SUCCESS] 测试完成！`
3. 脚本末尾显示绿色的 **模型回答 (Content)** 部分，包含模型返回的内容
4. 测试退出码为 0

**失败标志：**
1. 输出包含 `[ERROR]` 信息
2. 显示 `等待超时` 或 `检测到错误`
3. 模型回答部分显示 `未能提取到 content 字段`

**查看完整日志：**
```bash
# 查看最新测试结果
cat logs/test_*.log | tail -50

# 查看 Head Server 日志
cat logs/vllm_head_log.txt

# 查看 MPI Workers 日志
cat logs/mpi_workers_*.log

# 查看 MPI 清理日志（如需调试 MPI 进程终止问题）
cat logs/mpi_cleanup_*.log

# 查看特定 Worker 日志
cat logs/vllm_worker_log_rank0.txt

# 查看旧的备份日志
cat logs/vllm_*.txt.old | less
```

## Manual test qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager

Terminal session 1: start vllm head server

```bash
VLLM_CURRENT_PRESET=qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager bash ./scripts/serve_head_only_template.sh
```

Terminal session 2: start vllm headless engine cores and workers

```bash
VLLM_CURRENT_PRESET=qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager mpirun -np 4 bash ./scripts/serve_mp_rpc/serve_mp_rpc_all_mpi_template.sh
```

Terminal session 3: use curl to test service

```bash
VLLM_CURRENT_PRESET=qwen30b_a3b_dp2_tp2_ep_head_and_headless_mpi_eager bash ./scripts/serve_test_template.sh
```

# Manual start (old version)

## Test "[Config] Isolate DP master ip and head ip configurations in 

Copy and edit if needed.

```bash
cp scripts/env_template.sh scripts/env.sh
cp scripts/user_env_template.sh scripts/user_env.sh
```

Node 0: (API Server), IP: 172.33.0.10

```bash
bash ./scripts/serve_head_only_template.sh
```

Node 1: (DP-Rank0), IP: 172.33.0.11

```bash
RANK=0 bash ./scripts/serve_headless_dp_only_template.sh
```

Node 2: (DP-Rank1), IP: 172.33.0.12

```bash
RANK=1 bash ./scripts/serve_headless_dp_only_template.sh
```

## Test MP RPC

E.g. Headless API Server + DP2 + TP2

Copy and edit if needed.

```bash
cp scripts/env_template.sh scripts/env.sh
cp scripts/user_env_template.sh scripts/user_env.sh
```

Modify user_env.sh:

```bash
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_DATA_PARALLEL_ADDRESS="172.33.0.11"  # DP-Rank0 Executor IP
export USER_VLLM_DATA_PARALLEL_RPC_IP="172.33.0.10"   # HEAD IP (API Server in headless mode)
export VLLM_DP_MASTER_WORKER_IP="172.33.0.13"         # DP-Rank0 Worker0 IP
```

`VLLM_DP_MASTER_WORKER_IP` is needed for MoE models if DP-Rank0's Executor and Worker0 are not in the same node.

Node 0: (API Server), IP: 172.33.0.10

```bash
bash ./scripts/serve_head_only_template.sh
```

Node 1: (DP-Rank0 Executor), IP: 172.33.0.11

```bash
DP_RANK=0 bash ./scripts/serve_mp_rpc/serve_mp_rpc_executor_template.sh 
```

Node 2: (DP-Rank1 Executor), IP: 172.33.0.12

```bash
DP_RANK=1 bash ./scripts/serve_mp_rpc/serve_mp_rpc_executor_template.sh
```

Node 3: (DP-Rank0 2 Workers (TP-Rank0, TP-Rank1)), IP: 172.33.0.13

```bash
bash ./scripts/serve_mp_rpc/serve_mp_rpc_workers_template.sh 0 2 172.33.0.11
```

Node 4: (DP-Rank1 1 Workers (TP-Rank0)), IP: 172.33.0.14

```bash
bash ./scripts/serve_mp_rpc/serve_mp_rpc_workers_template.sh 0 1 172.33.0.12
```

Node 5: (DP-Rank1 1 Workers (TP-Rank1)), IP: 172.33.0.15

```bash
bash ./scripts/serve_mp_rpc/serve_mp_rpc_workers_template.sh 1 1 172.33.0.12
```

## Test mpirun + MP RPC

mpirun is just a launcher

- `export VLLM_CPU_USE_MPI=0` to use gloo
- `export VLLM_CPU_USE_MPI=1` to use mpi, need to cooperate with 

E.g. Configuration for Headless API Server + DP2 + TP2

Copy and edit if needed.

```bash
cp scripts/env_template.sh scripts/env.sh
cp scripts/user_env_template.sh scripts/user_env.sh
```

Modify `user_env.sh`:

```bash
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_DATA_PARALLEL_ADDRESS="172.33.0.11"  # DP-Rank0 Executor IP
export USER_VLLM_DATA_PARALLEL_RPC_IP="172.33.0.10"   # HEAD IP (API Server in headless mode)
export VLLM_DP_MASTER_WORKER_IP="172.33.0.11"         # DP-Rank0 Worker0 IP
```

Modify `scripts/serve_mp_rpc/serve_mp_rpc_all_mpi_template.sh`

```bash
TMP_IP_END=$((11 + DP_RANK * USER_VLLM_MPC_SIZE))   # uncomment this line
export ExecutorIP="172.33.0.${TMP_IP_END}"  # uncomment this line
check_and_print_env "ExecutorIP"
```

Node 0: (API Server), IP: 172.33.0.10

```bash
bash ./scripts/serve_head_only_template.sh
```

```bash
mpirun -np 4 --hostfile hostfile bash ./scripts/serve_mp_rpc/serve_mp_rpc_all_mpi_template.sh
```

mpi-workers' ip = 172.33.0.11 ... 172.33.0.14

Suppose you can `ssh mpi-worker1` ... `ssh mpi-worker4`, and their environment is the same.

mpich version hostfile,

```
mpi-worker1
mpi-worker2
mpi-worker3
mpi-worker4
```

openmpi version hostfile,

```
mpi-worker1 slots=1
mpi-worker2 slots=1
mpi-worker3 slots=1
mpi-worker4 slots=1
```

