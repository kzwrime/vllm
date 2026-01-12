# Test "[Config] Isolate DP master ip and head ip configurations in headless mode."

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

# Test MP RPC

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

# Test mpirun + MP RPC



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

