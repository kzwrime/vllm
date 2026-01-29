
export USER_VLLM_MODEL="facebook/opt-125m"
export USER_VLLM_MAX_MODEL_LEN=2048
export USER_VLLM_MAX_NUM_BATCHED_TOKENS=2048
export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_DATA_PARALLEL_ADDRESS="172.33.0.11"  # DP-Rank0 Executor IP
export USER_VLLM_DATA_PARALLEL_RPC_IP="172.33.0.10"   # HEAD IP (API Server in headless mode)
export VLLM_DP_MASTER_WORKER_IP="172.33.0.11"         # DP-Rank0 Worker0 IP
export USER_VLLM_DATA_PARALLEL_RPC_PORT=13345
export USER_VLLM_PORT=14800
export VLLM_CPU_KVCACHE_SPACE=4 # KV Cache Size

# IP addresses vary across different TP groups.
export ExecutorIP=172.33.0.11 
export USER_VLLM_MPC_SIZE=2
export USER_VLLM_MP_RPC_WORKER_PER_NODE=1

export VLLM_USE_CPU_SHM_DIST=0
export VLLM_LOOPBACK_IP=$(hostname -I | awk '{print $1}')
# export VLLM_LOOPBACK_IP=$(ifconfig eth0 | grep "inet " | awk '{print ^C}')

export VLLM_USE_XCPU_LINEAR=0

export VLLM_CPU_USE_MPI=0
export TORCHINDUCTOR_CPP_WRAPPER=1
export VLLM_DISABLE_TQDM_AND_MONITOR=1
export VLLM_SHARED_EXPERT_DISABLE_TP=1
export VLLM_ALL2ALL_BACKEND_XCPU="torch_all_to_all_single" # Fallback solution with universal compatibility
# export VLLM_ALL2ALL_BACKEND_XCPU="mpi_alltoallv" # Requires: VLLM_CPU_USE_MPI=1

_VLLM_OPTIONAL_ARGS=" "
# _VLLM_OPTIONAL_ARGS+=" --enforce-eager"
_VLLM_OPTIONAL_ARGS+=" --max-num-seqs 16"
_VLLM_OPTIONAL_ARGS+=' --profiler-config {"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_record_shapes":true,"torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"torch_profiler_with_flops":true,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":true}'
# _VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"

export VLLM_OPTIONAL_ARGS=${_VLLM_OPTIONAL_ARGS}

echo "VLLM_OPTIONAL_ARGS: ${VLLM_OPTIONAL_ARGS}"
