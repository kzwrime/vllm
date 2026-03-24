
export USER_VLLM_MODEL="facebook/opt-125m"
export USER_VLLM_MAX_MODEL_LEN=2048

export USER_VLLM_DATA_PARALLEL_SIZE=2
export USER_VLLM_TP_SIZE=2
export USER_VLLM_PP_SIZE=1
export USER_VLLM_DATA_PARALLEL_ADDRESS="127.0.0.1"  # DP-Rank0 Executor IP
export USER_VLLM_DATA_PARALLEL_RPC_IP="127.0.0.1"   # HEAD IP (API Server in headless mode)
export VLLM_DP_MASTER_WORKER_IP="127.0.0.1"         # DP-Rank0 Worker0 IP
export USER_VLLM_DATA_PARALLEL_RPC_PORT=13345
export USER_VLLM_PORT=14800
export VLLM_CPU_KVCACHE_SPACE=4 # KV Cache Size
# export USER_VLLM_EAGER_OR_NOT="--enforce-eager"

# 设置 VLLM_USE_MPI_COORD=1 时，会通过额外的 python 脚本，自动协调并设置以下变量
# - USER_VLLM_DATA_PARALLEL_ADDRESS
# - VLLM_DP_MASTER_WORKER_IP
# - ExecutorIP
export VLLM_USE_MPI_COORD=0

# IP addresses vary across different TP groups.
export ExecutorIP=127.0.0.1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
export USER_VLLM_MP_RPC_WORKER_PER_NODE=1

export VLLM_USE_CPU_SHM_DIST=0
export VLLM_LOOPBACK_IP=$(hostname -I | awk '{print $1}')
# export VLLM_LOOPBACK_IP=$(ifconfig eth0 | grep "inet " | awk '{print $2}')

export VLLM_USE_XCPU_LINEAR=0
# export TORCH_XCPU_ENABLE_CHECK=0
# export VLLM_CPU_MOCK_LINEAR=1
export VLLM_CPU_USE_MPI=0
export TORCHINDUCTOR_CPP_WRAPPER=1
export VLLM_DISABLE_TQDM_AND_MONITOR=1
export VLLM_SHARED_EXPERT_DISABLE_TP=1
export VLLM_USE_XCPU_TOPK_SOFTMAX=1

# 开启 torch_all_to_all_single / mpi_alltoallv 时，必须关闭，因此设置为总是关闭即可
export VLLM_ENABLE_SEQUENCE_PARALLEL_MOE=0

export VLLM_ALL2ALL_BACKEND_XCPU="torch_all_to_all_single" # Fallback solution with universal compatibility
# export VLLM_ALL2ALL_BACKEND_XCPU="mpi_alltoallv" # Requires: VLLM_CPU_USE_MPI=1
export VLLM_MPI_ALLTOALLV_VERSION="v1"  # v2 采用新的 alltoallv_put

# ========================================
# PD_MODE 配置
# ========================================
# PD_MODE 可选模式:
#   "PREFILL" : 对应场景 1 (分离架构中的 P 集群)，侧重 TTFT，提高计算密度
#   "DECODE"  : 对应场景 2 (分离架构中的 D 集群)，侧重 TPOT
#   "MIXED"   : 对应场景 3 (混合部署)，侧重 TPOT
#   "NOT_MOE" : 针对 Dense 模型，不做任何额外设置
export PD_MODE="${PD_MODE:-MIXED}"

_VLLM_OPTIONAL_ARGS=" "
_VLLM_OPTIONAL_ARGS+=" --max-num-seqs 16"
_VLLM_OPTIONAL_ARGS+=' --profiler-config {"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_record_shapes":true,"torch_profiler_with_memory":true,"torch_profiler_with_stack":true,"torch_profiler_with_flops":true,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":true}'
# _VLLM_OPTIONAL_ARGS+=" --no-enable-prefix-caching"
# _VLLM_OPTIONAL_ARGS+=" --load-format dummy"
# _VLLM_OPTIONAL_ARGS+=" --attention-backend TRITON_ATTN"

# 根据 PD_MODE 自动设置配置
case ${PD_MODE} in
    "PREFILL")
        ### 场景 1: Prefill 优先      ###
        ### 典型配置：TP16, DP4, EP64 ###
        echo "[VLLM-XCPU] PD Mode: PREFILL - Optimizing for TTFT"

        # Prefill 模式自动启用 enforce-eager 和专家并行
        export USER_VLLM_EAGER_OR_NOT="--enforce-eager"
        _VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"

        # _VLLM_OPTIONAL_ARGS+=" --enable-eplb"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.window_size 1000"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.step_interval 3000"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.num_redundant_experts 16"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.log_balancedness true"

        # 通过 DP_SIZE * MAX_BATCHED_TOKENS * min(topk, num_local_experts) 来控制 all2allv 和 MoE 缓冲区 Token 数
        # Prefill 时 DP 较少，将 MAX_BATCHED_TOKENS 调大
        MAX_BATCHED_TOKENS=4096
        export USER_VLLM_MAX_NUM_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}
        export VLLM_MOE_DP_CHUNK_SIZE=${MAX_BATCHED_TOKENS}
        export VLLM_ENABLE_MOE_DP_CHUNK=1
        ;;

    "DECODE" | "MIXED")
        ### 场景 2: Decode/混合模式 ###
        ### 典型配置: TP2, DP32, EP64 ###
        echo "[VLLM-XCPU] PD Mode: ${PD_MODE} - Optimizing for TPOT"

        # Decode/MIXED 模式自动启用专家并行
        _VLLM_OPTIONAL_ARGS+=" --enable-expert-parallel"

        # _VLLM_OPTIONAL_ARGS+=" --enable-eplb"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.window_size 1000"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.step_interval 3000"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.num_redundant_experts 16"
        # _VLLM_OPTIONAL_ARGS+=" --eplb-config.log_balancedness true"

        # 通过 DP_SIZE * MAX_BATCHED_TOKENS * min(topk, num_local_experts) 来控制 all2allv 和 MoE 缓冲区 Token 数
        # Decode 时 DP 较多，将 MAX_BATCHED_TOKENS 调小
        MAX_BATCHED_TOKENS=256
        export USER_VLLM_MAX_NUM_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}
        export VLLM_MOE_DP_CHUNK_SIZE=${MAX_BATCHED_TOKENS}
        export VLLM_ENABLE_MOE_DP_CHUNK=0 # torch.compile 目前不兼容 MOE_DP_CHUNK
        
        # 提示：调试时可开启 --enforce-eager，生产环境建议注释掉以利用 torch.compile 优化
        # _VLLM_OPTIONAL_ARGS+=" --enforce-eager"
        ;;

    "NOT_MOE")
        ### Dense 模型，不做任何额外设置 ###
        ;;

    *)
        echo "Error: Invalid PD_MODE '${PD_MODE}'. Must be PREFILL, DECODE, MIXED, or NOT_MOE."
        exit 1
        ;;
esac

export VLLM_OPTIONAL_ARGS=${_VLLM_OPTIONAL_ARGS}

# ========================================
# 打印配置摘要
# ========================================
echo "========================================="
echo "  VLLM Configuration Summary"
echo "========================================="
echo "  PD_MODE: ${PD_MODE}"
echo "  USER_VLLM_EAGER_OR_NOT: '${USER_VLLM_EAGER_OR_NOT}'"
echo "  VLLM_OPTIONAL_ARGS: ${VLLM_OPTIONAL_ARGS}"
echo "========================================="

# TORCHINDUCTOR_CACHE_DIR 必须是全局路径
export TORCHINDUCTOR_CACHE_DIR="$PWD/torch_compile_cache"
