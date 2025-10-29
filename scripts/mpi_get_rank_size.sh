#!/bin/bash

# 定义全局变量来存储 rank 和 size
RANK=""
SIZE=""

# --- 环境变量检测逻辑 ---
# 使用 if-elif-else 结构来按优先级顺序检查不同的环境变量
# 找到第一个匹配项后，后续的 elif 不会再被检查。
if [[ -n "$OMPI_COMM_WORLD_RANK" && -n "$OMPI_COMM_WORLD_SIZE" ]]; then
    # 1. 尝试 Open MPI
    RANK="$OMPI_COMM_WORLD_RANK"
    SIZE="$OMPI_COMM_WORLD_SIZE"
    echo "使用 Open MPI 环境变量检测到:"

elif [[ -n "$PMIX_RANK" && -n "$PMIX_SIZE" ]]; then
    # 2. 尝试 PMIX: MPICH / Intel MPI / PMIX
    # 注意：PMIX 也经常使用 PMI 环境变量
    RANK="$PMIX_RANK"
    SIZE="$PMIX_SIZE"
    echo "使用 PMIX 环境变量检测到:"

elif [[ -n "$PMI_RANK" && -n "$PMI_SIZE" ]]; then
    # 2. 尝试 PMI: MPICH / Intel MPI
    RANK="$PMI_RANK"
    SIZE="$PMI_SIZE"
    echo "使用 PMI 环境变量检测到:"

elif [[ -n "$CRAY_MPICH_RANK" && -n "$CRAY_MPICH_SIZE" ]]; then
    # 3. 尝试 Cray MPICH
    RANK="$CRAY_MPICH_RANK"
    SIZE="$CRAY_MPICH_SIZE"
    echo "使用 Cray MPICH 环境变量检测到:"

elif [[ -n "$SLURM_PROCID" && -n "$SLURM_NPROCS" ]]; then
    # 4. 尝试 Slurm 调度系统环境变量
    # 在 Slurm 环境中，这些变量有时可以直接使用
    RANK="$SLURM_PROCID"
    SIZE="$SLURM_NPROCS"
    echo "使用 Slurm 环境变量检测到:"

else
    # 5. 如果所有尝试都失败
    RANK="unknown"
    SIZE="unknown"
    echo "未能检测到已知的 MPI 环境变量。请确认该脚本是否在 mpirun 的环境中运行。"
fi

# --- 输出结果 ---
echo "  MPI Get Rank: $RANK, Size: $SIZE"

export MPI_RANK_DETECT=$RANK
export MPI_SIZE_DETECT=$SIZE
