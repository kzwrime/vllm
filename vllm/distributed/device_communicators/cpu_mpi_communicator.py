# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

try:
    import mpi4py.rc
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
except ImportError:
    raise ImportError("mpi4py not found.") from None

logger = init_logger(__name__)

# 定义 MPI 后端原生支持的 dtype
mpi_supported_dtypes = {
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float32,
    torch.float64
}


def convert_to_supported_dtype(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in mpi_supported_dtypes:
        return tensor
    elif tensor.dtype.is_floating_point:
        return tensor.to(dtype=torch.float32)
    elif tensor.dtype == torch.bool or 'int' in str(tensor.dtype):
        return tensor.to(dtype=torch.int32)
    else:
        raise NotImplementedError(
            f"Unsupported dtype {tensor.dtype} encountered. "
            f"Casting to {torch.float32} for all_reduce.")


def all_reduce_auto_convert(tensor: torch.Tensor,
                            group: dist.ProcessGroup) -> torch.Tensor:
    """
    一个通用的 all_reduce 函数，可以自动处理 MPI 后端不支持的数据类型。

    它会检查输入张量的数据类型，如果不是原生支持的类型
    （如 int32, int64, float32, float64）, 
    则会临时将其转换为一个支持的类型 (通常是 float32 或 int32), 
    执行 all_reduce 操作后，再转换回原始类型。

    Args:
        tensor (torch.Tensor): 需要进行 all_reduce 操作的输入张量。
        group (dist.ProcessGroup): 执行操作的进程组。

    Returns:
        torch.Tensor: all_reduce 操作后的张量，其数据类型与输入张量相同。
    """
    original_dtype = tensor.dtype

    if original_dtype in mpi_supported_dtypes:
        # 如果类型受支持，直接执行 all_reduce
        dist.all_reduce(tensor, group=group)
        return tensor
    else:
        temp_tensor = convert_to_supported_dtype(tensor)
        dist.all_reduce(temp_tensor, group=group)
        # 使用 .to() 将结果转换回原始类型并返回
        # 注意：这会创建一个新的张量
        return temp_tensor.to(dtype=original_dtype)


def custom_all_gather_into_tensor(output_tensor: torch.Tensor,
                                  input_tensor: torch.Tensor,
                                  group: dist.ProcessGroup = None):
    """
    使用 torch.distributed.all_gather 重写 all_gather_into_tensor 功能。

    Args:
        output_tensor (Tensor): 目标张量，用于存放所有进程收集到的数据。
          其大小应为 (world_size * N, ...)，
          其中 (N, ...) 是 input_tensor 的形状。
        input_tensor (Tensor): 当前进程需要被收集的张量。
        group (ProcessGroup, optional): 要操作的进程组。
    """
    # 1. 获取分布式环境的世界大小 (world_size)
    world_size = dist.get_world_size(group=group)

    # 2. 验证 output_tensor 的形状是否正确
    # 它的第一个维度应该是 input_tensor 第一个维度的 world_size 倍
    expected_shape = list(input_tensor.shape)
    if expected_shape[0] * world_size != output_tensor.shape[0]:
        raise ValueError(f"output_tensor 的形状不正确。期望第一个维度为 "
                         f"{expected_shape[0] * world_size}, "
                         f"但实际为 {output_tensor.shape[0]}。")

    # 3. 将 output_tensor 切分成一个张量列表
    # 列表中的每个张量都是 output_tensor 的一个视图 (view)，指向其内存的不同部分
    # 这样做可以避免额外的数据分配和拷贝
    tensor_list = list(torch.chunk(output_tensor, chunks=world_size, dim=0))

    # 4. 调用 all_gather
    # all_gather 会将每个进程的 input_tensor 填充到 tensor_list 对应的位置
    # 因为 tensor_list 中的元素是 output_tensor 的视图，
    # 所以 output_tensor 会被直接就地修改
    dist.all_gather(tensor_list, input_tensor, group=group)


class CpuMPICommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)

        logger.info("CpuMPICommunicator initializing ...")

        assert MPI.Is_initialized()

        num_ranks = cpu_group.size()
        assert num_ranks > 0
        logger.info("num_ranks: %d", num_ranks)

        global_rank_tensor = torch.tensor([self.global_rank],
                                          dtype=torch.int32)
        group_ranks = torch.zeros(num_ranks, dtype=torch.int32)
        dist.all_gather_into_tensor(group_ranks,
                                    global_rank_tensor,
                                    group=self.cpu_group)
        group_ranks = group_ranks.tolist()
        logger.info("group_ranks: %s", str(group_ranks))

        mpi_group = MPI.COMM_WORLD.group.Incl(group_ranks)
        self.mpi_group_comm = MPI.Intracomm.Create_from_group(mpi_group)
        self.mpi_group_rank = self.mpi_group_comm.Get_rank()
        self.mpi_group_size = self.mpi_group_comm.Get_size()
        logger.info("CpuMPICommunicator initialized, rank: %d, world_size: %d",
                    self.mpi_group_rank, self.mpi_group_size)

        assert self.mpi_group_rank == self.rank
        assert self.mpi_group_size == self.world_size

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # logger.info(f"all_reduce rank: {self.mpi_group_rank}, "
        #     f"input_.shape: {input_.shape}, input_.dtype: {input_.dtype}")
        tin = convert_to_supported_dtype(input_)
        self.mpi_group_comm.Allreduce(MPI.IN_PLACE, tin)
        return tin.to(dtype=input_.dtype)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # logger.info(f"all_gather rank: {self.mpi_group_rank}, "
        #     f"input_.shape: {input_.shape}, input_.dtype: {input_.dtype}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size, ) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(output_size,
                                    dtype=input_.dtype,
                                    device=input_.device)

        tin = convert_to_supported_dtype(input_)
        tout = convert_to_supported_dtype(output_tensor)
        self.mpi_group_comm.Allgather(tin, tout)
        output_tensor = tout.to(dtype=input_.dtype)

        # Reshape
        output_tensor = output_tensor.reshape((self.world_size, ) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(input_size[:dim] +
                                              (self.world_size *
                                               input_size[dim], ) +
                                              input_size[dim + 1:])
        return output_tensor

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, list[torch.Tensor]],
        dim: int = 0,
        sizes: Optional[list[int]] = None
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError

    def reduce_scatter(self,
                       input_: torch.Tensor,
                       dim: int = -1) -> torch.Tensor:
        raise NotImplementedError

    def reduce_scatterv(self,
                        input_: torch.Tensor,
                        dim: int = -1,
                        sizes: Optional[list[int]] = None) -> torch.Tensor:
        raise NotImplementedError

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        raise NotImplementedError

    pass
