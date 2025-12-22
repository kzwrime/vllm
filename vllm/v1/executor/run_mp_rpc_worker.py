# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os
import socket

host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print(f"{host_name}@{host_ip}", flush=True)

# import mpi before import vllm
if bool(os.getenv("VLLM_CPU_USE_MPI", "0")):
    from mpi4py import MPI
    print(
        f"[rank={MPI.COMM_WORLD.Get_rank()}][{MPI.Get_processor_name()}] "
        f"MPI.Is_initialized()={MPI.Is_initialized()}",
        flush=True)

import vllm.envs as envs  # noqa: E402
from vllm.utils import get_loopback_ip  # noqa: E402
# Assume multiproc_executor is in the same path or PYTHONPATH
from vllm.v1.executor.multiproc_rpc_executor import WorkerProc  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="vLLM Remote Worker")
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Rank of this worker in the executor [0, TP_SIZE)")
    parser.add_argument("--local-rank",
                        type=int,
                        required=True,
                        help="Local rank of this worker in the executor")
    parser.add_argument("--executor-ip",
                        type=str,
                        default=get_loopback_ip(),
                        help="IP address of the main executor")
    args = parser.parse_args()

    # All necessary configs will be received from the executor over the network
    kwargs = {
        "local_rank": args.local_rank,
        "rank": args.rank,
        "executor_addr": args.executor_ip,
        "ready_port": envs.VLLM_MP_RPC_READY_BASE_PORT + args.rank
    }

    # Call the modified worker entrypoint
    WorkerProc.worker_main(**kwargs)


if __name__ == "__main__":
    main()
