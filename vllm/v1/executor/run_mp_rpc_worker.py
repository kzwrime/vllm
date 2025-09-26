# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.utils import get_loopback_ip
# Assume multiproc_executor is in the same path or PYTHONPATH
from vllm.v1.executor.multiproc_rpc_executor import (
    REMOTE_WORKER_READY_BASE_PORT, WorkerProc)


def main():
    parser = argparse.ArgumentParser(description="vLLM Remote Worker")
    parser.add_argument("--rank",
                        type=int,
                        required=True,
                        help="Rank of this worker")
    parser.add_argument(
        "--local-rank",
        type=int,
        required=True,
        help="Local rank of this worker (usually same as rank for single node)"
    )
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
        "ready_port": REMOTE_WORKER_READY_BASE_PORT + args.rank
    }

    # Call the modified worker entrypoint
    WorkerProc.worker_main(**kwargs)


if __name__ == "__main__":
    main()
