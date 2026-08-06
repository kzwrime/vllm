# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Script to start a vLLM worker process that connects to a remote executor.

This script is used for distributed multiprocessing scenarios where workers
run on different nodes than the executor. The worker connects to the executor
via TCP sockets and receives its configuration.

Usage:
    python -m vllm.v1.executor.run_mp_rpc_worker \
        --executor-addr <executor_ip> \
        --rank <rank> \
        --local-rank <local_rank>

Environment variables:
    VLLM_MP_RPC_READY_BASE_PORT: Base port for ready connections (default: 28888)
        The actual port used will be VLLM_MP_RPC_READY_BASE_PORT + rank
    VLLM_USE_MP_RPC: Must be set to "1" for RPC mode
"""

import argparse
import os
import signal
import socket
import time
import traceback
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start a vLLM worker that connects to a remote executor"
    )
    parser.add_argument(
        "--executor-addr",
        "--executor-ip",
        type=str,
        required=True,
        help="IP address of the executor to connect to",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Global rank of this worker",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        required=True,
        help="Local rank of this worker",
    )
    return parser.parse_args()


def rpc_worker_main(
    *,
    executor_addr: str,
    rank: int,
    local_rank: int,
) -> None:
    """Entry-point for an externally-started RPC worker.

    Connects to the executor, receives configuration, initializes the worker,
    sends a READY signal, then runs the busy loop.
    """
    from multiprocessing import get_context as mp_get_context

    import vllm.envs as envs
    from vllm.envs import enable_envs_cache
    from vllm.logger import init_logger
    from vllm.tracing import maybe_init_worker_tracer
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.executor.socket_utils import sock_recv, sock_send
    from vllm.v1.executor.vllm_net_devices import set_worker_net_device

    logger = init_logger(__name__)

    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    base_port = int(
        os.environ.get("VLLM_MP_RPC_READY_BASE_PORT", envs.VLLM_MP_RPC_READY_BASE_PORT)
    )
    ready_port = base_port + rank

    # Retry connecting for up to 5 minutes
    max_wait = 300
    retry_interval = 2
    t0 = time.time()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    last_log = 0.0
    connected = False
    while not connected and (time.time() - t0) < max_wait:
        try:
            sock.connect((executor_addr, ready_port))
            connected = True
        except ConnectionRefusedError:
            now = time.time()
            if now - last_log >= 10:
                logger.debug(
                    "Waiting for executor at %s:%d (%.0fs elapsed)…",
                    executor_addr,
                    ready_port,
                    now - t0,
                )
                last_log = now
            time.sleep(retry_interval)

    if not connected:
        raise ConnectionError(
            f"Could not connect to executor at {executor_addr}:{ready_port} "
            f"after {max_wait}s"
        )

    worker: WorkerProc | None = None
    try:
        with sock:
            # 1. Receive config from executor
            # Receive length-prefixed pickle
            config_obj = sock_recv(sock)
            assert isinstance(config_obj, dict), "Config should be a dict"
            config_dict: dict[str, Any] = config_obj

            # Apply executor VLLM_* env overrides, except node-local network
            # identity.  Remote workers must retain the addresses selected by
            # their own launcher; copying the executor's VLLM_HOST_IP makes a
            # worker bind ZMQ sockets to an address that does not exist on its
            # host.
            skip_envs = {"VLLM_LOOPBACK_IP", "VLLM_HOST_IP"}
            for k, v in config_dict.get("vllm_envs", {}).items():
                if k in skip_envs:
                    continue
                existing = os.environ.get(k)
                if existing is not None and existing != v:
                    logger.warning("Overwriting env %s: '%s' → '%s'", k, existing, v)
                os.environ[k] = v

            vllm_config = config_dict["vllm_config"]
            input_shm_handle = config_dict["input_shm_handle"]
            distributed_init_method = config_dict["distributed_init_method"]
            cfg_local_rank = config_dict["local_rank"]
            cfg_rank = config_dict["rank"]
            is_driver_worker = config_dict["is_driver_worker"]

            assigned_physical_gpu_ids = (
                vllm_config.parallel_config.assigned_physical_gpu_ids
            )
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)

            set_worker_net_device(cfg_local_rank, vllm_config)
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                process_kind="worker",
                process_name=f"Worker_{cfg_rank}",
            )

            # Each worker creates its own lock
            worker_lock = mp_get_context("spawn").Lock()

            worker = WorkerProc(
                vllm_config=vllm_config,
                local_rank=cfg_local_rank,
                rank=cfg_rank,
                distributed_init_method=distributed_init_method,
                input_shm_handle=input_shm_handle,
                shared_worker_lock=worker_lock,
                is_driver_worker=is_driver_worker,
            )

            # 2. Send READY + response MQ handle back to executor
            assert worker.worker_response_mq is not None
            ready_payload = {
                "status": WorkerProc.READY_STR,
                "handle": worker.worker_response_mq.export_handle(),
                "peer_response_handles": worker.peer_response_handles,
            }
            sock_send(sock, ready_payload)

        # Socket closed. Wait for MQs to be ready.
        if worker.rpc_broadcast_mq is not None:
            worker.rpc_broadcast_mq.wait_until_ready()
        worker.worker_response_mq.wait_until_ready()

        enable_envs_cache()
        worker.worker_busy_loop()

    except SystemExit:
        logger.info("RPC worker rank %d received termination signal", rank)
        raise
    except Exception:
        logger.exception("RPC worker rank %d failed", rank)
        if sock.fileno() != -1:
            import contextlib

            with contextlib.suppress(Exception):
                sock_send(
                    sock,
                    {
                        "status": "FAILURE",
                        "error": traceback.format_exc(),
                    },
                )
    finally:
        if worker is not None:
            worker.shutdown()


def main():
    args = parse_args()

    # import mpi before import vllm
    if bool(int(os.getenv("VLLM_CPU_USE_MPI", "0"))):
        from mpi4py import MPI

        print(
            f"[rank={MPI.COMM_WORLD.Get_rank()}][{MPI.Get_processor_name()}] "
            f"MPI.Is_initialized()={MPI.Is_initialized()}",
            flush=True,
        )
        MPI.COMM_WORLD.Barrier()

    # Set RPC mode for code paths that inspect the worker environment.
    os.environ["VLLM_USE_MP_RPC"] = "1"

    # Call the worker main
    rpc_worker_main(
        executor_addr=args.executor_addr,
        rank=args.rank,
        local_rank=args.local_rank,
    )


if __name__ == "__main__":
    main()
