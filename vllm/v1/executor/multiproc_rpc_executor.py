# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MultiprocRPCExecutor: like MultiprocExecutor but workers are started
independently and connect back to the executor via TCP, rather than being
spawned as subprocesses.

Usage (single node, DP=1 TP=2):
  # Terminal 1: start the executor (vllm serve) with mp_rpc enabled
  VLLM_USE_MP_RPC_WORKERS=1 VLLM_LOGGING_LEVEL=DEBUG \
      vllm serve <model> --tensor-parallel-size 2 ...

  # Terminal 2 & 3: start each worker independently
  python -m vllm.v1.executor.run_mp_rpc_worker \
      --executor-addr <executor_ip> --rank 0 --local-rank 0
  python -m vllm.v1.executor.run_mp_rpc_worker \
      --executor-addr <executor_ip> --rank 1 --local-rank 1
"""

import os
import socket
import time
from contextlib import ExitStack
from typing import Any

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.utils.network_utils import get_distributed_init_method, get_ip
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    WorkerProcHandle,
)
from vllm.v1.executor.socket_utils import (
    RPC_PROTOCOL_VERSION,
    remaining_timeout,
    sock_recv,
    sock_send,
)

logger = init_logger(__name__)

# Default base port; each worker rank uses base_port + rank
_DEFAULT_MP_RPC_READY_BASE_PORT = envs.VLLM_MP_RPC_READY_BASE_PORT


def _get_rpc_base_port() -> int:
    return int(
        os.environ.get("VLLM_MP_RPC_READY_BASE_PORT", _DEFAULT_MP_RPC_READY_BASE_PORT)
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MultiprocRPCExecutor(MultiprocExecutor):
    """MultiprocExecutor variant where workers connect via TCP (mp_rpc mode).

    Workers are started independently (e.g. via run_mp_rpc_worker.py) and
    connect back to the executor.  After the initial TCP handshake the
    communication uses TCP MessageQueues with the same RPC payloads,
    so collective_rpc / execute_model / sample_tokens are all inherited.
    """

    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = False):
        assert not monitor_workers, (
            "monitor_workers is not supported in mp_rpc mode "
            "(workers are external processes)"
        )
        parallel_config = vllm_config.parallel_config
        if parallel_config.nnodes_within_dp != 1:
            raise ValueError(
                "mp_rpc uses one executor per DP group; do not combine it with "
                "native --nnodes TP/PP launch."
            )
        if parallel_config.enable_elastic_ep or parallel_config.enable_fault_tolerance:
            raise ValueError("mp_rpc does not support elastic EP or fault tolerance")
        if parallel_config.numa_bind:
            raise ValueError("mp_rpc requires NUMA binding in the external launcher")
        if vllm_config.model_config is not None:
            mm_config = vllm_config.model_config.multimodal_config
            if (
                mm_config is not None
                and mm_config.mm_processor_cache_type == "shm"
                and mm_config.mm_processor_cache_gb > 0
            ):
                raise ValueError(
                    "mp_rpc workers do not share a reader lock; use the lru "
                    "multimodal processor cache or disable that cache."
                )
        # Force TCP for all ZMQ channels so externally-started workers on
        # separate nodes (which may share /dev/shm but have separate /tmp
        # filesystems) can connect via network instead of IPC sockets.
        os.environ["VLLM_FORCE_ZMQ_TCP"] = "1"
        super().__init__(vllm_config, monitor_workers=False)

    # ------------------------------------------------------------------
    # Override: worker bootstrap via TCP instead of subprocess spawn
    # ------------------------------------------------------------------

    def _create_workers(
        self,
        global_start_rank: int,
        distributed_init_method: str,
        scheduler_output_handle: "Handle | None",
    ) -> "list[WorkerProcHandle]":
        """Listen on TCP sockets, send config to pre-started workers, wait
        for their READY replies, and return the ready handles.

        Two-pass design for cross-node correctness:
          Pass 1 – accept all connections and validate worker identities.
          Pass 2 – send configs.  distributed_init_method is patched to use
                   rank-0's actual peer IP (it hosts the PyTorch TCPStore
                   rendezvous), not the executor's IP.
        """
        if scheduler_output_handle is None or global_start_rank != 0:
            raise RuntimeError(
                "mp_rpc executor must run on the DP leader (node_rank_within_dp == 0)"
            )

        base_port = _get_rpc_base_port()
        executor_ip = get_ip()
        n = self.local_world_size
        if not 1 <= base_port <= 65536 - n:
            raise ValueError("MP RPC worker ports must be in [1, 65535]")
        deadline = time.monotonic() + envs.VLLM_ENGINE_READY_TIMEOUT_S

        # -- Pass 1: open listen sockets, accept all workers ------------------
        conns: list[socket.socket] = []
        peer_ips: list[str] = []
        worker_hellos: list[dict[str, Any]] = []
        success = False
        try:
            with ExitStack() as stack:
                servers = []
                for rank in range(n):
                    server = stack.enter_context(
                        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    )
                    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    server.bind(("0.0.0.0", base_port + rank))
                    server.listen(1)
                    servers.append(server)
                for rank, server in enumerate(servers):
                    server.settimeout(remaining_timeout(deadline))
                    logger.info(
                        "Waiting for worker rank %d at %s:%d",
                        rank,
                        executor_ip,
                        base_port + rank,
                    )
                    conn, addr = server.accept()
                    conns.append(conn)
                    conn.settimeout(remaining_timeout(deadline))
                    hello = sock_recv(conn)
                    if (
                        not isinstance(hello, dict)
                        or hello.get("protocol_version") != RPC_PROTOCOL_VERSION
                        or hello.get("rank") != rank
                        or type(hello.get("local_rank")) is not int
                        or hello["local_rank"] < 0
                    ):
                        raise ValueError(f"Invalid MP RPC HELLO for rank {rank}")
                    worker_hellos.append(hello)
                    peer_ips.append(addr[0])
                    logger.info("Worker rank %d connected from %s", rank, addr)

            # MPI-launched workers need not share the executor's /tmp.
            # Rank 0 selects the port on the host that will own the TCPStore.
            rank0_ip = peer_ips[0]
            dist_port = worker_hellos[0].get("distributed_init_port")
            if type(dist_port) is not int or not 1 <= dist_port <= 65535:
                raise ValueError("Rank 0 must supply a valid TCPStore port")
            distributed_init_method = get_distributed_init_method(rank0_ip, dist_port)
            logger.info("MP RPC distributed_init_method: %s", distributed_init_method)

            # -- Pass 2: send configs -----------------------------------------
            # vllm_envs propagates VLLM_FORCE_ZMQ_TCP=1 (set in __init__)
            # along with other VLLM_* settings to each worker.
            unready_workers: list[UnreadyWorkerProcHandle] = []
            vllm_envs = {k: v for k, v in os.environ.items() if k.startswith("VLLM_")}
            for local_rank in range(n):
                global_rank = global_start_rank + local_rank
                conn = conns[local_rank]
                config_payload = {
                    "protocol_version": RPC_PROTOCOL_VERSION,
                    "vllm_config": self.vllm_config,
                    "local_rank": worker_hellos[local_rank]["local_rank"],
                    "rank": global_rank,
                    "distributed_init_method": distributed_init_method,
                    "input_shm_handle": scheduler_output_handle,
                    "is_driver_worker": self._is_driver_worker(global_rank),
                    "vllm_envs": vllm_envs,
                }
                conn.settimeout(remaining_timeout(deadline))
                sock_send(conn, config_payload)
                logger.debug("Config sent to worker rank %d", global_rank)
                unready_workers.append(
                    UnreadyWorkerProcHandle(
                        proc=None,
                        rank=global_rank,
                        ready_pipe=conn,
                        death_writer=None,
                    )
                )

            workers = _wait_for_ready_rpc(unready_workers, deadline)
            success = True
            return workers
        finally:
            if not success:
                import contextlib

                for conn in conns:
                    with contextlib.suppress(Exception):
                        conn.close()

    # ------------------------------------------------------------------
    # Override: shutdown via broadcast MQ command instead of death pipe
    # ------------------------------------------------------------------

    def shutdown(self):
        """Shut down the executor.  Signal workers via the broadcast MQ
        (they have no death pipe in RPC mode)."""
        if not getattr(self, "shutting_down", False):
            self.shutting_down = True
            if rpc_mq := getattr(self, "rpc_broadcast_mq", None):
                logger.info("Sending shutdown command to all RPC workers…")
                try:
                    rpc_mq.enqueue(("shutdown", (), {}, None))
                except Exception as exc:
                    logger.warning("Could not send shutdown command: %s", exc)

        # Clean up MQs; workers will shut down their own response MQs.
        if rpc_mq := getattr(self, "rpc_broadcast_mq", None):
            rpc_mq.shutdown()
            self.rpc_broadcast_mq = None
        for mq in getattr(self, "response_mqs", []):
            mq.shutdown()
        self.response_mqs = []
        for worker in getattr(self, "workers", []):
            worker.worker_response_mq = None

    # ------------------------------------------------------------------
    # No-op: can't monitor external processes via sentinel
    # ------------------------------------------------------------------

    def start_worker_monitor(self, inline: bool = False) -> None:
        logger.debug("start_worker_monitor: skipped in mp_rpc mode")


def _wait_for_ready_rpc(
    unready_handles: list[UnreadyWorkerProcHandle],
    deadline: float,
) -> list[WorkerProcHandle]:
    """Receive the READY reply from each worker (via their TCP socket) and
    build WorkerProcHandle objects."""
    logger.info("Waiting for READY replies from %d RPC workers…", len(unready_handles))
    ready_handles: list[WorkerProcHandle] = []

    for uw in unready_handles:
        rank = uw.rank
        sock = uw.ready_pipe
        assert isinstance(sock, socket.socket), "RPC mode should use socket"
        try:
            sock.settimeout(remaining_timeout(deadline))
            response_obj = sock_recv(sock)
            assert isinstance(response_obj, dict), "Response should be a dict"
            response: dict[str, Any] = response_obj
        except Exception as exc:
            raise RuntimeError(
                f"Worker rank {rank} failed to send READY reply"
            ) from exc
        finally:
            sock.close()

        if response.get("status") != WorkerProc.READY_STR:
            raise RuntimeError(
                f"Worker rank {rank} initialization failed: "
                f"{response.get('error', 'unknown')}"
            )

        logger.info("Worker rank %d is READY", rank)
        worker_response_mq = MessageQueue.create_from_handle(response["handle"], -1)
        peer_response_handles = response.get("peer_response_handles", [])
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        ready_handles.append(
            WorkerProcHandle.from_unready_handle(
                uw,
                worker_response_mq=worker_response_mq,
                peer_worker_response_mqs=peer_worker_response_mqs,
            )
        )

    return ready_handles
