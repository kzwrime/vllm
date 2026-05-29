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
  python -m vllm.v1.executor.run_mp_rpc_worker --rank 0 --local-rank 0
  python -m vllm.v1.executor.run_mp_rpc_worker --rank 1 --local-rank 1
"""

import os
import socket
import time
from typing import Any

from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    WorkerProcHandle,
)
from vllm.v1.executor.socket_utils import sock_recv, sock_send

logger = init_logger(__name__)

# Default base port; each worker rank uses base_port + rank
_DEFAULT_MP_RPC_READY_BASE_PORT = 17300


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
    communication path (SHM MessageQueues) is identical to the local-mp case,
    so collective_rpc / execute_model / sample_tokens are all inherited.
    """

    supports_pp: bool = True

    def __init__(self, vllm_config, monitor_workers: bool = False):
        assert not monitor_workers, (
            "monitor_workers is not supported in mp_rpc mode "
            "(workers are external processes)"
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
          Pass 1 – accept all connections, record each worker's peer IP.
          Pass 2 – send configs.  distributed_init_method is patched to use
                   rank-0's actual peer IP (it hosts the PyTorch TCPStore
                   rendezvous), not the executor's IP.
        """
        base_port = _get_rpc_base_port()
        executor_ip = get_ip()
        n = self.local_world_size

        # -- Pass 1: open listen sockets, accept all workers ------------------
        conns: list[socket.socket] = []
        peer_ips: list[str] = []
        success = False
        try:
            for local_rank in range(n):
                global_rank = global_start_rank + local_rank
                port = base_port + global_rank
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("0.0.0.0", port))
                server.listen(1)
                logger.info(
                    "Waiting for worker rank %d/%d to connect at %s:%d …",
                    global_rank,
                    global_start_rank + n - 1,
                    executor_ip,
                    port,
                )
                conn, addr = server.accept()
                server.close()
                logger.info("Worker rank %d connected from %s", global_rank, addr)
                conns.append(conn)
                peer_ips.append(addr[0])

            # Rank 0 hosts the PyTorch TCPStore rendezvous; use its peer IP
            # if different from the executor (cross-node mode).
            rank0_ip = peer_ips[0]
            if rank0_ip != executor_ip:
                from vllm.utils.network_utils import get_distributed_init_method

                dist_port = get_open_port()
                distributed_init_method = get_distributed_init_method(
                    rank0_ip, dist_port
                )
                logger.info(
                    "Cross-node mode: patched distributed_init_method → %s",
                    distributed_init_method,
                )

            # -- Pass 2: send configs -----------------------------------------
            # vllm_envs propagates VLLM_FORCE_ZMQ_TCP=1 (set in __init__)
            # along with other VLLM_* settings to each worker.
            unready_workers: list[UnreadyWorkerProcHandle] = []
            vllm_envs = {k: v for k, v in os.environ.items() if k.startswith("VLLM_")}
            for local_rank in range(n):
                global_rank = global_start_rank + local_rank
                conn = conns[local_rank]
                config_payload = {
                    "vllm_config": self.vllm_config,
                    "local_rank": local_rank,
                    "rank": global_rank,
                    "distributed_init_method": distributed_init_method,
                    "input_shm_handle": scheduler_output_handle,
                    "is_driver_worker": self._is_driver_worker(global_rank),
                    "vllm_envs": vllm_envs,
                }
                sock_send(conn, config_payload)
                logger.debug("Config sent to worker rank %d", global_rank)
                unready_workers.append(
                    UnreadyWorkerProcHandle(
                        proc=None,  # type: ignore[arg-type]
                        rank=global_rank,
                        ready_pipe=conn,  # type: ignore[arg-type]
                        death_writer=None,
                    )
                )

            workers = _wait_for_ready_rpc(unready_workers)
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
        # Release references but do NOT call .shutdown() on response MQs —
        # workers own those and will clean them up themselves.
        self.response_mqs = []

    # ------------------------------------------------------------------
    # No-op: can't monitor external processes via sentinel
    # ------------------------------------------------------------------

    def start_worker_monitor(self, inline: bool = False) -> None:
        logger.debug("start_worker_monitor: skipped in mp_rpc mode")


def _wait_for_ready_rpc(
    unready_handles: list[UnreadyWorkerProcHandle],
) -> list[WorkerProcHandle]:
    """Receive the READY reply from each worker (via their TCP socket) and
    build WorkerProcHandle objects."""
    logger.info("Waiting for READY replies from %d RPC workers…", len(unready_handles))
    start = time.perf_counter()
    ready_handles: list[WorkerProcHandle | None] = [None] * len(unready_handles)

    for uw in unready_handles:
        rank = uw.rank
        sock = uw.ready_pipe  # type: ignore[assignment]
        assert isinstance(sock, socket.socket), "RPC mode should use socket"
        rank_start = time.perf_counter()
        try:
            response: dict[str, Any] = sock_recv(sock)  # type: ignore[assignment]
            assert isinstance(response, dict), "Response should be a dict"
        except Exception as exc:
            raise RuntimeError(
                f"Worker rank {rank} failed to send READY reply"
            ) from exc
        finally:
            sock.close()  # type: ignore[union-attr]

        if response.get("status") != WorkerProc.READY_STR:
            raise RuntimeError(
                f"Worker rank {rank} initialization failed: "
                f"{response.get('error', 'unknown')}"
            )

        logger.info("Worker rank %d is READY", rank)
        logger.info(
            "[TIMING] mp_rpc.wait_ready_rank_%d: %.6f seconds",
            rank,
            time.perf_counter() - rank_start,
        )
        worker_response_mq = MessageQueue.create_from_handle(response["handle"], 0)
        ready_handles[rank % len(ready_handles)] = WorkerProcHandle.from_unready_handle(
            uw,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=[],  # single-node only for now
        )

    logger.info(
        "[TIMING] mp_rpc.wait_ready_total: %.6f seconds",
        time.perf_counter() - start,
    )
    return ready_handles  # type: ignore[return-value]
