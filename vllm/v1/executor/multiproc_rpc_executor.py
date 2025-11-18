# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import pickle
import signal
import socket
import sys
import threading
import time
import traceback
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Optional, Union, cast

import cloudpickle

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.multiproc_worker_utils import (
    _add_prefix, set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_loopback_ip,
                        get_open_port)
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class MultiprocRPCExecutor(Executor):

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())
        self.distributed_init_method = distributed_init_method

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            0,  # self.world_size,
            max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            ready_sockets = []
            host_ip = "0.0.0.0"
            display_ip = get_loopback_ip()
            for rank in range(self.world_size):
                port = envs.VLLM_MP_RPC_READY_BASE_PORT + rank
                server_socket = socket.socket(socket.AF_INET,
                                              socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET,
                                         socket.SO_REUSEADDR, 1)
                server_socket.bind((host_ip, port))
                server_socket.listen(1)
                ready_sockets.append(server_socket)
                unready_workers.append(
                    UnreadyWorkerProcHandle(
                        proc=None,  # type: ignore
                        rank=rank,
                        ready_pipe=server_socket))  # type: ignore

            logger.info("Executor waiting for %d workers to connect...",
                        self.world_size)

            for rank, sock in enumerate(ready_sockets):
                port = envs.VLLM_MP_RPC_READY_BASE_PORT + rank
                logger.info("  - Worker Rank %d should connect to %s:%d", rank,
                            display_ip, port)

            # Step 1: Accept connections from all workers
            connections: list[Optional[socket.socket]] = [None
                                                          ] * self.world_size
            for unready_handle in unready_workers:
                server_socket = unready_handle.ready_pipe
                rank = unready_handle.rank
                conn, addr = server_socket.accept()
                logger.info("Accepted connection from worker rank %d at %s",
                            rank, str(addr))
                connections[rank] = conn
                server_socket.close()  # Close listening socket

            # Step 2: Sequentially send configs to all workers
            self._send_configs_to_workers(connections, scheduler_output_handle)

            # Step 3: Sequentially wait for ready signals from all workers
            self.workers = self._wait_for_workers_ready(
                connections, unready_workers)

            # Ensure message queues are ready.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            logger.warning("Remote worker monitoring is not implemented. "
                           "System relies on RPC timeouts to detect failures.")
            success = True
        finally:
            if not success:
                if self.rpc_broadcast_mq:
                    logger.info("Sending shutdown command to all workers...")
                    try:
                        self.rpc_broadcast_mq.enqueue(
                            ("shutdown", (), {}, None))
                    except Exception as e:
                        logger.warning(
                            "Could not send shutdown command to workers: %s",
                            e)

                # Clean up the worker procs if there was a failure.
                for handle in unready_workers:
                    if handle.ready_pipe:
                        handle.ready_pipe.close()

        if self.max_concurrent_batches > 1:
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        non_block = self.max_concurrent_batches > 1

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            (output, ) = self.collective_rpc(
                "execute_model",
                args=(scheduler_output, ),
                unique_reply_rank=self.output_rank,
                non_block=non_block,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
            return output

        # get output from all workers
        outputs = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

        # aggregate all workers output to a single output
        if non_block:
            return self.kv_output_aggregator.async_aggregate(
                outputs, self.output_rank)
        return self.kv_output_aggregator.aggregate(outputs, self.output_rank)

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue(
                (send_method, args, kwargs, unique_reply_rank))

            workers = (self.workers[unique_reply_rank],
                       ) if unique_reply_rank is not None else self.workers
            responses = []

            def get_response(w: "WorkerProcHandle",
                             dequeue_timeout: Optional[float] = None,
                             cancel_event: Optional[threading.Event] = None):
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout, cancel=cancel_event)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause")
                return result

            for w in workers:
                dequeue_timeout = None if deadline is None else (
                    deadline - time.monotonic())

                if non_block:
                    result = self.io_thread_pool.submit(  # type: ignore
                        get_response, w, dequeue_timeout, self.shutdown_event)
                else:
                    result = get_response(w, dequeue_timeout)

                responses.append(result)

            return responses
        except TimeoutError as e:
            raise TimeoutError(f"RPC call to {method} timed out.") from e

    def shutdown(self):
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self.shutdown_event.set()

            if self.rpc_broadcast_mq:
                logger.info("Sending shutdown command to all workers...")
                try:
                    self.rpc_broadcast_mq.enqueue(("shutdown", (), {}, None))
                except Exception as e:
                    logger.warning(
                        "Could not send shutdown command to workers: %s", e)

            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                self.io_thread_pool = None
        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        return self.world_size - self.parallel_config.tensor_parallel_size

    def _send_configs_to_workers(self,
                                 connections: list[Optional[socket.socket]],
                                 scheduler_output_handle: Handle):
        """Sequentially sends configuration to all connected workers."""
        logger.info("Sending configuration to all workers...")

        # MODIFIED: Gather all VLLM_ prefixed environment variables.
        vllm_envs = {
            k: v
            for k, v in os.environ.items() if k.startswith("VLLM_")
        }

        config_payload = pickle.dumps({
            "vllm_config": self.vllm_config,
            "distributed_init_method": self.distributed_init_method,
            "input_shm_handle": scheduler_output_handle,
            "vllm_envs": vllm_envs,  # Add envs to the payload
        })

        for rank, conn in enumerate(connections):
            assert conn is not None
            try:
                conn.sendall(len(config_payload).to_bytes(4, 'big'))
                conn.sendall(config_payload)
                logger.info("Configuration sent to worker rank %d.", rank)
            except Exception as e:
                logger.error("Failed to send config to worker %d: %s", rank, e)
                raise e

    def _wait_for_workers_ready(
        self, connections: list[Optional[socket.socket]],
        unready_proc_handles: list["UnreadyWorkerProcHandle"]
    ) -> list["WorkerProcHandle"]:
        """Sequentially waits for a READY signal from all workers."""
        logger.info("Waiting for ready signal from all workers...")
        ready_proc_handles: list[Optional[WorkerProcHandle]] = (
            [None] * self.world_size)

        e = Exception(
            "WorkerProc initialization failed. See logs for details.")

        for rank, conn in enumerate(connections):
            unready_proc_handle = unready_proc_handles[rank]
            assert conn is not None
            try:
                with conn:
                    len_data = conn.recv(4)
                    if not len_data:
                        raise ConnectionAbortedError(
                            "Worker %d disconnected before "
                            "sending ready signal.", rank)

                    payload_len = int.from_bytes(len_data, 'big')
                    payload = conn.recv(payload_len, socket.MSG_WAITALL)
                    if not payload:
                        raise ConnectionAbortedError(
                            f"Worker {rank} sent an empty ready payload.")

                    response: dict[str, Any] = pickle.loads(payload)

                    if response["status"] != WorkerProc.READY_STR:
                        logger.error("Worker %d failed to initialize: %s",
                                     rank, response.get('error'))
                        raise e

                    logger.info("Received ready signal from worker rank %d.",
                                rank)
                    worker_response_mq = MessageQueue.create_from_handle(
                        response["handle"], 0)
                    ready_proc_handles[rank] = (
                        WorkerProcHandle.from_unready_handle(
                            unready_proc_handle, worker_response_mq))

            except Exception as e_inner:
                e.__suppress_context__ = True
                raise e from e_inner

        return cast(list[WorkerProcHandle], ready_proc_handles)


@dataclass
class UnreadyWorkerProcHandle:
    proc: Optional[BaseProcess]
    rank: int
    ready_pipe: socket.socket


@dataclass
class WorkerProcHandle:
    proc: Optional[BaseProcess]
    rank: int
    worker_response_mq: MessageQueue

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = (
            rank % vllm_config.parallel_config.tensor_parallel_size == 0)
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        pid = os.getpid()
        _add_prefix(sys.stdout, f"VllmWorker rank={rank}", pid)
        _add_prefix(sys.stderr, f"VllmWorker rank={rank}", pid)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Initializes a message queue for sending the model output
        # TODO: dynamically detect the number of local readers
        self.worker_response_mq = MessageQueue(1, n_local_reader=0)

        # Initialize device and loads weights
        self.worker.init_device()
        self.worker.load_model()

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(**kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        executor_addr = kwargs.pop("executor_addr")
        ready_port = kwargs.pop("ready_port")
        local_rank = kwargs.pop("local_rank")
        rank = kwargs.pop("rank")

        ready_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Set connection timeout parameters
        max_retry_time = 300  # 300 seconds maximum wait time
        retry_interval = 60  # 60 seconds between retries
        start_time = time.time()

        try:
            # Add connection retry logic with logging
            connected = False
            last_log_time = 0.0

            while not connected and (time.time() -
                                     start_time) < max_retry_time:
                try:
                    ready_socket.connect((executor_addr, ready_port))
                    connected = True
                except ConnectionRefusedError:
                    current_time = time.time()
                    if current_time - last_log_time >= retry_interval:
                        logger.debug(
                            "Waiting for executor connection... "
                            "Executor: %s:%d, Time elapsed: %ds",
                            executor_addr, ready_port,
                            int(current_time - start_time))
                        last_log_time = current_time

                    # Check if we should continue waiting
                    if (time.time() - start_time) < max_retry_time:
                        time.sleep(
                            retry_interval)  # Sleep briefly before retry
                    else:
                        raise ConnectionError(
                            f"Failed to connect to executor at "
                            f"{executor_addr}:{ready_port} after "
                            f"{max_retry_time} seconds") from None

            if not connected:
                raise ConnectionError(
                    f"Unable to establish connection to executor "
                    f"after {max_retry_time} seconds")

            with ready_socket:
                # 1. RECEIVE config from executor
                len_data = ready_socket.recv(4)
                if not len_data:
                    raise ConnectionError(
                        "Executor closed connection during config exchange.")
                payload_len = int.from_bytes(len_data, 'big')

                payload = ready_socket.recv(payload_len, socket.MSG_WAITALL)
                if not payload:
                    raise ConnectionError(
                        "Did not receive config payload from executor.")

                config_data = pickle.loads(payload)

                # Set environment variables received from the executor.
                # This should be done before initializing other components.
                vllm_envs = config_data.get("vllm_envs", {})
                for k, v in vllm_envs.items():
                    existing_v = os.getenv(k)
                    if existing_v is not None and existing_v != v:
                        logger.warning(
                            "Overwriting worker's environment variable '%s'. "
                            "Existing value: '%s', New value: '%s'",
                            (k, existing_v, v))
                    os.environ[k] = v

                vllm_config = config_data["vllm_config"]
                distributed_init_method = config_data[
                    "distributed_init_method"]
                input_shm_handle = config_data["input_shm_handle"]

                worker = WorkerProc(vllm_config, local_rank, rank,
                                    distributed_init_method, input_shm_handle)

                # 2. SEND ready signal back to executor
                ready_payload = pickle.dumps({
                    "status":
                    WorkerProc.READY_STR,
                    "handle":
                    worker.worker_response_mq.export_handle(),
                })
                ready_socket.sendall(len(ready_payload).to_bytes(4, 'big'))
                ready_socket.sendall(ready_payload)

            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()

            worker.worker_busy_loop()

        except Exception as e:
            logger.exception(
                "WorkerProc failed during initialization or execution.")
            if ready_socket and ready_socket.fileno() != -1:
                error_payload = pickle.dumps({
                    "status":
                    "FAILURE",
                    "error":
                    str(e),
                    "traceback":
                    traceback.format_exc()
                })
                ready_socket.sendall(len(error_payload).to_bytes(4, 'big'))
                ready_socket.sendall(error_payload)
            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_socket and ready_socket.fileno() != -1:
                ready_socket.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()

            if method == "shutdown":
                logger.info("Received shutdown command. Exiting busy loop.")
                break

            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.worker_response_mq.enqueue(
                        (WorkerProc.ResponseStatus.FAILURE, str(e)))
                continue

            if output_rank is None or self.rank == output_rank:
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.SUCCESS, output))
