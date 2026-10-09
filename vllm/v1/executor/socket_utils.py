# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared socket utilities for RPC-based executors."""

import pickle
import socket
import time

RPC_PROTOCOL_VERSION = 1


def remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("MP RPC worker startup timed out")
    return remaining


def sock_send(sock: socket.socket, obj: object) -> None:
    """Send an object over a socket with length-prefix framing."""
    data = pickle.dumps(obj)
    sock.sendall(len(data).to_bytes(4, "big"))
    sock.sendall(data)


def sock_recv(sock: socket.socket) -> object:
    """Receive an object from a socket with length-prefix framing."""
    raw_len = sock_recv_exact(sock, 4)
    length = int.from_bytes(raw_len, "big")
    return pickle.loads(sock_recv_exact(sock, length))


def sock_recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a TCP stream."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError(f"Socket closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)
