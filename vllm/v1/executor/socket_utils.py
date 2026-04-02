# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared socket utilities for RPC-based executors."""

import pickle
import socket
from multiprocessing.connection import Connection


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


def sock_recv_exact(sock: Connection | socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket or pipe.

    This is a shared utility that works with both socket.socket objects
    and multiprocessing.Connection objects.
    """
    buf = bytearray()
    while len(buf) < n:
        if isinstance(sock, socket.socket):
            chunk = sock.recv(n - len(buf))
        else:
            # multiprocessing.Connection.recv_bytes(maxlength)
            chunk = sock.recv_bytes(n - len(buf))
        if not chunk:
            raise EOFError(f"Socket closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)
