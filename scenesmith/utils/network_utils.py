"""Network utilities for server management."""

import logging
import os
import random
import signal
import socket
import subprocess
import time

console_logger = logging.getLogger(__name__)


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: The host address to check.
        port: The port number to check.

    Returns:
        True if the port is available, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
        return True
    except OSError:
        return False


def kill_pids_on_port(port: int, grace_s: float = 2.0) -> None:
    """SIGTERM then SIGKILL any process listening on port.

    Sends SIGTERM, waits grace_s, then SIGKILL to anything still alive.
    Silently no-ops if lsof is unavailable or no process is found.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    except Exception as exc:
        console_logger.warning(f"kill_pids_on_port: lsof failed: {exc}")
        return

    if not pids:
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    time.sleep(grace_s)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # already gone


def find_available_port(host: str, port_range: tuple[int, int]) -> int | None:
    """Find an available port in the given range.

    Ports are tried in random order to reduce collision probability when
    multiple processes are searching for ports simultaneously.

    Args:
        host: The host address to check.
        port_range: Tuple of (start_port, end_port) inclusive.

    Returns:
        An available port, or None if no ports are available.
    """
    start_port, end_port = port_range
    ports = list(range(start_port, end_port + 1))
    random.shuffle(ports)
    for port in ports:
        if is_port_available(host, port):
            return port
    return None
