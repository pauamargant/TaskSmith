#!/usr/bin/env python3
"""
HSSD retrieval server entry point.

This module provides the main entry point for running the HSSD retrieval server.
It uses a class-based architecture with proper lifecycle management and no global
variables or import side effects.
"""

import argparse
import logging
import signal
import sys

from .server_manager import HssdRetrievalServer

console_logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="HSSD retrieval server for semantic object search using CLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to, default: %(default)s.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7001,
        help="Port to bind to, default: %(default)s.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=True,
        help="Preload the HSSD retriever (includes CLIP model loading) on server start "
        "for consistent performance (default: True).",
    )
    parser.add_argument(
        "--no-preload",
        action="store_false",
        dest="preload",
        help="Disable retriever preloading. The retriever will be loaded lazily on "
        "first request, which can be useful for development/testing.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--hssd-data-path",
        type=str,
        default=None,
        help="Path to HSSD models directory. If not specified, uses environment "
        "variable HSSD_DATA_PATH or default 'data/hssd-models'.",
    )
    parser.add_argument(
        "--hssd-preprocessed-path",
        type=str,
        default=None,
        help="Path to preprocessed data directory. If not specified, uses environment "
        "variable HSSD_PREPROCESSED_PATH or default 'data/preprocessed'.",
    )
    parser.add_argument(
        "--hssd-top-k",
        type=int,
        default=5,
        help="Number of top CLIP candidates before size ranking (default: %(default)s).",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable debug level logging if True.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _try_enable_parent_death_sigterm() -> None:
    """Best-effort: ask Linux kernel to SIGTERM this process if its parent exits.

    This prevents orphaned server processes when the main TaskSmith run
    crashes or is killed. No-op on non-Linux platforms.
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes  # local import to avoid non-Linux issues

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        rc = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        if rc != 0:
            console_logger.warning("Failed to set PR_SET_PDEATHSIG (rc=%s)", rc)
    except Exception as e:
        console_logger.warning("Failed to enable parent-death SIGTERM: %s", e)


def main() -> int:
    """Main entry point for the HSSD retrieval server.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    _try_enable_parent_death_sigterm()
    args = parse_arguments()
    setup_logging(args.verbose)

    server = None

    def signal_handler(signum, _):
        """Handle shutdown signals gracefully."""
        console_logger.info(f"Received signal {signum}, shutting down...")
        if server:
            server.stop()
        sys.exit(0)

    # Register signal handlers for graceful shutdown.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Create and start the server.
        server = HssdRetrievalServer(
            host=args.host,
            port=args.port,
            preload_retriever=args.preload,
            hssd_data_path=args.hssd_data_path,
            hssd_preprocessed_path=args.hssd_preprocessed_path,
            hssd_top_k=args.hssd_top_k,
        )
        server.start()

        console_logger.info(f"HSSD retrieval server running on {args.host}:{args.port}")
        console_logger.info("Press Ctrl+C to stop the server")

        # Keep the main thread alive.
        # The server runs in its own thread, so we just wait.
        try:
            while server.is_running():
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            console_logger.info("Keyboard interrupt received")

    except Exception as e:
        console_logger.error(f"Failed to start server: {e}")
        return 1

    finally:
        if server:
            server.stop()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
