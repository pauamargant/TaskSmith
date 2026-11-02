#!/usr/bin/env python3
"""Materials retrieval server entry point.

This module provides the main entry point for running the materials retrieval
server. It uses a class-based architecture with proper lifecycle management and
no global variables or import side effects.

Usage:
    python -m scenesmith.agent_utils.materials_retrieval_server.standalone_server
    python -m scenesmith.agent_utils.materials_retrieval_server.standalone_server --port 7018
    python -m scenesmith.agent_utils.materials_retrieval_server.standalone_server --no-preload
"""

import argparse
import logging
import signal
import sys

from .server_manager import MaterialsRetrievalServer

console_logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Materials retrieval server for semantic material search using CLIP",
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
        default=7018,
        help="Port to bind to, default: %(default)s.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=True,
        help="Preload the materials retriever (includes CLIP model loading) on server "
        "start for consistent performance (default: True).",
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
        "--materials-path",
        type=str,
        default=None,
        help="Path to materials data directory. If not specified, uses "
        "default 'data/materials'.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=None,
        help="Path to preprocessed embeddings directory. If not specified, uses "
        "default 'data/materials/embeddings'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top CLIP candidates to return (default: %(default)s).",
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
    """Main entry point for the materials retrieval server.

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
        # Build config from CLI args if paths are provided.
        materials_config = None
        if args.materials_path or args.embeddings_path:
            from pathlib import Path

            from scenesmith.agent_utils.materials_retrieval_server.config import (
                MaterialsConfig,
            )

            materials_path = Path(args.materials_path or "data/materials")
            embeddings_path = Path(args.embeddings_path or "data/materials/embeddings")

            materials_config = MaterialsConfig(
                data_path=materials_path,
                embeddings_path=embeddings_path,
                use_top_k=args.top_k,
            )

        # Create and start the server.
        server = MaterialsRetrievalServer(
            host=args.host,
            port=args.port,
            preload_retriever=args.preload,
            materials_config=materials_config,
        )
        server.start()

        console_logger.info(
            f"Materials retrieval server running on {args.host}:{args.port}"
        )
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
