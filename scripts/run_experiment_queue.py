#!/usr/bin/env python3
"""Run a sequence of shell commands with timeouts and server cleanup.

Use this to run E1-style ablations sequentially. If a run exceeds the hard timeout,
the script terminates the run's process group, kills known TaskSmith auxiliary servers,
and proceeds to the next command.

Example:
  E0_BASE_PATH=outputs/YYYY-MM-DD/e0_base_HH-MM-SS_xxxxxx \\
    python scripts/run_experiment_queue.py \\
      --timeout-min 120 \\
      --commands-file .plans/e1_queue.txt

Resume from a specific experiment (skip already-completed runs):
  python scripts/run_experiment_queue.py \\
    --timeout-min 120 \\
    --commands-file .plans/e1_queue.txt \\
    --start-from e1c_low_views
"""

from __future__ import annotations

import argparse
import os
import select
import shlex
import signal
import subprocess
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PortRange:
    """Inclusive TCP port range used for cleanup between runs."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start <= 0 or self.end <= 0 or self.end < self.start:
            raise ValueError(f"Invalid port range: {self.start}-{self.end}")


def _pid_owned_by_current_user(pid: int) -> bool:
    """Return True iff *pid* belongs to the current OS user (best-effort).

    Prevents cleanup from killing other users' processes on shared machines.
    """
    try:
        data = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return False
    for line in data.splitlines():
        if line.startswith("Uid:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) == os.getuid()
                except ValueError:
                    return False
    return False


DEFAULT_PORT_RANGES = [
    # Retrieval servers (HSSD/materials/articulated/objaverse/geometry).
    PortRange(7005, 7009),
    # BlenderServer (rendering) ports.
    PortRange(8000, 8350),
    # Convex decomposition server ports.
    PortRange(7100, 7450),
]

_DEFAULT_SUCCESS_SUBSTRINGS = (
    "Completed task: generate_scenes",
    "Experiment execution completed in",
)

_DEFAULT_SERVER_PGREP_PATTERNS = (
    "scenesmith.agent_utils.hssd_retrieval_server",
    "scenesmith.agent_utils.materials_retrieval_server",
    "scenesmith.agent_utils.articulated_retrieval_server",
    "scenesmith.agent_utils.objaverse_retrieval_server",
    "scenesmith.agent_utils.geometry_generation_server",
    "scenesmith.agent_utils.convex_decomposition_server",
    "scenesmith.agent_utils.blender.standalone_server",
    "scenesmith.agent_utils.blender.server_app",
)


class _InactivityTimeout(RuntimeError):
    """Raised when a subprocess produces no output for too long."""


def _kill_pids(*, pids: Iterable[int], grace_s: float) -> None:
    uniq = sorted({int(p) for p in pids if int(p) > 0})
    if not uniq:
        return
    for pid in uniq:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(max(0.0, grace_s))
    for pid in uniq:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def kill_ports(*, port_ranges: list[PortRange], grace_s: float) -> None:
    """Kill any processes listening on these ports (best-effort)."""
    pids: set[int] = set()
    for rng in port_ranges:
        for port in range(rng.start, rng.end + 1):
            try:
                out = subprocess.check_output(
                    ["lsof", "-ti", f":{port}"], text=True
                ).strip()
            except subprocess.CalledProcessError:
                continue
            for line in out.splitlines():
                try:
                    pid = int(line.strip())
                    if _pid_owned_by_current_user(pid):
                        pids.add(pid)
                except ValueError:
                    pass
    _kill_pids(pids=pids, grace_s=grace_s)


def kill_known_servers_by_name(*, patterns: tuple[str, ...], grace_s: float) -> None:
    """Kill known TaskSmith auxiliary servers by command-line pattern (best-effort)."""
    if not patterns:
        return
    uid = str(os.getuid())
    pids: set[int] = set()
    for pat in patterns:
        if not pat:
            continue
        try:
            out = subprocess.check_output(
                ["pgrep", "-u", uid, "-f", pat], text=True
            ).strip()
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            try:
                pids.add(int(line.strip()))
            except ValueError:
                pass
    _kill_pids(pids=pids, grace_s=grace_s)


def _read_commands(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _extract_name(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    for token in parts:
        if token.startswith("+name="):
            name = token.split("=", 1)[1].strip()
            return name or None
    return None


def run_one(
    *,
    cmd: str,
    timeout_s: float,
    stuck_timeout_s: float,
    kill_after_s: float,
    success_substrings: tuple[str, ...],
    success_grace_s: float,
    dry_run: bool = False,
) -> tuple[int, bool]:
    """Run one command in its own process group.

    - Hard timeout: SIGTERM + SIGKILL after timeout_s.
    - Inactivity timeout: if no output for stuck_timeout_s, move on.
    - success_grace_s: after success markers appear, terminate if process lingers.

    Returns (return_code, saw_success_marker).
    """
    if dry_run:
        safe = shlex.join(shlex.split(cmd)) if cmd.strip() else cmd
        print(f"[queue] DRY-RUN would execute: {safe}")
        return 0, True

    start = time.time()
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    saw: dict[str, bool] = {s: False for s in success_substrings}
    success_seen_at: float | None = None
    last_output_at = time.time()

    def _terminate_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def _note(line: str) -> None:
        nonlocal success_seen_at, last_output_at
        last_output_at = time.time()
        for s in success_substrings:
            if not saw[s] and s in line:
                saw[s] = True
        if success_seen_at is None and all(saw.values()):
            success_seen_at = time.time()

    try:
        while True:
            now = time.time()

            if now - start > timeout_s:
                raise subprocess.TimeoutExpired(cmd, timeout_s)

            if stuck_timeout_s > 0 and now - last_output_at > stuck_timeout_s:
                raise _InactivityTimeout(f"No output for {stuck_timeout_s/60:.1f}min")

            if (
                success_seen_at is not None
                and success_grace_s > 0
                and now - success_seen_at > success_grace_s
                and proc.poll() is None
            ):
                _terminate_group(signal.SIGTERM)
                deadline = time.time() + min(kill_after_s, 10.0)
                while time.time() < deadline:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.25)
                if proc.poll() is None:
                    _terminate_group(signal.SIGKILL)
                return 0, True

            rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
            if rlist:
                line = proc.stdout.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    _note(line)
                elif proc.poll() is not None:
                    break
            elif proc.poll() is not None:
                break

        return int(proc.returncode or 0), all(saw.values())

    except subprocess.TimeoutExpired:
        _terminate_group(signal.SIGTERM)
        deadline = start + timeout_s + kill_after_s
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        _terminate_group(signal.SIGKILL)
        return 124, False

    except _InactivityTimeout as exc:
        print(f"[queue] WARNING: stuck ({exc}), moving on.", file=sys.stderr)
        _terminate_group(signal.SIGTERM)
        deadline = time.time() + kill_after_s
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        _terminate_group(signal.SIGKILL)
        return 126, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--commands-file", type=Path, required=True)
    parser.add_argument("--timeout-min", type=float, default=120.0)
    parser.add_argument(
        "--stuck-min",
        type=float,
        default=12.0,
        help="Treat run as stuck if no output for this many minutes (0 disables).",
    )
    parser.add_argument("--kill-after-s", type=float, default=30.0)
    parser.add_argument(
        "--success-grace-s",
        type=float,
        default=60.0,
        help=(
            "After success markers appear, allow this many seconds for the process "
            "to exit; terminate it if it does not. Set 0 to disable."
        ),
    )
    parser.add_argument("--server-kill-grace-s", type=float, default=2.0)
    parser.add_argument(
        "--kill-ports",
        action="append",
        default=[],
        metavar="A-B",
        help="Additional port range to clean up between runs (e.g. 8000-8350). Repeatable.",
    )
    parser.add_argument(
        "--require-success-marker",
        action="store_true",
        default=True,
        help="Fail a command even if rc=0 unless the success marker appeared in output.",
    )
    parser.add_argument(
        "--no-require-success-marker",
        action="store_false",
        dest="require_success_marker",
    )
    parser.add_argument(
        "--success-substring",
        action="append",
        default=list(_DEFAULT_SUCCESS_SUBSTRINGS),
        help="Substring that must appear in output for the command to be considered successful. Repeatable.",
    )
    parser.add_argument(
        "--kill-pattern",
        action="append",
        default=list(_DEFAULT_SERVER_PGREP_PATTERNS),
        help="Process pattern to kill between runs (pgrep -u <uid> -f PATTERN). Repeatable.",
    )
    parser.add_argument(
        "--start-from",
        metavar="NAME",
        default=None,
        help=(
            "Skip commands until the one with +name=NAME, then run from there. "
            "Useful to resume a partially-completed queue without editing the file."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands that would run (respecting --start-from) without executing them.",
    )
    args = parser.parse_args()

    if not args.commands_file.exists():
        raise FileNotFoundError(str(args.commands_file))

    port_ranges = list(DEFAULT_PORT_RANGES)
    for spec in args.kill_ports:
        if "-" not in spec:
            raise ValueError(f"Invalid --kill-ports value: {spec!r} (expected A-B)")
        a, b = spec.split("-", 1)
        port_ranges.append(PortRange(int(a), int(b)))

    commands = _read_commands(args.commands_file)
    if not commands:
        print(f"No commands found in {args.commands_file}", file=sys.stderr)
        return 2

    # Apply --start-from: skip commands until we see +name=NAME.
    if args.start_from:
        skip_until = args.start_from.strip()
        start_idx = None
        for i, cmd in enumerate(commands):
            if _extract_name(cmd) == skip_until:
                start_idx = i
                break
        if start_idx is None:
            print(
                f"[queue] ERROR: --start-from {skip_until!r} not found in queue "
                f"(names: {[_extract_name(c) for c in commands]})",
                file=sys.stderr,
            )
            return 2
        skipped = start_idx
        commands = commands[start_idx:]
        print(f"[queue] --start-from {skip_until!r}: skipping {skipped} command(s).")

    timeout_s = args.timeout_min * 60.0
    stuck_timeout_s = args.stuck_min * 60.0
    success_substrings = tuple(s for s in args.success_substring if s)

    if args.require_success_marker and not success_substrings:
        raise ValueError(
            "--require-success-marker needs at least one --success-substring"
        )

    print(
        f"[queue] commands={len(commands)} timeout={args.timeout_min:.1f}min "
        f"stuck={args.stuck_min:.1f}min kill_after={args.kill_after_s:.1f}s "
        f"require_success={args.require_success_marker}"
        + (" DRY-RUN" if args.dry_run else "")
    )

    results: list[tuple[str, int, bool]] = []
    total = len(commands)
    for idx, cmd in enumerate(commands, start=1):
        safe = shlex.join(shlex.split(cmd)) if cmd.strip() else cmd
        print(f"\n[queue] ({idx}/{total}) {safe}")

        rc, saw_success = run_one(
            cmd=cmd,
            timeout_s=timeout_s,
            stuck_timeout_s=stuck_timeout_s,
            kill_after_s=float(args.kill_after_s),
            success_substrings=success_substrings,
            success_grace_s=float(args.success_grace_s),
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            kill_ports(port_ranges=port_ranges, grace_s=float(args.server_kill_grace_s))
            kill_known_servers_by_name(
                patterns=tuple(p for p in args.kill_pattern if p),
                grace_s=float(args.server_kill_grace_s),
            )

        if rc == 124:
            print(f"[queue] TIMEOUT (>{args.timeout_min:.1f}min). cleaned servers.")
        elif rc == 126:
            print(
                f"[queue] STUCK (>{args.stuck_min:.1f}min inactivity). cleaned servers."
            )
        elif (
            rc == 0
            and args.require_success_marker
            and not saw_success
            and not args.dry_run
        ):
            print(
                "[queue] WARNING: rc=0 but success marker not seen; treating as failure."
            )
            rc = 125

        print(f"[queue] done rc={rc}")
        results.append((cmd, rc, saw_success))

    print("\n[queue] summary:")
    label = {124: "TIMEOUT", 125: "NO_SUCCESS_MARKER", 126: "STUCK"}
    for cmd, rc, saw_success in results:
        status = label.get(rc, str(rc))
        marker = f" saw_success={saw_success}" if args.require_success_marker else ""
        print(f"- rc={status}{marker} cmd={cmd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
