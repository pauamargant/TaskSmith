"""
watch_agents.py — Real-time TUI monitor for TaskSmith pipeline runs.

Reads outputs/<date>/<run>/scene_000/scene.log, run_analysis/usage_records.jsonl,
run_analysis/events.jsonl, and designer.db / critic.db (read-only SQLite).
Refreshes every 1 second; all file reads are incremental (byte-offset delta).

Usage:
    uv run python scripts/watch_agents.py
    uv run python scripts/watch_agents.py --run outputs/2026-05-18/my_run
Keys: q=quit, i=DB dump/dismiss, o=open latest floor-plan PNG

Record schemas:
  usage_records.jsonl: LLMCallRecord (scenesmith/experiment_tracking/records.py:53-91)
    fields: timestamp, source, stage_name, agent_label, agent_role, agent_type,
            tool_calls_in_response, input_tokens, output_tokens, total_tokens,
            duration_s, subsystem
  designer.db / critic.db — table agent_messages(id, session_id, message_data TEXT, created_at)
    message_data JSON keys: type (message|function_call|function_call_output|reasoning),
                             role, name, arguments, call_id, output
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import termios
import threading
import time
import tty

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Patterns for scene.log parsing
# ---------------------------------------------------------------------------
_RE_TOOL_CALLED = re.compile(r"INFO - Tool called: ([^\s(]+)")
_RE_HTTP = re.compile(r'httpx - INFO - HTTP Request: POST .+ "HTTP/1\.1 (\d+)')
_RE_ERROR = re.compile(r" - ERROR - (.+)")
_RE_LOG_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")


def _log_hms(line: str) -> str:
    m = _RE_LOG_TIMESTAMP.match(line)
    return m.group(1)[11:] if m else ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class DBState:
    path: Path
    last_id: int = 0
    messages: list[dict] = field(default_factory=list)


@dataclass
class AppState:
    run_path: Path
    scene_dir: Path

    log_offset: int = 0
    records_offset: int = 0
    events_offset: int = 0

    log_lines: deque = field(default_factory=lambda: deque(maxlen=200))
    records: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    activity: deque = field(default_factory=lambda: deque(maxlen=100))
    db_states: dict[str, DBState] = field(default_factory=dict)

    quit: bool = False
    key: str = ""
    show_db_dump: bool = False
    start_time: float = field(default_factory=time.monotonic)
    _old_term: list | None = None  # saved termios state for restoration


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def discover_run(override: str | None) -> tuple[Path, Path]:
    if override:
        run_path = Path(override)
        candidates = sorted(
            run_path.glob("scene_*/"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        scene_dir = candidates[0] if candidates else run_path / "scene_000"
        return run_path, scene_dir

    matches = sorted(
        glob.glob("outputs/*/*/scene_000"),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not matches:
        print("No outputs/*/*/scene_000 found. Use --run <path>.", file=sys.stderr)
        sys.exit(1)
    scene_dir = Path(matches[0])
    return scene_dir.parent, scene_dir


# ---------------------------------------------------------------------------
# Keyboard thread
# ---------------------------------------------------------------------------
def _key_reader(state: AppState) -> None:
    if not sys.stdin.isatty():
        return  # Not an interactive terminal; keyboard disabled
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return  # Not a real TTY (e.g. piped)
    state._old_term = old
    try:
        tty.setraw(fd)
        while not state.quit:
            ch = sys.stdin.read(1)
            state.key = ch
            if ch == "q":
                state.quit = True
                break
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Incremental readers
# ---------------------------------------------------------------------------
def _tail_log(state: AppState) -> None:
    log_path = state.scene_dir / "scene.log"
    if not log_path.exists():
        return
    try:
        with open(log_path, "rb") as f:
            f.seek(state.log_offset)
            chunk = f.read()
            state.log_offset = f.tell()
        if not chunk:
            return
        for line in chunk.decode("utf-8", errors="replace").split("\n"):
            line = line.rstrip()
            if not line:
                continue
            state.log_lines.append(line)
            ts = _log_hms(line)
            m = _RE_TOOL_CALLED.search(line)
            if m:
                state.activity.append((ts, "TOOL", m.group(1)))
                continue
            m = _RE_HTTP.search(line)
            if m:
                status = m.group(1)
                kind = "HTTP_OK" if status == "200" else "HTTP_ERR"
                state.activity.append((ts, kind, f"POST → {status}"))
                continue
            m = _RE_ERROR.search(line)
            if m:
                state.activity.append((ts, "ERROR", m.group(1)[:120]))
    except OSError:
        pass


def _tail_jsonl(path: Path, offset_attr: str, target: list, state: AppState) -> None:
    if not path.exists():
        return
    try:
        with open(path, "rb") as f:
            f.seek(getattr(state, offset_attr))
            chunk = f.read()
            setattr(state, offset_attr, f.tell())
        if not chunk:
            return
        for line in chunk.decode("utf-8", errors="replace").split("\n"):
            line = line.strip()
            if line:
                try:
                    target.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass


def _read_events(state: AppState) -> None:
    prev_len = len(state.events)
    _tail_jsonl(
        state.scene_dir / "run_analysis" / "events.jsonl",
        "events_offset",
        state.events,
        state,
    )
    for ev in state.events[prev_len:]:
        ts = (ev.get("timestamp", "") or "")[:19].replace("T", " ")[11:]
        etype = ev.get("event_type", "")
        sname = ev.get("stage_name", "")
        if etype == "stage_started":
            state.activity.append((ts, "STAGE", f"{sname} started"))
        elif etype == "stage_completed":
            dur = ev.get("duration_s")
            suffix = f" ({dur:.1f}s)" if dur else ""
            state.activity.append((ts, "STAGE", f"{sname} completed{suffix}"))


def _read_sqlite(state: AppState) -> None:
    for db_name in ("designer.db", "critic.db"):
        db_path = state.scene_dir / db_name
        if not db_path.exists():
            continue
        if db_name not in state.db_states:
            state.db_states[db_name] = DBState(path=db_path)
        ds = state.db_states[db_name]
        try:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro",
                uri=True,
                timeout=0.1,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, message_data FROM agent_messages WHERE id > ? ORDER BY id LIMIT 50",
                (ds.last_id,),
            ).fetchall()
            conn.close()
            for row in rows:
                ds.last_id = row["id"]
                try:
                    md = json.loads(row["message_data"])
                except Exception:
                    md = {"type": "raw", "content": str(row["message_data"])[:120]}
                ds.messages.append(md)
            if len(ds.messages) > 30:
                ds.messages = ds.messages[-30:]
        except (sqlite3.OperationalError, OSError):
            pass


def refresh_state(state: AppState) -> None:
    _tail_log(state)
    _tail_jsonl(
        state.scene_dir / "run_analysis" / "usage_records.jsonl",
        "records_offset",
        state.records,
        state,
    )
    _read_events(state)
    _read_sqlite(state)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    s = ts_str.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _secs_ago(ts_str: str) -> float | None:
    dt = _parse_ts(ts_str)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds()


def _fmt_elapsed(secs: float) -> str:
    secs = int(secs)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# Pipeline stage order (from scenesmith/experiments/indoor_scene_generation.py:81-89)
_PIPELINE_STAGES = [
    "task_analysis",
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
    "reward_spec",
    "morphology_task",
]


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------
def _active_stage(state: AppState) -> str:
    for ev in reversed(state.events):
        if ev.get("status") == "running":
            return ev.get("stage_name", "—")
    if state.records:
        return state.records[-1].get("stage_name") or "—"
    return "—"


def _active_agent(state: AppState) -> str:
    for rec in reversed(state.records):
        label = rec.get("agent_label")
        if label:
            return label
        role = rec.get("agent_role")
        atype = rec.get("agent_type")
        if role:
            return f"{role.upper()} ({atype})" if atype else role.upper()
    return "—"


def _last_tool_from_log(state: AppState) -> str:
    for line in reversed(list(state.log_lines)):
        m = _RE_TOOL_CALLED.search(line)
        if m:
            return m.group(1)
    if state.records:
        tcs = state.records[-1].get("tool_calls_in_response") or []
        if tcs:
            return tcs[0]
    return "—"


def _current_stage_elapsed(state: AppState) -> float | None:
    """Seconds since the active stage started, from events.jsonl started_at."""
    for ev in reversed(state.events):
        if ev.get("status") == "running":
            dt = _parse_ts(ev.get("started_at") or ev.get("timestamp", ""))
            if dt:
                return (datetime.now(timezone.utc) - dt).total_seconds()
    return None


def _stage_statuses(state: AppState) -> dict[str, str]:
    """Returns {stage_name: 'done'|'running'} from events."""
    statuses: dict[str, str] = {}
    for ev in state.events:
        sname = ev.get("stage_name", "")
        if not sname:
            continue
        if ev.get("event_type") == "stage_completed":
            statuses[sname] = "done"
        elif ev.get("event_type") == "stage_started" and sname not in statuses:
            statuses[sname] = "running"
    return statuses


def _per_stage_stats(state: AppState) -> dict[str, dict]:
    """Aggregate {stage_name: {calls, in_tok, out_tok}} from usage_records."""
    stats: dict[str, dict] = {}
    for r in state.records:
        sname = r.get("stage_name") or "unknown"
        if sname not in stats:
            stats[sname] = {"calls": 0, "in_tok": 0, "out_tok": 0}
        stats[sname]["calls"] += 1
        stats[sname]["in_tok"] += r.get("input_tokens", 0)
        stats[sname]["out_tok"] += r.get("output_tokens", 0)
    return stats


def render_top_left(state: AppState) -> Panel:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold", min_width=16)
    grid.add_column()

    stage = _active_stage(state)
    agent = _active_agent(state)
    tool = _last_tool_from_log(state)

    last_ts = state.records[-1].get("timestamp", "") if state.records else ""
    secs = _secs_ago(last_ts)
    stage_elapsed = _current_stage_elapsed(state)

    if secs is not None:
        ago_style = "bold red" if secs > 30 else "green"
        ago_text = Text(f"{secs:.0f}s ago", style=ago_style)
        inflight = (
            Text(f"~{secs:.0f}s in flight", style="yellow")
            if secs > 10
            else Text("—", style="dim")
        )
    else:
        ago_text = Text("no data yet", style="dim")
        inflight = Text("—", style="dim")

    stage_dur_text = (
        Text(_fmt_elapsed(stage_elapsed), style="cyan")
        if stage_elapsed
        else Text("—", style="dim")
    )

    grid.add_row("Stage:", Text(stage, style="cyan bold"))
    grid.add_row("Stage elapsed:", stage_dur_text)
    grid.add_row("Agent:", Text(agent, style="magenta"))
    grid.add_row("Last tool:", Text(tool, style="cyan"))
    grid.add_row("Last LLM call:", ago_text)
    grid.add_row("In-flight est.:", inflight)

    return Panel(grid, title="[bold]Currently Running[/bold]", border_style="blue")


def render_top_right(state: AppState) -> Panel:
    records = state.records
    total_calls = len(records)
    total_in = sum(r.get("input_tokens", 0) for r in records)
    total_out = sum(r.get("output_tokens", 0) for r in records)
    total_tok = sum(r.get("total_tokens", 0) for r in records)

    now = datetime.now(timezone.utc)
    recent_count = sum(
        1
        for r in records
        if (dt := _parse_ts(r.get("timestamp", "")))
        and (now - dt).total_seconds() <= 30
    )
    rate = recent_count / 30.0

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold", min_width=18)
    summary.add_column(justify="right")
    summary.add_row("Total calls:", str(total_calls))
    summary.add_row("Input tokens:", f"{total_in:,}")
    summary.add_row("Output tokens:", f"{total_out:,}")
    summary.add_row("Total tokens:", f"{total_tok:,}")
    summary.add_row("Rate (30s):", f"{rate:.2f} calls/s")

    return Panel(summary, title="[bold]Token Stats[/bold]", border_style="green")


def _stage_durations(state: AppState) -> dict[str, float]:
    """Completed stage durations from events; running stage estimated from started_at."""
    durations: dict[str, float] = {}
    for ev in state.events:
        sname = ev.get("stage_name", "")
        if not sname:
            continue
        if ev.get("event_type") == "stage_completed":
            dur = ev.get("duration_s")
            if dur is not None:
                durations[sname] = float(dur)
        elif ev.get("event_type") == "stage_started" and sname not in durations:
            started = _parse_ts(ev.get("started_at") or ev.get("timestamp", ""))
            if started:
                durations[sname] = (
                    datetime.now(timezone.utc) - started
                ).total_seconds()
    return durations


def render_stage_progress(state: AppState) -> Panel:
    """Full-width table: one row per seen/active stage with token counts and duration."""
    stage_data = _per_stage_stats(state)
    statuses = _stage_statuses(state)
    durations = _stage_durations(state)

    known = [s for s in _PIPELINE_STAGES if s in stage_data or s in statuses]
    extra = [s for s in stage_data if s not in _PIPELINE_STAGES]
    all_stages = known + extra

    table = Table(
        show_header=True, header_style="bold dim", box=None, padding=(0, 1), expand=True
    )
    table.add_column("Stage", min_width=16)
    table.add_column("Status", width=10)
    table.add_column("Calls", justify="right", width=6)
    table.add_column("Input tok", justify="right", width=11)
    table.add_column("Output tok", justify="right", width=11)
    table.add_column("Duration", justify="right", width=10)

    for sname in all_stages:
        st = statuses.get(sname, "pending" if sname not in stage_data else "done")
        d = stage_data.get(sname, {"calls": 0, "in_tok": 0, "out_tok": 0})
        dur = durations.get(sname)

        if st == "running":
            status_text = Text("▶ running", style="bold cyan")
            row_style = "cyan"
        elif st == "done":
            status_text = Text("✓ done", style="dim green")
            row_style = "dim"
        else:
            status_text = Text("· pending", style="dim")
            row_style = "dim"

        dur_s = _fmt_elapsed(dur) if dur else "—"
        table.add_row(
            Text(sname, style=row_style),
            status_text,
            Text(str(d["calls"]) if d["calls"] else "—", style=row_style),
            Text(f"{d['in_tok']:,}" if d["in_tok"] else "—", style=row_style),
            Text(f"{d['out_tok']:,}" if d["out_tok"] else "—", style=row_style),
            Text(dur_s, style=row_style),
        )

    return Panel(table, title="[bold]Stage Progress[/bold]", border_style="blue")


_KIND_STYLE = {
    "TOOL": "cyan",
    "HTTP_OK": "green",
    "HTTP_ERR": "bold red",
    "ERROR": "bold red",
    "STAGE": "blue bold",
}


def render_bottom_activity(state: AppState) -> Panel:
    table = Table(
        show_header=True, header_style="bold dim", box=None, padding=(0, 1), expand=True
    )
    table.add_column("Time", width=8)
    table.add_column("Kind", width=10)
    table.add_column("Detail", overflow="fold")

    for ts, kind, text in list(state.activity)[-15:]:
        style = _KIND_STYLE.get(kind, "white")
        table.add_row(
            Text(ts, style="dim"),
            Text(f"[{kind}]", style=style),
            Text(text, style=style),
        )

    return Panel(table, title="[bold]Recent Activity[/bold]", border_style="yellow")


def render_db_dump(state: AppState) -> Panel:
    grid = Table.grid(padding=(0, 0))
    grid.add_column()

    if not state.db_states:
        grid.add_row(Text("No agent DBs found yet.", style="dim"))
    else:
        for db_name, ds in state.db_states.items():
            sub = Table(
                show_header=True,
                header_style="dim",
                box=None,
                padding=(0, 1),
                expand=True,
            )
            sub.add_column("Type", width=26)
            sub.add_column("Content", overflow="fold")
            for msg in ds.messages[-5:]:
                mtype = msg.get("type", "?")
                if mtype == "function_call":
                    label = f"function_call/{msg.get('name','')}"
                    content = str(msg.get("arguments", ""))[:100]
                elif mtype == "function_call_output":
                    label = "function_call_output"
                    out = msg.get("output", "")
                    content = (
                        str(out[0]) if isinstance(out, list) and out else str(out)
                    )[:100]
                elif mtype == "message":
                    role = msg.get("role", "?")
                    label = f"message/{role}"
                    raw = msg.get("content", "")
                    content = (
                        str(raw[0])[:100]
                        if isinstance(raw, list) and raw
                        else str(raw)[:100]
                    )
                elif mtype == "reasoning":
                    label = "reasoning"
                    content = str(msg.get("content", ""))[:100]
                else:
                    label = mtype
                    content = str(msg)[:100]
                sub.add_row(Text(label, style="magenta"), Text(content))
            grid.add_row(
                Panel(
                    sub,
                    title=f"[bold]{db_name}[/bold] ({ds.last_id} rows seen)",
                    border_style="magenta",
                )
            )

    return Panel(
        grid,
        title="[bold]DB Dump[/bold] — press any key to dismiss",
        border_style="magenta",
    )


def render_status_bar(state: AppState) -> Text:
    elapsed = _fmt_elapsed(time.monotonic() - state.start_time)
    last_ts_disp = "—"
    stuck = False
    if state.records:
        raw = state.records[-1].get("timestamp", "")
        last_ts_disp = raw[11:19] if len(raw) >= 19 else raw
        secs = _secs_ago(raw)
        stuck = secs is not None and secs > 60

    bar = Text()
    bar.append(
        f" Run: {state.run_path.name}  Scene: {state.scene_dir.name}"
        f"  Elapsed: {elapsed}  Last record: {last_ts_disp}",
        style="dim",
    )
    if stuck:
        bar.append("  [STUCK]", style="bold red")
    bar.append("    [q] quit  [i] db dump  [o] open PNG", style="dim")
    return bar


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def make_renderable(state: AppState) -> Group:
    """Build the full renderable for one display tick."""
    # Side-by-side top panels via a two-column grid
    top_grid = Table.grid(expand=True, padding=0)
    top_grid.add_column(ratio=1)
    top_grid.add_column(ratio=1)
    top_grid.add_row(render_top_left(state), render_top_right(state))

    bottom = (
        render_db_dump(state) if state.show_db_dump else render_bottom_activity(state)
    )

    return Group(
        top_grid, render_stage_progress(state), bottom, render_status_bar(state)
    )


# ---------------------------------------------------------------------------
# Key handler
# ---------------------------------------------------------------------------
def handle_key(state: AppState) -> None:
    key, state.key = state.key, ""
    if not key:
        return
    if key == "q":
        state.quit = True
    elif key == "i":
        state.show_db_dump = not state.show_db_dump
    elif key == "o":
        pngs = sorted(
            glob.glob(
                str(state.scene_dir / "floor_plans" / "**" / "*.png"), recursive=True
            ),
            key=os.path.getmtime,
            reverse=True,
        )
        if pngs:
            subprocess.Popen(
                ["xdg-open", pngs[0]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif state.show_db_dump:
        state.show_db_dump = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time TUI monitor for TaskSmith pipeline runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/watch_agents.py\n"
            "  uv run python scripts/watch_agents.py --run outputs/2026-05-18/my_run\n\n"
            "Keys: q=quit  i=db dump/dismiss  o=open latest floor-plan PNG"
        ),
    )
    parser.add_argument(
        "--run", metavar="PATH", help="Path to a specific run directory"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render one frame to stdout and exit (for CI / screenshot capture)",
    )
    args = parser.parse_args()

    run_path, scene_dir = discover_run(args.run)
    state = AppState(run_path=run_path, scene_dir=scene_dir)

    refresh_state(state)

    if args.once:
        console = Console(force_terminal=True, width=120)
        console.print(make_renderable(state))
        return

    key_thread = threading.Thread(target=_key_reader, args=(state,), daemon=True)
    key_thread.start()

    console = Console()
    try:
        with Live(
            make_renderable(state), console=console, refresh_per_second=1
        ) as live:
            while not state.quit:
                t0 = time.monotonic()
                refresh_state(state)
                handle_key(state)
                live.update(make_renderable(state))
                time.sleep(max(0.0, 1.0 - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        state.quit = True
        if state._old_term is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, state._old_term
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()
