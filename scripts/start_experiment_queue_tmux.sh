#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-e1}
WORKDIR=${WORKDIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."}
COMMANDS_FILE=${COMMANDS_FILE:?Must set COMMANDS_FILE to a path containing experiment commands}
TIMEOUT_MIN=${TIMEOUT_MIN:-120}

# Optional: set externally (recommended)
#   export E0_BASE_PATH=outputs/....

DATE_DIR=${DATE_DIR:-$(date +%F)}
LOG_DIR=${LOG_DIR:-"$WORKDIR/outputs/$DATE_DIR"}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/${SESSION}_queue_tmux.log"}

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" >&2
  exit 2
fi

# Use -A so re-running this script is idempotent (attaches to existing session).
# Use bash -lc so env + PATH behave like an interactive shell.
TMUX_CMD=$(cat <<EOF
set -euo pipefail
cd "$WORKDIR"

# Make python output line-buffered in tmux logs.
export PYTHONUNBUFFERED=1

# Optional: show provider config (no secrets).
if [ -n "${PROVIDER_BASE_URL:-}" ]; then echo "[env] PROVIDER_BASE_URL=$PROVIDER_BASE_URL"; fi
if [ -n "${PROVIDER_MODEL:-}" ]; then echo "[env] PROVIDER_MODEL=$PROVIDER_MODEL"; fi
if [ -n "${E0_BASE_PATH:-}" ]; then echo "[env] E0_BASE_PATH=$E0_BASE_PATH"; fi

echo "[tmux] session=$SESSION";
echo "[tmux] workdir=$WORKDIR";
echo "[tmux] commands_file=$COMMANDS_FILE";
echo "[tmux] timeout_min=$TIMEOUT_MIN";
echo "[tmux] log_file=$LOG_FILE";

# Auto-restart the queue runner if it crashes (not if it exits cleanly).
# The queue runner itself continues past per-command failures/timeouts.
while true; do
  echo "\n===== $(date -Is) starting queue =====" | tee -a "$LOG_FILE"
  python scripts/run_experiment_queue.py \
    --timeout-min "$TIMEOUT_MIN" \
    --commands-file "$COMMANDS_FILE" \
    2>&1 | tee -a "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  echo "===== $(date -Is) queue exit rc=$rc =====\n" | tee -a "$LOG_FILE"

  if [ "$rc" -eq 0 ]; then
    break
  fi

  echo "[tmux] queue runner crashed (rc=$rc). restarting in 10s..." | tee -a "$LOG_FILE"
  sleep 10

done
EOF
)

tmux new-session -d -A -s "$SESSION" bash -lc "$TMUX_CMD"

echo "Started tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
