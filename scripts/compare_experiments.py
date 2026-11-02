#!/usr/bin/env python3
"""Cross-run comparison CLI for TaskSmith ablations.

Thin wrapper around :mod:`scenesmith.experiment_tracking.compare`. The
first positional argument is the baseline by default; pass
``--baseline`` to choose another run.

Examples::

    python scripts/compare_experiments.py \
        outputs/2026-05-13/e0_base_22-41-07_599672 \
        outputs/2026-05-15/e1a_rounds1_02-36-36_805351 \
        outputs/2026-05-15/e1b_low_vision_22-55-33_213358

    python scripts/compare_experiments.py outputs/.../e0 outputs/.../e1e \
        --per-stage --per-tool --csv /tmp/e1_compare.csv
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenesmith.experiment_tracking.compare import compare_experiments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Experiment or scene directories to compare",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to use as baseline for deltas (defaults to first positional)",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Write CSV to this path")
    parser.add_argument("--per-stage", action="store_true")
    parser.add_argument("--per-tool", action="store_true")
    args = parser.parse_args()

    result = compare_experiments(args.paths, baseline=args.baseline)
    print(result.to_markdown(per_stage=args.per_stage, per_tool=args.per_tool))

    if args.csv:
        args.csv.write_text(result.to_csv(), encoding="utf-8")
        print(f"\nCSV written to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
