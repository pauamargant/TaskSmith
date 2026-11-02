#!/usr/bin/env python3
"""Check RCP (EPFL LiteLLM proxy) budget and recent usage for your API key.

Reads ``RCP_BASE_URL`` and ``RCP_API_KEY`` from the environment (loading
``.env`` from the repo root if present). Uses only the LiteLLM proxy
endpoints that work with a non-admin key:

* ``/key/info`` for budget remaining
* ``/user/daily/activity`` for token / spend / request history

Usage::

    python scripts/check_rcp_usage.py                  # budget + last 7 days
    python scripts/check_rcp_usage.py --days 30        # last 30 days
    python scripts/check_rcp_usage.py --days 30 --top 20
    python scripts/check_rcp_usage.py --budget-only    # skip activity fetch
    python scripts/check_rcp_usage.py --json           # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env (no shell expansion)."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class HttpStatusError(Exception):
    def __init__(self, code: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {code} from {url}: {body}")
        self.code = code
        self.url = url
        self.body = body


def http_get(url: str, api_key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise HttpStatusError(e.code, url, body) from None


def fetch_key_info(root: str, key: str) -> dict:
    try:
        return http_get(f"{root}/key/info", key)
    except HttpStatusError as e:
        raise SystemExit(str(e)) from None


def fetch_activity(root: str, key: str, start: date, end: date) -> list[dict]:
    """Page through /user/daily/activity and return the raw daily entries."""
    results: list[dict] = []
    page = 1
    while True:
        params = urllib.parse.urlencode(
            {"start_date": start.isoformat(), "end_date": end.isoformat(), "page": page}
        )
        data = http_get(f"{root}/user/daily/activity?{params}", key)
        results.extend(data.get("results", []))
        meta = data.get("metadata", {}) or {}
        if not meta.get("has_more"):
            break
        page += 1
        if page > 200:
            break
    return results


def aggregate(entries: list[dict], key_hash: str | None = None) -> dict:
    """Aggregate daily entries. If ``key_hash`` is given, restrict to that key.

    ``/user/daily/activity`` reports across *all* keys owned by the user. To
    isolate a single key we read each day's ``breakdown.api_keys[<hash>]``
    subtree instead of the day's top-level metrics, and only sum models
    inside that subtree.
    """
    per_day: dict[str, dict] = {}
    per_model: dict[str, dict] = defaultdict(
        lambda: {"spend": 0.0, "tokens": 0, "requests": 0}
    )
    seen_day: set[str] = set()
    for entry in entries:
        d = entry["date"]
        breakdown = entry.get("breakdown", {}) or {}

        if key_hash:
            key_node = (breakdown.get("api_keys") or {}).get(key_hash)
            if not key_node:
                continue
            km = key_node["metrics"]
            if d not in seen_day:
                per_day[d] = {
                    "spend": km["spend"],
                    "tokens": km["total_tokens"],
                    "requests": km["api_requests"],
                    "failed": km["failed_requests"],
                    "ok": km["successful_requests"],
                }
                seen_day.add(d)
            models_node = (
                (key_node.get("metadata") or {}).get("models")
                or key_node.get("models")
                or {}
            )
        else:
            m = entry["metrics"]
            if d not in seen_day:
                per_day[d] = {
                    "spend": m["spend"],
                    "tokens": m["total_tokens"],
                    "requests": m["api_requests"],
                    "failed": m["failed_requests"],
                    "ok": m["successful_requests"],
                }
                seen_day.add(d)
            models_node = breakdown.get("models") or {}

        for model, mb in models_node.items():
            mm = mb.get("metrics") or mb
            per_model[model]["spend"] += mm.get("spend", 0.0)
            per_model[model]["tokens"] += mm.get("total_tokens", 0)
            per_model[model]["requests"] += mm.get("api_requests", 0)
    return {"per_day": per_day, "per_model": dict(per_model)}


def print_budget(info_payload: dict) -> None:
    info = info_payload.get("info", info_payload)
    spend = info.get("spend") or 0.0
    budget = info.get("max_budget")
    print("=" * 64)
    print("Key budget")
    print("=" * 64)
    print(f"  alias:        {info.get('key_alias', '-')}")
    print(f"  key_name:     {info.get('key_name', '-')}")
    print(f"  spend:        ${spend:.4f}")
    if budget:
        remaining = budget - spend
        pct = 100.0 * spend / budget if budget else 0.0
        print(f"  max_budget:   ${budget:.2f}")
        print(f"  remaining:    ${remaining:.2f}  ({pct:.2f}% used)")
    else:
        print("  max_budget:   (unlimited / unset)")
    for label in ("budget_reset_at", "expires", "tpm_limit", "rpm_limit"):
        if info.get(label) is not None:
            print(f"  {label}: {info[label]}")
    n_models = len(info.get("models", []) or [])
    if n_models:
        print(f"  models:       {n_models} allowed")


def print_activity(agg: dict, top: int, scope: str = "this key") -> None:
    per_day = agg["per_day"]
    per_model = agg["per_model"]
    if not per_day:
        print("\n(no activity in range)")
        return

    days = sorted(per_day)
    total_spend = sum(v["spend"] for v in per_day.values())
    total_tokens = sum(v["tokens"] for v in per_day.values())
    total_requests = sum(v["requests"] for v in per_day.values())
    total_failed = sum(v["failed"] for v in per_day.values())
    fail_pct = 100.0 * total_failed / total_requests if total_requests else 0.0
    avg_per_day = total_spend / len(per_day) if per_day else 0.0
    cost_per_mtok = 1e6 * total_spend / total_tokens if total_tokens else 0.0

    print()
    print("=" * 64)
    print(f"Activity ({scope})  {days[0]} -> {days[-1]}  ({len(per_day)} days)")
    print("=" * 64)
    print(f"  spend:        ${total_spend:,.2f}")
    print(f"  tokens:       {total_tokens:,}")
    print(
        f"  requests:     {total_requests:,}  ({total_failed:,} failed, {fail_pct:.1f}%)"
    )
    print(f"  avg $/day:    ${avg_per_day:.2f}")
    print(f"  $/1M tokens:  ${cost_per_mtok:.3f}")

    print()
    print(f"  {'date':<12} {'spend':>9} {'tokens':>14} {'reqs':>9} {'fail':>7}")
    for d in days:
        v = per_day[d]
        print(
            f"  {d:<12} ${v['spend']:>8.2f} {v['tokens']:>14,} "
            f"{v['requests']:>9,} {v['failed']:>7,}"
        )

    if per_model:
        print()
        print(f"Top models by spend (top {top})")
        print("-" * 64)
        print(f"  {'model':<48} {'spend':>9} {'tokens':>14}")
        ranked = sorted(per_model.items(), key=lambda x: -x[1]["spend"])[:top]
        for model, v in ranked:
            name = model if len(model) <= 47 else model[:44] + "..."
            print(f"  {name:<48} ${v['spend']:>8.2f} {v['tokens']:>14,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--days", type=int, default=7, help="Lookback window (default 7)."
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD (overrides --days).")
    parser.add_argument("--end", help="End date YYYY-MM-DD (defaults to today).")
    parser.add_argument("--top", type=int, default=10, help="Top-N models to show.")
    parser.add_argument(
        "--budget-only", action="store_true", help="Skip activity fetch."
    )
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Report activity for all keys on your user account (default: current key only).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    base_url = os.environ.get("RCP_BASE_URL")
    api_key = os.environ.get("RCP_API_KEY")
    if not base_url or not api_key:
        print(
            "error: RCP_BASE_URL and RCP_API_KEY must be set (in env or .env)",
            file=sys.stderr,
        )
        return 2
    root = base_url.rstrip("/").removesuffix("/v1")

    info = fetch_key_info(root, api_key)
    key_hash = info.get("key") or info.get("info", {}).get("key")

    activity = None
    scope = "this key"
    activity_error: str | None = None
    if not args.budget_only:
        end = date.fromisoformat(args.end) if args.end else date.today()
        start = (
            date.fromisoformat(args.start)
            if args.start
            else end - timedelta(days=args.days)
        )
        try:
            entries = fetch_activity(root, api_key, start, end)
        except HttpStatusError as e:
            if e.code in (401, 403):
                activity_error = (
                    f"HTTP {e.code} on /user/daily/activity: this key is not "
                    "authorized for user analytics (likely a service-account "
                    "key). Budget info above still applies. Use a user-bound "
                    "key or an admin key to see token/spend history."
                )
                entries = []
            else:
                raise SystemExit(str(e)) from None
        if entries:
            if args.all_keys or not key_hash:
                activity = aggregate(entries, key_hash=None)
                scope = "all keys on user account"
            else:
                activity = aggregate(entries, key_hash=key_hash)

    if args.json:
        payload = {
            "key_info": info.get("info", info),
            "activity": activity,
            "scope": scope,
            "activity_error": activity_error,
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print_budget(info)
    if activity is not None:
        print_activity(activity, top=args.top, scope=scope)
    elif activity_error:
        print()
        print(f"warning: {activity_error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
