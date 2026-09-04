#!/usr/bin/env python3
"""Adapt a dashboard template so it works against the PTD stack.

Upstream dashboards (vllm, sglang) are written for whatever datasource the author
had, and offline replay adds two more mismatches. All of it is fixed here rather
than by hand-editing the JSON, so upstream files stay pristine:

  1. every datasource reference is rebound to the local uid
  2. a per-panel interval floor is injected, because on a few-minute offline span
     $__interval / $__rate_interval resolve below the scrape interval and leave
     rate() with fewer than two points per window
  3. the time range is pinned to the replayed data, else it opens at "last 6 hours",
     and auto-refresh is switched off since replayed data never changes
"""

import json
from datetime import datetime, timezone


def _panels(dashboard: dict):
    """Every panel, including ones nested inside collapsed rows."""
    stack = list(dashboard.get("panels", []))
    while stack:
        panel = stack.pop()
        stack += panel.get("panels") or []
        yield panel


def _rebind(node, uid: str) -> None:
    """Point datasource references at `uid`, leaving the built-in Grafana one alone."""
    if isinstance(node, dict):
        ds = node.get("datasource")
        if isinstance(ds, dict) and ds.get("uid") not in (None, "-- Grafana --"):
            ds["uid"] = uid
        elif isinstance(ds, str) and ds != "-- Grafana --":
            node["datasource"] = {"type": "prometheus", "uid": uid}
        for value in node.values():
            _rebind(value, uid)
    elif isinstance(node, list):
        for value in node:
            _rebind(value, uid)


def _iso(ms: int) -> str:
    return (datetime.fromtimestamp(ms / 1000, timezone.utc)
            .isoformat(timespec="milliseconds").replace("+00:00", "Z"))


def render(template, output, datasource_uid="ptd", time_range=None,
           min_interval=None, kv_bytes=167_772_160) -> str:
    """Write the adapted dashboard to `output`; returns its uid."""
    dashboard = json.loads(template.read_text())
    _rebind(dashboard, datasource_uid)

    for var in dashboard.get("templating", {}).get("list", []):
        if var.get("type") == "datasource":
            var["current"] = {"text": "Prometheus", "value": datasource_uid}
        if isinstance(var.get("datasource"), dict):
            var["datasource"]["uid"] = datasource_uid
        # Only the PTD dashboards declare this one.
        if var.get("name") == "kv_bytes_per_request":
            value = str(kv_bytes)
            var.update(query=value,
                       current={"selected": True, "text": value, "value": value},
                       options=[{"selected": True, "text": value, "value": value}])

    if min_interval:
        dashboard["refresh"] = ""
        for panel in _panels(dashboard):
            if panel.get("type") != "row":
                panel["interval"] = min_interval

    if time_range:
        start, end = time_range
        dashboard["time"] = {"from": _iso(start - 15_000), "to": _iso(end + 15_000)}
        print(f"  time range {dashboard['time']['from'][:19]}"
              f" .. {dashboard['time']['to'][:19]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dashboard, ensure_ascii=False, indent=1) + "\n")
    return dashboard.get("uid", "")
