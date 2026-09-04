#!/usr/bin/env python3
"""Prepare replay data and Grafana provisioning under build/."""

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prom2openmetrics import bounds, convert
from render_dashboard import render

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

DASHBOARDS = {
    "ptd-vllm":   "dashboard/ptd/vllm-ptd.json",
    "ptd-sglang": "dashboard/ptd/sglang-ptd.json",
}


def _container_path(name: str, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return (ROOT / path).resolve()
    if path.exists():
        return path

    if host_root_value := os.environ.get("HOST_PTD_ROOT"):
        for item in host_root_value.split(","):
            host_root = Path(os.path.abspath(item.strip()))
            try:
                return ROOT / path.relative_to(host_root)
            except ValueError:
                continue
    return path


def absolute_log_paths(name: str, default: str | None = None) -> list[Path]:
    value = os.environ.get(name, default) or ""
    return [_container_path(name, item.strip())
            for item in value.split(",") if item.strip()]


def absolute_log_path(name: str, default: str | None = None) -> Path | None:
    paths = absolute_log_paths(name, default)
    if len(paths) > 1:
        sys.exit(f"{name} accepts one path, got {len(paths)}")
    return paths[0] if paths else None


def _labelled(paths: list[Path], role: str) -> list[tuple[Path, str]]:
    if len(paths) == 1:
        return [(paths[0], role)]
    return [(path, f"{role}-{index}") for index, path in enumerate(paths, 1)]


def _replay_signature(inputs: list[tuple[Path, str]]) -> str:
    items = []
    for path, label in inputs:
        stat = path.stat()
        items.append({
            "label": label,
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return json.dumps(items, sort_keys=True)


def _pause_prometheus_for_rebuild() -> None:
    try:
        request = urllib.request.Request(
            "http://prometheus:9090/-/quit", data=b"", method="POST")
        urllib.request.urlopen(request, timeout=1).close()
    except (OSError, urllib.error.URLError):
        return

    for _ in range(50):
        try:
            urllib.request.urlopen("http://prometheus:9090/-/ready", timeout=0.1).close()
        except (OSError, urllib.error.URLError):
            return
        time.sleep(0.1)
    sys.exit("Prometheus did not stop before TSDB rebuild")


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _detect_dashboard(om: Path) -> str:
    found = set()
    with om.open(errors="replace") as source:
        for line in source:
            if "sglang:" in line:
                found.add("ptd-sglang")
            if "vllm:" in line:
                found.add("ptd-vllm")
            if len(found) > 1:
                break
    if len(found) != 1:
        sys.exit("cannot detect dashboard; set DASH=ptd-sglang or DASH=ptd-vllm")
    return found.pop()


def main() -> None:
    dash = os.environ.get("DASH", "").strip()
    if dash and dash not in DASHBOARDS:
        sys.exit(f"unknown DASH={dash!r}, pick one of: {' | '.join(DASHBOARDS)}")

    gf = BUILD / "grafana"
    shutil.rmtree(gf, ignore_errors=True)
    for sub in ("dashboards", "provisioning/dashboards", "provisioning/datasources"):
        (gf / sub).mkdir(parents=True)

    targets = [t.strip() for t in os.environ.get("TARGETS", "").split(",") if t.strip()]
    time_range = None

    if targets:
        dash = dash or "ptd-sglang"
        (BUILD / "prometheus.yml").write_text(
            "global:\n"
            "  scrape_interval: 5s\n"
            "scrape_configs:\n"
            "  - job_name: ptd\n"
            "    static_configs:\n"
            "      - targets:\n"
            + "".join(f"          - '{t}'\n" for t in targets)
        )
        print(f"scraping: {', '.join(targets)}")
    else:
        om = BUILD / "om.txt"
        om.parent.mkdir(parents=True, exist_ok=True)
        src = absolute_log_path("OPENMETRICS")
        prefills = absolute_log_paths("PREFILL")
        decodes = absolute_log_paths("DECODE")
        inputs = None

        if src:
            if not src.is_file():
                sys.exit(f"no such file: {src}")
            inputs = [(src, "openmetrics")]
        elif prefills or decodes:
            if not prefills or not decodes:
                sys.exit("PREFILL and DECODE must both be set")
            inputs = _labelled(prefills, "prefill") + _labelled(decodes, "decode")
            for f, _ in inputs:
                if not f.is_file():
                    sys.exit(f"no such file: {f}\npaths must be inside the PTD directory")

        tsdb = BUILD / "tsdb"
        signature_file = BUILD / "replay-input.json"
        rebuild_marker = BUILD / "rebuild-tsdb"

        if inputs is None:
            if not (om.is_file() and tsdb.is_dir() and any(tsdb.iterdir())):
                sys.exit("set PREFILL and DECODE, or set OPENMETRICS")
            time_range = bounds(om)
            print("reusing the last replay")
        else:
            signature = _replay_signature(inputs)
            reusable = (
                signature_file.is_file()
                and signature_file.read_text() == signature
                and om.is_file()
                and tsdb.is_dir()
                and any(tsdb.iterdir())
                and not rebuild_marker.exists()
            )
            if reusable:
                time_range = bounds(om)
                print("reusing the last replay")
            else:
                rebuild_marker.write_text("rebuild required\n")
                _pause_prometheus_for_rebuild()
                _clear_directory(tsdb)
                if src:
                    shutil.copyfile(src, om)
                    time_range = bounds(om)
                else:
                    time_range = convert(inputs, om)
                signature_file.write_text(signature)

        dash = dash or _detect_dashboard(om)

    uid = render(
        template=ROOT / DASHBOARDS[dash],
        output=gf / "dashboards/current.json",
        datasource_uid="ptd",
        time_range=time_range,
        min_interval=None if targets else "30s",
        kv_bytes=int(os.environ.get("KV_BYTES", 167_772_160)),
    )

    url = "http://127.0.0.1:9090" if targets else "http://prometheus:9090"
    (gf / "provisioning/datasources/prom.yaml").write_text(
        "apiVersion: 1\n"
        "datasources:\n"
        "  - name: Prometheus\n"
        "    type: prometheus\n"
        "    uid: ptd\n"
        f"    url: {url}\n"
        "    isDefault: true\n"
    )
    (gf / "provisioning/dashboards/dash.yaml").write_text(
        "apiVersion: 1\n"
        "providers:\n"
        "  - name: ptd\n"
        "    type: file\n"
        "    options:\n"
        "      path: /var/lib/grafana/dashboards\n"
    )
    print()
    print(f"  Grafana     http://localhost:{os.environ.get('GF_PORT', '3000')}")
    print(f"  Prometheus  http://localhost:{os.environ.get('PROM_PORT', '9090')}")


if __name__ == "__main__":
    main()
