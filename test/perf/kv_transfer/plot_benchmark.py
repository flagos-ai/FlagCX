#!/usr/bin/env python3
"""
Plot KV Transfer Benchmark results.

Parses .log files (benchmark stdout) from the same directory and generates
a comparison figure with latency and bandwidth subplots.

Usage:
    python plot_benchmark.py                     # auto-discover *.log in script dir
    python plot_benchmark.py --log-dir=./results
    python plot_benchmark.py --show              # display interactively
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Regex patterns for parsing benchmark output
# ---------------------------------------------------------------------------

CONNECTOR_RE = re.compile(r"connector=(\w+)")
BACKEND_RE = re.compile(r"nixl-backend=(\w+)")
DATA_RE = re.compile(
    r"([\d.]+)\s*(KB|MB|GB)\s*\|\s*lat=\s*([\d.]+)\s*ms\s*\|\s*BW=\s*([\d.]+)\s*GB/s"
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _get_style(idx):
    return COLORS[idx % len(COLORS)], MARKERS[idx % len(MARKERS)]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_log(filepath):
    """Parse a single benchmark log file.

    Returns (label, data) where data is a list of
    {"size_mb": float, "lat_ms": float, "bw_gbs": float}.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Extract connector label
    m = CONNECTOR_RE.search(content)
    connector = m.group(1) if m else os.path.splitext(os.path.basename(filepath))[0]

    m = BACKEND_RE.search(content)
    backend = m.group(1).lower() if m else None

    if connector == "nixl" and backend:
        label = f"nixl_{backend}"
    else:
        label = connector

    # Extract data points
    data = []
    for m in DATA_RE.finditer(content):
        size_val = float(m.group(1))
        size_unit = m.group(2)
        lat_ms = float(m.group(3))
        bw_gbs = float(m.group(4))

        if size_unit == "GB":
            size_mb = size_val * 1024
        elif size_unit == "KB":
            size_mb = size_val / 1024
        else:
            size_mb = size_val
        data.append({"size_mb": size_mb, "lat_ms": lat_ms, "bw_gbs": bw_gbs})

    data.sort(key=lambda d: d["size_mb"])
    return label, data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def fmt_size(x, _):
    """Format size in MB to human-readable string."""
    if x >= 1024:
        return f"{x / 1024:.0f} GB"
    if x >= 1:
        return f"{x:.0f} MB"
    return f"{x * 1024:.0f} KB"


def plot(all_data, output_path, show=False):
    """Generate the comparison figure.

    all_data: dict of {label: [{"size_mb", "lat_ms", "bw_gbs"}, ...]}
    """
    fig, (ax_lat, ax_bw) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("KV Transfer Benchmark Comparison", fontsize=14)

    # Collect all unique sizes for explicit tick positions
    all_sizes = sorted(set(
        d["size_mb"] for data in all_data.values() for d in data
    ))

    for idx, (label, data) in enumerate(sorted(all_data.items())):
        color, marker = _get_style(idx)
        sizes = [d["size_mb"] for d in data]
        lats = [d["lat_ms"] for d in data]
        bws = [d["bw_gbs"] for d in data]

        ax_lat.plot(
            sizes, lats,
            marker=marker, label=label, color=color,
            linewidth=2, markersize=7,
        )
        ax_bw.plot(
            sizes, bws,
            marker=marker, label=label, color=color,
            linewidth=2, markersize=7,
        )

    # Latency subplot
    ax_lat.set_xlabel("Transfer Size")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Latency vs Transfer Size")
    ax_lat.set_xscale("log")
    ax_lat.set_yscale("log")
    ax_lat.set_xticks(all_sizes)
    ax_lat.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_size))
    ax_lat.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_lat.legend(fontsize=9)
    ax_lat.grid(True, linestyle="--", alpha=0.5)

    # Bandwidth subplot
    ax_bw.set_xlabel("Transfer Size")
    ax_bw.set_ylabel("Bandwidth (GB/s)")
    ax_bw.set_title("Bandwidth vs Transfer Size")
    ax_bw.set_xscale("log")
    ax_bw.set_xticks(all_sizes)
    ax_bw.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_size))
    ax_bw.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_bw.legend(fontsize=9)
    ax_bw.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {os.path.abspath(output_path)}")

    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot KV Transfer Benchmark results")
    parser.add_argument(
        "--log-dir", default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing .log files (default: script directory)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output image path (default: benchmark.png in log-dir)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the plot interactively",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    output_path = args.output or os.path.join(log_dir, "benchmark.png")

    # Auto-discover log files
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not log_files:
        sys.exit(f"No .log files found in {log_dir}")

    # Parse all logs
    all_data = {}
    for path in log_files:
        label, data = parse_log(path)
        if not data:
            print(f"  Skipping {os.path.basename(path)} (no data points found)")
            continue
        # Handle duplicate labels by appending filename
        if label in all_data:
            label = f"{label}_{os.path.splitext(os.path.basename(path))[0]}"
        all_data[label] = data
        print(f"  {label}: {len(data)} data points from {os.path.basename(path)}")

    if not all_data:
        sys.exit("No valid data found in any log file")

    plot(all_data, output_path, show=args.show)


if __name__ == "__main__":
    main()
