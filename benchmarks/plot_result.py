#!/usr/bin/env python3
"""
Parse perf_analyzer CSVs and produce benchmark plots.
Overlays InterLink, bare k8s, and bare HTCondor results on the same plot.

Usage:
    python plot_result.py --model higgsInteractionNet
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PerfResult:
    batch_size: int
    throughput_infer_per_sec: float
    avg_latency_ms: float
    avg_server_queue_ms: float = 0.0
    avg_server_compute_input_ms: float = 0.0
    avg_server_compute_infer_ms: float = 0.0
    avg_server_compute_output_ms: float = 0.0


def parse_csv(csv_path):
    bs = int(re.search(r"bs(\d+)", os.path.basename(csv_path)).group(1))

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    row = rows[-1]

    def col(substring):
        for k, v in row.items():
            if substring.lower() in k.lower():
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

    return PerfResult(
        batch_size=bs,
        throughput_infer_per_sec=col("inferences/second"),
        avg_latency_ms=col("avg latency") / 1000.0,
        avg_server_queue_ms=col("server queue") / 1000.0,
        avg_server_compute_input_ms=col("compute input") / 1000.0,
        avg_server_compute_infer_ms=col("compute infer") / 1000.0,
        avg_server_compute_output_ms=col("compute output") / 1000.0,
    )


def load_all(input_dir):
    results = []
    csv_files = sorted(Path(input_dir).glob("perf_bs*.csv"))
    print(f"  Found {len(csv_files)} CSV files in {input_dir}")

    for f in csv_files:
        try:
            r = parse_csv(str(f))
            if r:
                results.append(r)
                print(f"    {f.name}: bs={r.batch_size} "
                      f"throughput={r.throughput_infer_per_sec:.1f} infer/s "
                      f"latency={r.avg_latency_ms:.2f} ms")
        except Exception as e:
            print(f"    [WARN] Skipping {f.name}: {e}")

    results.sort(key=lambda r: r.batch_size)
    return results


def save_summary(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "batch_size", "throughput_infer_per_sec", "avg_latency_ms",
            "server_queue_ms", "compute_input_ms",
            "compute_infer_ms", "compute_output_ms",
        ])
        for r in results:
            w.writerow([
                r.batch_size,
                f"{r.throughput_infer_per_sec:.2f}",
                f"{r.avg_latency_ms:.3f}",
                f"{r.avg_server_queue_ms:.3f}",
                f"{r.avg_server_compute_input_ms:.3f}",
                f"{r.avg_server_compute_infer_ms:.3f}",
                f"{r.avg_server_compute_output_ms:.3f}",
            ])
    print(f"  Summary saved to {path}")


def plot_combined(curves, outdir, model):
    """Plot all curves on the same figure."""

    fig, ax = plt.subplots(figsize=(12, 7))

    for curve in curves:
        results = curve["results"]
        if not results:
            continue

        throughput = [r.throughput_infer_per_sec for r in results]
        latency = [r.avg_latency_ms for r in results]
        bs_labels = [r.batch_size for r in results]
        color = curve["color"]

        ax.plot(throughput, latency, "o-",
                color=color, linewidth=2, markersize=8,
                markeredgecolor=curve["marker_edge"],
                markerfacecolor=color,
                label=curve["label"])

        for i, bs in enumerate(bs_labels):
            ax.annotate(
                f"b={bs}",
                (throughput[i], latency[i]),
                textcoords="offset points",
                xytext=(8, 10),
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xlabel("Throughput [infer/s]", fontsize=14)
    ax.set_ylabel("Average batch latency [ms]", fontsize=14)
    ax.set_title(f"{model} — Avg Batch Latency vs Throughput", fontsize=16)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    path_png = os.path.join(outdir, "latency_vs_throughput.png")
    path_pdf = os.path.join(outdir, "latency_vs_throughput.pdf")
    fig.savefig(path_png, dpi=150)
    fig.savefig(path_pdf)
    print(f"Plot saved: {path_png}")
    print(f"Plot saved: {path_pdf}")
    plt.close(fig)


def plot_breakdown(results, outdir, model, label):
    """Stacked bar chart for server-side latency breakdown."""
    if not results:
        return

    bs_labels = [r.batch_size for r in results]
    x = np.arange(len(bs_labels))
    queue = [r.avg_server_queue_ms for r in results]
    comp_in = [r.avg_server_compute_input_ms for r in results]
    comp_inf = [r.avg_server_compute_infer_ms for r in results]
    comp_out = [r.avg_server_compute_output_ms for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, queue, label="Queue", color="tab:orange")
    ax.bar(x, comp_in, bottom=np.array(queue),
           label="Compute Input", color="tab:cyan")
    ax.bar(x, comp_inf, bottom=np.array(queue) + np.array(comp_in),
           label="Compute Infer", color="tab:green")
    ax.bar(x, comp_out,
           bottom=np.array(queue) + np.array(comp_in) + np.array(comp_inf),
           label="Compute Output", color="tab:purple")

    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Time (ms)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs_labels])
    ax.legend()
    ax.set_title(f"{model} — Server Latency Breakdown ({label})", fontsize=15)
    fig.tight_layout()

    safe_label = label.lower().replace(" ", "_").replace(":", "").replace("+", "_")
    path_png = os.path.join(outdir, f"server_breakdown_{safe_label}.png")
    path_pdf = os.path.join(outdir, f"server_breakdown_{safe_label}.pdf")
    fig.savefig(path_png, dpi=150)
    fig.savefig(path_pdf)
    print(f"Plot saved: {path_png}")
    print(f"Plot saved: {path_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interlink", default="results/benchmark_interlink/",
                        help="Directory with InterLink perf_bs*.csv files")
    parser.add_argument("--k8s", default="results/benchmark_k8s/",
                        help="Directory with bare k8s perf_bs*.csv files")
    parser.add_argument("--htc", default="results/benchmark_htc/",
                        help="Directory with bare HTCondor perf_bs*.csv files")
    parser.add_argument("--output", default="results/",
                        help="Directory for output plots")
    parser.add_argument("--model", default="higgsInteractionNet",
                        help="Model name (for plot titles)")
    args = parser.parse_args()

    # ── Load all datasets ─────────────────────────────────────────────
    print("Loading InterLink results...")
    results_interlink = load_all(args.interlink)

    print("\nLoading bare k8s results...")
    results_k8s = load_all(args.k8s)

    print("\nLoading bare HTCondor results...")
    results_htc = load_all(args.htc)

    if not results_interlink and not results_k8s and not results_htc:
        print("[ERROR] No data found in any directory.")
        exit(1)

    # ── Save summaries ────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    if results_interlink:
        save_summary(results_interlink,
                     os.path.join(args.interlink, "summary.csv"))
    if results_k8s:
        save_summary(results_k8s,
                     os.path.join(args.k8s, "summary.csv"))
    if results_htc:
        save_summary(results_htc,
                     os.path.join(args.htc, "summary.csv"))

    # ── Combined latency vs throughput plot ────────────────────────────
    print("\nGenerating plots...")

    curves = [
        {
            "results": results_interlink,
            "label": "InterLink: k8s+HTCondor",
            "color": "tab:red",
            "marker_edge": "darkred",
        },
        {
            "results": results_k8s,
            "label": "Bare k8s",
            "color": "tab:blue",
            "marker_edge": "darkblue",
        },
        {
            "results": results_htc,
            "label": "Bare HTCondor",
            "color": "tab:green",
            "marker_edge": "darkgreen",
        },
    ]

    plot_combined(curves, args.output, args.model)

    # ── Individual breakdown plots ────────────────────────────────────
    plot_breakdown(results_interlink, args.output, args.model,
                   "InterLink: k8s+HTCondor")
    plot_breakdown(results_k8s, args.output, args.model,
                   "Bare k8s")
    plot_breakdown(results_htc, args.output, args.model,
                   "Bare HTCondor")

    print("\nDone!")