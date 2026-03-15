#!/usr/bin/env python3
"""
compare_logs.py
---------------
Compares all VIO trajectory evaluation runs under ./logs/.
For each run, parses summary.txt and (if present) pipeline_params.yaml.
Prints a rich table ordered by ESKF ATE RMSE (best → worst).

Usage:
    python3 compare_logs.py [--logs-dir ./logs] [--sort eskf_ate|vio_ate]
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ── parsers ────────────────────────────────────────────────────────────────────

def parse_summary(path: Path) -> dict:
    """Extract key metrics from a summary.txt file."""
    txt = path.read_text()
    m = {}

    def grab(label, pattern, flags=re.DOTALL):
        hit = re.search(pattern, txt, flags)
        m[label] = float(hit.group(1)) if hit else None

    # ESKF
    grab("eskf_ate_rmse",  r"ESKF vs Ground Truth.*?RMSE\s*:\s*([\d.]+)")
    grab("eskf_ate_mean",  r"ESKF vs Ground Truth.*?mean\s*:\s*([\d.]+)\s*cm")
    grab("eskf_rot_mean",  r"ESKF vs Ground Truth.*?ATE — rotation.*?mean\s*:\s*([\d.]+)")
    grab("eskf_rpe_1m",    r"ESKF vs Ground Truth.*?1\.0m\s+([\d.]+)%")

    # VIO
    grab("vio_ate_rmse",   r"VIO\s+vs Ground Truth.*?RMSE\s*:\s*([\d.]+)")
    grab("vio_ate_mean",   r"VIO\s+vs Ground Truth.*?mean\s*:\s*([\d.]+)\s*cm")
    grab("vio_rot_mean",   r"VIO\s+vs Ground Truth.*?ATE — rotation.*?mean\s*:\s*([\d.]+)")
    grab("vio_rpe_1m",     r"VIO\s+vs Ground Truth.*?1\.0m\s+([\d.]+)%")

    # VIO frame count
    hit = re.search(r"VIO\s+vs Ground Truth.*?1\.0m\s+[\d.]+%\s+[\d.]+%\s+[\d.]+%\s+(\d+)", txt, re.DOTALL)
    m["vio_frames"] = int(hit.group(1)) if hit else None

    # Run metadata
    hit = re.search(r"Generated\s*:\s*(\S+)", txt)
    m["generated"] = hit.group(1) if hit else ""

    hit = re.search(r"Duration\s*:\s*([\d.]+)", txt)
    m["duration_s"] = float(hit.group(1)) if hit else None

    return m


def parse_params(yaml_path: Path) -> dict:
    """Load pipeline_params.yaml and flatten into a dict of display strings."""
    if not HAS_YAML:
        # fallback: crude regex extraction of key values
        txt = yaml_path.read_text()
        out = {}
        for key in ["meas_pos_std", "meas_ang_std", "max_corners", "max_epipolar_err"]:
            hit = re.search(rf"{key}:\s*([\d.e+-]+)", txt)
            out[key] = hit.group(1) if hit else "?"
        return out

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    eskf = data.get("eskf", {})
    imu  = data.get("imu",  {})
    feat = data.get("feature_tracking", {})
    vio  = data.get("vio",  {})

    return {
        "meas_pos_std":      eskf.get("meas_pos_std", "?"),
        "meas_ang_std":      eskf.get("meas_ang_std", "?"),
        "init_pos_std":      eskf.get("init_pos_std", "?"),
        "init_vel_std":      eskf.get("init_vel_std", "?"),
        "init_att_std":      eskf.get("init_att_std", "?"),
        "init_ba_std":       eskf.get("init_ba_std",  "?"),
        "init_bg_std":       eskf.get("init_bg_std",  "?"),
        "init_duration":     imu.get("init_duration", "?"),
        "gyro_lpf_cutoff":   imu.get("gyro_lpf_cutoff", "?"),
        "accel_lpf_cutoff":  imu.get("accel_lpf_cutoff", "?"),
        "max_corners":       feat.get("max_corners", "?"),
        "quality_level":     feat.get("quality_level", "?"),
        "min_distance":      feat.get("min_distance", "?"),
        "win_size":          str(feat.get("win_size", "?")),
        "max_level":         feat.get("max_level", "?"),
        "max_epipolar_err":  feat.get("max_epipolar_err", "?"),
        "min_tracks":        vio.get("min_tracks", "?"),
        "min_inlier_ratio":  vio.get("min_inlier_ratio", "?"),
        "max_depth":         vio.get("max_depth", "?"),
        "max_translation":   vio.get("max_translation", "?"),
        "max_rotation_deg":  vio.get("max_rotation_deg", "?"),
    }


# ── formatting helpers ─────────────────────────────────────────────────────────

def fmt(val, decimals=1, unit=""):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{unit}"

def delta_str(eskf, vio):
    """Show ESKF − VIO difference with sign."""
    if eskf is None or vio is None:
        return ""
    d = eskf - vio
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.1f})"


# ── plain-text fallback ────────────────────────────────────────────────────────

def print_plain(runs: list):
    col_w = [max(len(r["name"]) for r in runs) + 2, 10, 10, 10, 10, 10, 10, 8, 12]
    headers = ["Run", "ESKF ATE↓", "VIO ATE↓", "ESKF rot↓", "VIO rot↓",
               "ESKF RPE1m↓", "VIO RPE1m↓", "Frames", "pos/ang std"]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  ".join("-" * w for w in col_w)

    print("\n" + row_fmt.format(*headers))
    print(sep)
    for r in runs:
        m = r["metrics"]
        p = r.get("params", {})
        pos_ang = (f"{p.get('meas_pos_std','?')}/{p.get('meas_ang_std','?')}"
                   if p else r["name"])
        row = [
            r["name"],
            fmt(m["eskf_ate_rmse"], 1, " cm"),
            fmt(m["vio_ate_rmse"],  1, " cm"),
            fmt(m["eskf_rot_mean"], 1, "°"),
            fmt(m["vio_rot_mean"],  1, "°"),
            fmt(m["eskf_rpe_1m"],   1, "%"),
            fmt(m["vio_rpe_1m"],    1, "%"),
            str(m["vio_frames"] or "?"),
            str(pos_ang),
        ]
        print(row_fmt.format(*row))
    print()


# ── rich table ─────────────────────────────────────────────────────────────────

PARAM_KEYS = [
    ("meas_pos_std",     "pos_std (m)"),
    ("meas_ang_std",     "ang_std (rad)"),
    ("max_corners",      "corners"),
    ("quality_level",    "quality"),
    ("min_distance",     "min_dist"),
    ("win_size",         "win_size"),
    ("max_level",        "pyr_lvl"),
    ("max_epipolar_err", "epipolar (px)"),
]


def print_rich(runs: list, show_params: bool):
    console = Console()

    # ── metrics table ─────────────────────────────────────────────────────────
    mtable = Table(
        title="[bold cyan]VIO Pipeline — Run Comparison[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    mtable.add_column("#",            justify="center", style="dim", width=3)
    mtable.add_column("Run",          min_width=20, no_wrap=False)
    mtable.add_column("ESKF ATE↓\n(cm)",   justify="right", style="green")
    mtable.add_column("ESKF rot↓\n(°)",    justify="right", style="green")
    mtable.add_column("ESKF RPE↓\n1m (%)", justify="right", style="green")
    mtable.add_column("VIO ATE↓\n(cm)",    justify="right", style="yellow")
    mtable.add_column("VIO rot↓\n(°)",     justify="right", style="yellow")
    mtable.add_column("VIO RPE↓\n1m (%)",  justify="right", style="yellow")
    mtable.add_column("Frames",  justify="right")
    mtable.add_column("Dur\n(s)", justify="right", style="dim")

    best_eskf = runs[0]["metrics"]["eskf_ate_rmse"] if runs else None

    for rank, r in enumerate(runs, 1):
        m = r["metrics"]
        is_best = (m["eskf_ate_rmse"] == best_eskf)

        eskf_ate_str = fmt(m["eskf_ate_rmse"], 1)
        if is_best:
            eskf_ate_str = f"[bold green]{eskf_ate_str} ★[/bold green]"
        delta = delta_str(m["eskf_ate_rmse"], m["vio_ate_rmse"])

        mtable.add_row(
            str(rank),
            r["name"],
            f"{eskf_ate_str}\n[dim]{delta}[/dim]",
            fmt(m["eskf_rot_mean"], 1),
            fmt(m["eskf_rpe_1m"],   1),
            fmt(m["vio_ate_rmse"],  1),
            fmt(m["vio_rot_mean"],  1),
            fmt(m["vio_rpe_1m"],    1),
            str(m["vio_frames"]) if m["vio_frames"] else "?",
            fmt(m["duration_s"],    1),
            style="bold" if is_best else "",
        )

    console.print()
    console.print(mtable)

    # ── params table (only runs that have a saved snapshot) ───────────────────
    if not show_params:
        console.print()
        return

    ptable = Table(
        title="[bold cyan]Parameters (runs with saved pipeline_params.yaml)[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold blue",
    )
    ptable.add_column("#",       justify="center", style="dim", width=3)
    ptable.add_column("Run",     min_width=20, no_wrap=False)
    for _, header in PARAM_KEYS:
        ptable.add_column(header, justify="right", style="cyan")

    for rank, r in enumerate(runs, 1):
        p = r.get("params")
        if p is None:
            continue
        ptable.add_row(
            str(rank),
            r["name"],
            *[str(p.get(key, "—")) for key, _ in PARAM_KEYS],
        )

    console.print()
    console.print(ptable)
    console.print()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="./logs",
                        help="Directory containing run sub-folders (default: ./logs)")
    parser.add_argument("--sort", choices=["eskf_ate", "vio_ate"], default="eskf_ate",
                        help="Sort by ESKF or VIO ATE RMSE (default: eskf_ate)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"ERROR: logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    # ── collect runs ──────────────────────────────────────────────────────────
    runs = []
    for sub in sorted(logs_dir.iterdir()):
        summary = sub / "summary.txt"
        if not sub.is_dir() or not summary.exists():
            continue

        metrics = parse_summary(summary)

        # only load params that were saved alongside this specific run
        run_yaml = sub / "pipeline_params.yaml"
        params = parse_params(run_yaml) if run_yaml.exists() else None

        runs.append({"name": sub.name, "metrics": metrics, "params": params})

    if not runs:
        print("No runs found. Check --logs-dir.", file=sys.stderr)
        sys.exit(1)

    # ── sort ──────────────────────────────────────────────────────────────────
    sort_key = "eskf_ate_rmse" if args.sort == "eskf_ate" else "vio_ate_rmse"
    runs.sort(key=lambda r: (r["metrics"][sort_key] is None,
                              r["metrics"][sort_key] or 1e9))

    # ── render ────────────────────────────────────────────────────────────────
    # show param columns only if at least one run has a saved params snapshot
    show_params = any(r["params"] is not None for r in runs)

    if HAS_RICH:
        print_rich(runs, show_params=show_params)
    else:
        print_plain(runs)

    if not HAS_RICH:
        print("Tip: pip install rich  — for a prettier table")


if __name__ == "__main__":
    main()
