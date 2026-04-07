#!/usr/bin/env python3
"""
compare_logs.py
---------------
Compares all VIO trajectory evaluation runs under ./logs/.
For each run, parses summary.txt and (if present) pipeline_params.yaml.
Detects whether each run used ESKF, FGO, or both, and labels it accordingly.
Prints a rich table ordered by backend ATE RMSE (best → worst).

Usage:
    python3 compare_logs.py [--logs-dir ./logs] [--sort backend_ate|vio_ate|fgo_ate|eskf_ate]
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

    def grab(label, pattern):
        hit = re.search(pattern, txt, re.DOTALL)
        m[label] = float(hit.group(1)) if hit else None

    # ESKF block
    grab("eskf_ate_rmse",  r"ESKF vs Ground Truth.*?RMSE\s*:\s*([\d.]+)")
    grab("eskf_ate_mean",  r"ESKF vs Ground Truth.*?mean\s*:\s*([\d.]+)\s*cm")
    grab("eskf_rot_mean",  r"ESKF vs Ground Truth.*?ATE — rotation.*?mean\s*:\s*([\d.]+)")
    grab("eskf_rpe_1m",    r"ESKF vs Ground Truth.*?1\.0m\s+([\d.]+)%")

    # FGO block
    grab("fgo_ate_rmse",   r"FGO vs Ground Truth.*?RMSE\s*:\s*([\d.]+)")
    grab("fgo_ate_mean",   r"FGO vs Ground Truth.*?mean\s*:\s*([\d.]+)\s*cm")
    grab("fgo_rot_mean",   r"FGO vs Ground Truth.*?ATE — rotation.*?mean\s*:\s*([\d.]+)")
    grab("fgo_rpe_1m",     r"FGO vs Ground Truth.*?1\.0m\s+([\d.]+)%")

    # VIO block
    grab("vio_ate_rmse",   r"VIO\s+vs Ground Truth.*?RMSE\s*:\s*([\d.]+)")
    grab("vio_ate_mean",   r"VIO\s+vs Ground Truth.*?mean\s*:\s*([\d.]+)\s*cm")
    grab("vio_rot_mean",   r"VIO\s+vs Ground Truth.*?ATE — rotation.*?mean\s*:\s*([\d.]+)")
    grab("vio_rpe_1m",     r"VIO\s+vs Ground Truth.*?1\.0m\s+([\d.]+)%")

    # Detect run type from Backend(s) line in header, or infer from available data
    hit = re.search(r"Backend\(s\)\s*:\s*(.+)", txt)
    if hit:
        m["run_type"] = hit.group(1).strip()
    elif m.get("fgo_ate_rmse") is not None and m.get("eskf_ate_rmse") is not None:
        m["run_type"] = "ESKF + FGO"
    elif m.get("fgo_ate_rmse") is not None:
        m["run_type"] = "FGO"
    elif m.get("eskf_ate_rmse") is not None:
        m["run_type"] = "ESKF"
    else:
        m["run_type"] = "?"

    # Run metadata
    hit = re.search(r"Generated\s*:\s*(\S+)", txt)
    m["generated"] = hit.group(1) if hit else ""

    hit = re.search(r"Duration\s*:\s*([\d.]+)", txt)
    m["duration_s"] = float(hit.group(1)) if hit else None

    return m


def parse_params(yaml_path: Path) -> dict:
    """Load pipeline_params.yaml and flatten into a dict of display strings."""
    if not HAS_YAML:
        txt = yaml_path.read_text()
        out = {}
        for key in ["meas_pos_std", "meas_ang_std", "max_corners",
                    "max_epipolar_err", "window_size", "preint_min_rot_std"]:
            hit = re.search(rf"{key}:\s*([\d.e+-]+)", txt)
            out[key] = hit.group(1) if hit else "?"
        return out

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    eskf = data.get("eskf", {})
    imu  = data.get("imu",  {})
    feat = data.get("feature_tracking", {})
    vio  = data.get("vio",  {})
    fgo  = data.get("fgo",  {})

    return {
        # ESKF params
        "meas_pos_std":        eskf.get("meas_pos_std", "?"),
        "meas_ang_std":        eskf.get("meas_ang_std", "?"),
        "init_pos_std":        eskf.get("init_pos_std", "?"),
        "init_vel_std":        eskf.get("init_vel_std", "?"),
        "init_att_std":        eskf.get("init_att_std", "?"),
        "init_ba_std":         eskf.get("init_ba_std",  "?"),
        "init_bg_std":         eskf.get("init_bg_std",  "?"),
        # IMU
        "init_duration":       imu.get("init_duration", "?"),
        "gyro_lpf_cutoff":     imu.get("gyro_lpf_cutoff", "?"),
        "accel_lpf_cutoff":    imu.get("accel_lpf_cutoff", "?"),
        # Feature tracking
        "max_corners":         feat.get("max_corners", "?"),
        "quality_level":       feat.get("quality_level", "?"),
        "min_distance":        feat.get("min_distance", "?"),
        "win_size":            str(feat.get("win_size", "?")),
        "max_level":           feat.get("max_level", "?"),
        "max_epipolar_err":    feat.get("max_epipolar_err", "?"),
        # VIO
        "min_tracks":          vio.get("min_tracks", "?"),
        "min_inlier_ratio":    vio.get("min_inlier_ratio", "?"),
        "max_depth":           vio.get("max_depth", "?"),
        "max_translation":     vio.get("max_translation", "?"),
        "max_rotation_deg":    vio.get("max_rotation_deg", "?"),
        # FGO
        "fgo_window_size":     fgo.get("window_size", "?"),
        "fgo_lm_iter":         fgo.get("lm_max_iter", "?"),
        "fgo_imu_scale":       fgo.get("imu_noise_scale", "?"),
        "preint_min_rot_std":  fgo.get("preint_min_rot_std", "?"),
        "preint_min_vel_std":  fgo.get("preint_min_vel_std", "?"),
        "preint_min_pos_std":  fgo.get("preint_min_pos_std", "?"),
    }


# ── formatting helpers ─────────────────────────────────────────────────────────

def fmt(val, decimals=1, unit=""):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{unit}"

def delta_str(backend_val, vio_val):
    """Show backend − VIO difference with sign."""
    if backend_val is None or vio_val is None:
        return ""
    d = backend_val - vio_val
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:.1f})"


def run_type_color(rt: str) -> str:
    """Map run type to a rich color tag."""
    if "FGO" in rt and "ESKF" in rt:
        return "bold magenta"
    if "FGO" in rt:
        return "bold green"
    if "ESKF" in rt:
        return "bold blue"
    return "dim"


# ── plain-text fallback ────────────────────────────────────────────────────────

def print_plain(runs: list):
    col_w = [max(len(r["name"]) for r in runs) + 2, 7, 10, 10, 10, 10, 10, 10, 12]
    headers = ["Run", "Type", "Bknd ATE↓", "VIO ATE↓",
               "Bknd rot↓", "VIO rot↓", "Bknd RPE↓", "VIO RPE↓", "pos/ang_std"]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  ".join("-" * w for w in col_w)

    print("\n" + row_fmt.format(*headers))
    print(sep)
    for r in runs:
        m = r["metrics"]
        p = r.get("params", {})
        rt = m.get("run_type", "?")

        # Primary backend metrics: prefer FGO if present, else ESKF
        if m.get("fgo_ate_rmse") is not None:
            bknd_ate = m["fgo_ate_rmse"]
            bknd_rot = m["fgo_rot_mean"]
            bknd_rpe = m["fgo_rpe_1m"]
        else:
            bknd_ate = m.get("eskf_ate_rmse")
            bknd_rot = m.get("eskf_rot_mean")
            bknd_rpe = m.get("eskf_rpe_1m")

        pos_ang = (f"{p.get('meas_pos_std','?')}/{p.get('meas_ang_std','?')}"
                   if p else "?")
        row = [
            r["name"], rt,
            fmt(bknd_ate, 1, " cm"),
            fmt(m["vio_ate_rmse"], 1, " cm"),
            fmt(bknd_rot, 1, "°"),
            fmt(m["vio_rot_mean"], 1, "°"),
            fmt(bknd_rpe, 1, "%"),
            fmt(m["vio_rpe_1m"], 1, "%"),
            str(pos_ang),
        ]
        print(row_fmt.format(*row))
    print()


# ── rich table ─────────────────────────────────────────────────────────────────

ESKF_PARAM_KEYS = [
    ("meas_pos_std",     "pos_std"),
    ("meas_ang_std",     "ang_std"),
    ("max_corners",      "corners"),
    ("win_size",         "win_size"),
    ("max_epipolar_err", "epipolar"),
]

FGO_PARAM_KEYS = [
    ("fgo_window_size",    "win"),
    ("fgo_lm_iter",        "iter"),
    ("fgo_imu_scale",      "imu_scale"),
    ("preint_min_rot_std", "rot_floor"),
    ("preint_min_vel_std", "vel_floor"),
    ("preint_min_pos_std", "pos_floor"),
]


def print_rich(runs: list, show_params: bool):
    console = Console()

    has_fgo  = any(r["metrics"].get("fgo_ate_rmse")  is not None for r in runs)
    has_eskf = any(r["metrics"].get("eskf_ate_rmse") is not None for r in runs)

    # ── metrics table ─────────────────────────────────────────────────────────
    mtable = Table(
        title="[bold cyan]VIO Pipeline — Run Comparison[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    mtable.add_column("#",         justify="center", style="dim", width=3)
    mtable.add_column("Run",       min_width=20, no_wrap=False)
    mtable.add_column("Type",      justify="center", min_width=9)

    if has_eskf:
        mtable.add_column("ESKF ATE↓\n(cm)",   justify="right", style="cyan")
        mtable.add_column("ESKF rot↓\n(°)",    justify="right", style="cyan")
        mtable.add_column("ESKF RPE↓\n1m (%)", justify="right", style="cyan")

    if has_fgo:
        mtable.add_column("FGO ATE↓\n(cm)",    justify="right", style="green")
        mtable.add_column("FGO rot↓\n(°)",     justify="right", style="green")
        mtable.add_column("FGO RPE↓\n1m (%)",  justify="right", style="green")

    mtable.add_column("VIO ATE↓\n(cm)",    justify="right", style="yellow")
    mtable.add_column("VIO rot↓\n(°)",     justify="right", style="yellow")
    mtable.add_column("VIO RPE↓\n1m (%)",  justify="right", style="yellow")
    mtable.add_column("Dur\n(s)",  justify="right", style="dim")

    # Determine best run for each column
    best_eskf_ate = min(
        (r["metrics"]["eskf_ate_rmse"] for r in runs
         if r["metrics"].get("eskf_ate_rmse") is not None),
        default=None
    )
    best_fgo_ate = min(
        (r["metrics"]["fgo_ate_rmse"] for r in runs
         if r["metrics"].get("fgo_ate_rmse") is not None),
        default=None
    )

    for rank, r in enumerate(runs, 1):
        m = r["metrics"]
        rt = m.get("run_type", "?")
        rt_col = run_type_color(rt)

        row = [str(rank), r["name"], f"[{rt_col}]{rt}[/{rt_col}]"]

        if has_eskf:
            eskf_ate_s = fmt(m.get("eskf_ate_rmse"), 1)
            if m.get("eskf_ate_rmse") == best_eskf_ate and best_eskf_ate is not None:
                eskf_ate_s = f"[bold]{eskf_ate_s} ★[/bold]"
            delta = delta_str(m.get("eskf_ate_rmse"), m.get("vio_ate_rmse"))
            row += [
                f"{eskf_ate_s}\n[dim]{delta}[/dim]",
                fmt(m.get("eskf_rot_mean"), 1),
                fmt(m.get("eskf_rpe_1m"),   1),
            ]

        if has_fgo:
            fgo_ate_s = fmt(m.get("fgo_ate_rmse"), 1)
            if m.get("fgo_ate_rmse") == best_fgo_ate and best_fgo_ate is not None:
                fgo_ate_s = f"[bold]{fgo_ate_s} ★[/bold]"
            delta = delta_str(m.get("fgo_ate_rmse"), m.get("vio_ate_rmse"))
            row += [
                f"{fgo_ate_s}\n[dim]{delta}[/dim]",
                fmt(m.get("fgo_rot_mean"), 1),
                fmt(m.get("fgo_rpe_1m"),   1),
            ]

        row += [
            fmt(m.get("vio_ate_rmse"),  1),
            fmt(m.get("vio_rot_mean"),  1),
            fmt(m.get("vio_rpe_1m"),    1),
            fmt(m.get("duration_s"),    1),
        ]
        mtable.add_row(*row)

    console.print()
    console.print(mtable)

    if not show_params:
        console.print()
        return

    # ── params table ──────────────────────────────────────────────────────────
    param_keys = list(ESKF_PARAM_KEYS)
    if has_fgo:
        param_keys += FGO_PARAM_KEYS

    ptable = Table(
        title="[bold cyan]Parameters (runs with saved pipeline_params.yaml)[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold blue",
    )
    ptable.add_column("#",    justify="center", style="dim", width=3)
    ptable.add_column("Run",  min_width=20, no_wrap=False)
    ptable.add_column("Type", justify="center")
    for _, header in param_keys:
        ptable.add_column(header, justify="right", style="cyan")

    for rank, r in enumerate(runs, 1):
        p = r.get("params")
        if p is None:
            continue
        m = r["metrics"]
        rt = m.get("run_type", "?")
        ptable.add_row(
            str(rank),
            r["name"],
            f"[{run_type_color(rt)}]{rt}[/{run_type_color(rt)}]",
            *[str(p.get(key, "—")) for key, _ in param_keys],
        )

    console.print()
    console.print(ptable)
    console.print()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="./logs",
                        help="Directory containing run sub-folders (default: ./logs)")
    parser.add_argument(
        "--sort",
        choices=["backend_ate", "eskf_ate", "fgo_ate", "vio_ate"],
        default="backend_ate",
        help=(
            "Sort metric: 'backend_ate' uses FGO ATE if present else ESKF ATE, "
            "'eskf_ate'/'fgo_ate' force that backend, 'vio_ate' sorts by VIO. "
            "(default: backend_ate)"
        ),
    )
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
        run_yaml = sub / "pipeline_params.yaml"
        params = parse_params(run_yaml) if run_yaml.exists() else None

        runs.append({"name": sub.name, "metrics": metrics, "params": params})

    if not runs:
        print("No runs found. Check --logs-dir.", file=sys.stderr)
        sys.exit(1)

    # ── sort ──────────────────────────────────────────────────────────────────
    def sort_key(r):
        m = r["metrics"]
        if args.sort == "eskf_ate":
            v = m.get("eskf_ate_rmse")
        elif args.sort == "fgo_ate":
            v = m.get("fgo_ate_rmse")
        elif args.sort == "vio_ate":
            v = m.get("vio_ate_rmse")
        else:  # backend_ate: FGO if present, else ESKF
            v = m.get("fgo_ate_rmse") if m.get("fgo_ate_rmse") is not None \
                else m.get("eskf_ate_rmse")
        return (v is None, v or 1e9)

    runs.sort(key=sort_key)

    # ── render ────────────────────────────────────────────────────────────────
    show_params = any(r["params"] is not None for r in runs)

    if HAS_RICH:
        print_rich(runs, show_params=show_params)
    else:
        print_plain(runs)

    if not HAS_RICH:
        print("Tip: pip install rich  — for a prettier table")


if __name__ == "__main__":
    main()
