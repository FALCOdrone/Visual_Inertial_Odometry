#!/usr/bin/env python3
"""
evaluate_trajectory.py
======================
Offline scoring of ESKF vs VIO vs Ground Truth trajectories.

Reads pose CSVs written by debug_logger_node and computes standard VIO/SLAM
evaluation metrics:

  ATE (Absolute Trajectory Error)  — global accuracy after Umeyama SE3 alignment
  RPE (Relative Pose Error)        — local drift over fixed path-length segments

Output saved to logs/<YYYY-MM-DD_HH-MM-SS>/:
  summary.txt        Human-readable metric table
  trajectory.png     Top-down XY and side-view XZ comparison
  ate_over_time.png  Per-sample ATE vs elapsed time
  rpe_boxplot.png    RPE distribution at multiple segment lengths

Usage (run from project root):
    python evaluate_trajectory.py [--input tmp/] [--logs logs/]
"""

import argparse
import csv
import glob
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("WARNING: matplotlib not found — plots will be skipped.", file=sys.stderr)


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_traj(path: Path):
    """
    Load a pose_*.csv written by debug_logger_node.

    Returns
    -------
    ts   : (N,)   int64   nanosecond timestamps
    pos  : (N, 3) float64 [x, y, z] in metres
    quat : (N, 4) float64 [qx, qy, qz, qw]
    """
    ts, pos, quat = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                t = int(row["timestamp_ns"])
                p = [float(row["px"]), float(row["py"]), float(row["pz"])]
                q = [float(row["qx"]), float(row["qy"]),
                     float(row["qz"]), float(row["qw"])]
            except (ValueError, KeyError):
                continue
            ts.append(t)
            pos.append(p)
            quat.append(q)

    ts   = np.array(ts,   dtype=np.int64)
    pos  = np.array(pos,  dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)
    order = np.argsort(ts)
    return ts[order], pos[order], quat[order]


# ── Time helpers ───────────────────────────────────────────────────────────────

def trim(ts, pos, quat, t0, t1):
    mask = (ts >= t0) & (ts <= t1)
    return ts[mask], pos[mask], quat[mask]


def interp_pos(ts_src, pos_src, ts_tgt):
    """Linear position interpolation to target timestamps."""
    out = np.empty((len(ts_tgt), 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = np.interp(ts_tgt, ts_src, pos_src[:, i])
    return out


def interp_quat_nn(ts_src, quat_src, ts_tgt):
    """Nearest-neighbour quaternion interpolation, normalised."""
    idx = np.clip(np.searchsorted(ts_src, ts_tgt), 0, len(ts_src) - 1)
    q = quat_src[idx].copy()
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q /= np.where(norms > 1e-9, norms, 1.0)
    return q


# ── Umeyama SE3 alignment ──────────────────────────────────────────────────────

def umeyama_align(src: np.ndarray, dst: np.ndarray):
    """
    Estimate the rigid body transform  dst ≈ R @ src + t  (no scale).

    Parameters
    ----------
    src, dst : (N, 3) — estimated and GT positions, time-matched

    Returns
    -------
    R : (3, 3)
    t : (3,)
    """
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst

    H = src_c.T @ dst_c / len(src)          # (3, 3) cross-covariance
    U, _, Vt = np.linalg.svd(H)

    # Enforce proper rotation (det = +1)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [x, y, z, w]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s,
                         (R[1,0]-R[0,1])*s, 0.25/s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s,
                         (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s,
                         (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s,
                         0.25*s, (R[1,0]-R[0,1])/s])


def qmul_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch quaternion multiplication (N,4) × (N,4) → (N,4), [x,y,z,w]."""
    ax, ay, az, aw = a[:,0], a[:,1], a[:,2], a[:,3]
    bx, by, bz, bw = b[:,0], b[:,1], b[:,2], b[:,3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=1)


def apply_align(R: np.ndarray, t: np.ndarray,
                pos: np.ndarray, quat: np.ndarray):
    """Apply SE3 to positions and quaternions."""
    pos_aligned  = (R @ pos.T).T + t

    q_R = rot_to_quat(R)                           # scalar alignment quaternion
    q_R_batch = np.tile(q_R, (len(quat), 1))       # (N, 4)
    quat_aligned = qmul_batch(q_R_batch, quat)
    norms = np.linalg.norm(quat_aligned, axis=1, keepdims=True)
    quat_aligned /= np.where(norms > 1e-9, norms, 1.0)

    return pos_aligned, quat_aligned


# ── ATE ───────────────────────────────────────────────────────────────────────

def compute_ate_pos(p_est_aligned: np.ndarray, p_gt: np.ndarray) -> dict:
    """Translation ATE (metres) after alignment."""
    err = np.linalg.norm(p_est_aligned - p_gt, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean": float(np.mean(err)),
        "std":  float(np.std(err)),
        "max":  float(np.max(err)),
        "samples": err,
    }


def compute_ate_rot(q_est_aligned: np.ndarray, q_gt: np.ndarray) -> dict:
    """
    Rotation ATE (degrees) after alignment.
    Uses geodesic angle:  θ = 2 · arccos(|w of q_gt⁻¹ * q_est|)
    """
    # q_gt_inv: conjugate of unit quaternion = [-x,-y,-z,w]
    q_gt_inv = q_gt * np.array([-1, -1, -1, 1], dtype=np.float64)
    q_rel    = qmul_batch(q_gt_inv, q_est_aligned)
    norms    = np.linalg.norm(q_rel, axis=1, keepdims=True)
    q_rel   /= np.where(norms > 1e-9, norms, 1.0)
    w        = np.clip(np.abs(q_rel[:, 3]), 0.0, 1.0)
    angles   = np.degrees(2.0 * np.arccos(w))
    return {
        "mean": float(np.mean(angles)),
        "max":  float(np.max(angles)),
        "samples": angles,
    }


# ── RPE ───────────────────────────────────────────────────────────────────────

def compute_rpe(p_est: np.ndarray, p_gt: np.ndarray,
                segment_lengths_m=(0.5, 1.0, 2.0, 5.0)) -> dict:
    """
    Relative Pose Error (translation) over fixed path-length segments.

    For each segment length d, finds pairs (i,j) where GT path-length ≈ d,
    then measures the deviation of the estimated relative displacement from GT.

    Returns
    -------
    dict: {segment_length_m: np.ndarray of translation errors [m]}
    """
    step_d = np.linalg.norm(np.diff(p_gt, axis=0), axis=1)
    cum_d  = np.concatenate([[0.0], np.cumsum(step_d)])

    results = {}
    for seg in segment_lengths_m:
        errors = []
        for i in range(len(p_gt)):
            target = cum_d[i] + seg
            if target > cum_d[-1]:
                break
            j = int(np.searchsorted(cum_d, target, side="left"))
            if j >= len(p_gt):
                break
            rel_gt  = p_gt[j]  - p_gt[i]
            rel_est = p_est[j] - p_est[i]
            errors.append(float(np.linalg.norm(rel_est - rel_gt)))
        results[seg] = np.array(errors, dtype=np.float64)
    return results


# ── Text summary ───────────────────────────────────────────────────────────────

def format_block(label: str, ate_pos: dict, ate_rot: dict, rpe: dict) -> str:
    lines = [
        f"{'─'*52}",
        f"  {label}",
        f"{'─'*52}",
        f"  ATE — position (after Umeyama SE3 alignment)",
        f"    RMSE : {ate_pos['rmse']*100:8.3f} cm",
        f"    mean : {ate_pos['mean']*100:8.3f} cm",
        f"    std  : {ate_pos['std']*100:8.3f} cm",
        f"    max  : {ate_pos['max']*100:8.3f} cm",
        f"",
        f"  ATE — rotation (after SE3 alignment)",
        f"    mean : {ate_rot['mean']:8.3f} °",
        f"    max  : {ate_rot['max']:8.3f} °",
        f"",
        f"  RPE — translation drift per segment",
        f"    {'seg':>6}  {'mean':>8}  {'std':>8}  {'max':>8}  {'N':>5}",
    ]
    for seg, errs in sorted(rpe.items()):
        if len(errs) == 0:
            lines.append(f"    {seg:>5.1f}m   {'N/A':>8}")
            continue
        pct = errs / seg * 100
        lines.append(
            f"    {seg:>5.1f}m"
            f"  {pct.mean():>7.3f}%"
            f"  {pct.std():>7.3f}%"
            f"  {pct.max():>7.3f}%"
            f"  {len(errs):>5}"
        )
    lines.append("")
    return "\n".join(lines)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_trajectories(gt_pos, eskf_pos, vio_pos, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Trajectory Comparison  (ESKF and VIO after SE3 alignment to GT)",
                 fontsize=12)

    views = [
        (0, 1, "X [m]", "Y [m]", "Top-down  XY"),
        (0, 2, "X [m]", "Z [m]", "Side view  XZ"),
    ]
    for ax, (xi, yi, xl, yl, title) in zip(axes, views):
        ax.plot(gt_pos[:,xi],   gt_pos[:,yi],   "k-",  lw=1.4, label="GT",   alpha=0.9)
        ax.plot(eskf_pos[:,xi], eskf_pos[:,yi], "b--", lw=1.0, label="ESKF", alpha=0.85)
        ax.plot(vio_pos[:,xi],  vio_pos[:,yi],  "r:",  lw=1.0, label="VIO",  alpha=0.85)
        # Start markers
        ax.plot(gt_pos[0,xi],   gt_pos[0,yi],   "ko", ms=6)
        ax.plot(eskf_pos[0,xi], eskf_pos[0,yi], "b^", ms=6)
        ax.plot(vio_pos[0,xi],  vio_pos[0,yi],  "rs", ms=6)
        ax.set(xlabel=xl, ylabel=yl, title=title)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ate_over_time(ts_eskf, ate_eskf_s, ts_vio, ate_vio_s,
                       ts_eskf_r, rot_eskf_s, ts_vio_r, rot_vio_s,
                       out_path: Path):
    fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle("Absolute Trajectory Error over Time", fontsize=12)

    t_e = (ts_eskf - ts_eskf[0]) * 1e-9
    t_v = (ts_vio  - ts_vio[0])  * 1e-9

    ax_t.plot(t_e, ate_eskf_s * 100, "b-",  lw=0.7, label="ESKF", alpha=0.9)
    ax_t.plot(t_v, ate_vio_s  * 100, "r--", lw=0.7, label="VIO",  alpha=0.85)
    ax_t.set(ylabel="Position ATE [cm]")
    ax_t.legend(fontsize=9)
    ax_t.grid(True, alpha=0.35)

    t_er = (ts_eskf_r - ts_eskf_r[0]) * 1e-9
    t_vr = (ts_vio_r  - ts_vio_r[0])  * 1e-9

    ax_r.plot(t_er, rot_eskf_s, "b-",  lw=0.7, label="ESKF", alpha=0.9)
    ax_r.plot(t_vr, rot_vio_s,  "r--", lw=0.7, label="VIO",  alpha=0.85)
    ax_r.set(xlabel="Elapsed time [s]", ylabel="Rotation ATE [°]")
    ax_r.legend(fontsize=9)
    ax_r.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rpe_boxplot(rpe_eskf: dict, rpe_vio: dict, out_path: Path):
    seg_lens = sorted(rpe_eskf.keys())
    n = len(seg_lens)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("RPE — translation drift % per segment length", fontsize=12)

    colours = {"ESKF": "#4477aa", "VIO": "#cc3333"}
    for ax, seg in zip(axes, seg_lens):
        data = {
            "ESKF": rpe_eskf[seg] / seg * 100,
            "VIO":  rpe_vio[seg]  / seg * 100,
        }
        bp = ax.boxplot(
            list(data.values()),
            labels=list(data.keys()),
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            flierprops=dict(marker=".", ms=3, alpha=0.5),
        )
        for patch, label in zip(bp["boxes"], data.keys()):
            patch.set_facecolor(colours[label])
        ax.set_title(f"{seg:.1f} m segment")
        ax.set_ylabel("Drift [%]" if ax is axes[0] else "")
        ax.grid(True, axis="y", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate ESKF / VIO trajectories against ground truth."
    )
    ap.add_argument("--input", default="tmp",
                    help="Directory containing pose_*.csv files (default: tmp/)")
    ap.add_argument("--logs",  default="logs",
                    help="Parent directory for output folders (default: logs/)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir   = Path(args.logs) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    files = {
        "gt":   input_dir / "pose_gt.csv",
        "eskf": input_dir / "pose_eskf.csv",
        "vio":  input_dir / "pose_vio.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        print(f"ERROR: missing files: {missing}", file=sys.stderr)
        sys.exit(1)

    print("Loading trajectories...")
    ts_gt,   pos_gt,   q_gt   = load_traj(files["gt"])
    ts_eskf, pos_eskf, q_eskf = load_traj(files["eskf"])
    ts_vio,  pos_vio,  q_vio  = load_traj(files["vio"])

    for name, ts in [("GT", ts_gt), ("ESKF", ts_eskf), ("VIO", ts_vio)]:
        if len(ts) == 0:
            print(f"ERROR: {name} CSV is empty.", file=sys.stderr)
            sys.exit(1)

    print(f"  GT   : {len(ts_gt):6d} samples")
    print(f"  ESKF : {len(ts_eskf):6d} samples")
    print(f"  VIO  : {len(ts_vio):6d} samples")

    # ── Trim to common time window ────────────────────────────────────────────
    t0 = max(ts_gt[0],  ts_eskf[0],  ts_vio[0])
    t1 = min(ts_gt[-1], ts_eskf[-1], ts_vio[-1])
    if t0 >= t1:
        print("ERROR: trajectories have no overlapping time window.", file=sys.stderr)
        sys.exit(1)

    ts_gt,   pos_gt,   q_gt   = trim(ts_gt,   pos_gt,   q_gt,   t0, t1)
    ts_eskf, pos_eskf, q_eskf = trim(ts_eskf, pos_eskf, q_eskf, t0, t1)
    ts_vio,  pos_vio,  q_vio  = trim(ts_vio,  pos_vio,  q_vio,  t0, t1)

    duration_s = (t1 - t0) * 1e-9
    total_dist = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    print(f"  Overlap: {duration_s:.1f} s,  GT path: {total_dist:.2f} m")

    # ── Interpolate GT to estimator timestamps ────────────────────────────────
    pos_gt_at_eskf = interp_pos(ts_gt, pos_gt, ts_eskf)
    q_gt_at_eskf   = interp_quat_nn(ts_gt, q_gt, ts_eskf)

    pos_gt_at_vio  = interp_pos(ts_gt, pos_gt, ts_vio)
    q_gt_at_vio    = interp_quat_nn(ts_gt, q_gt, ts_vio)

    # ── Umeyama SE3 alignment ─────────────────────────────────────────────────
    print("Aligning trajectories (Umeyama SE3)...")
    R_eskf, t_eskf = umeyama_align(pos_eskf, pos_gt_at_eskf)
    R_vio,  t_vio  = umeyama_align(pos_vio,  pos_gt_at_vio)

    pos_eskf_al, q_eskf_al = apply_align(R_eskf, t_eskf, pos_eskf, q_eskf)
    pos_vio_al,  q_vio_al  = apply_align(R_vio,  t_vio,  pos_vio,  q_vio)

    # ── ATE ───────────────────────────────────────────────────────────────────
    print("Computing ATE...")
    ate_pos_eskf = compute_ate_pos(pos_eskf_al, pos_gt_at_eskf)
    ate_rot_eskf = compute_ate_rot(q_eskf_al,   q_gt_at_eskf)

    ate_pos_vio  = compute_ate_pos(pos_vio_al,  pos_gt_at_vio)
    ate_rot_vio  = compute_ate_rot(q_vio_al,    q_gt_at_vio)

    # ── RPE ───────────────────────────────────────────────────────────────────
    print("Computing RPE...")
    seg_lens = [0.5, 1.0, 2.0, 5.0]
    rpe_eskf = compute_rpe(pos_eskf_al, pos_gt_at_eskf, seg_lens)
    rpe_vio  = compute_rpe(pos_vio_al,  pos_gt_at_vio,  seg_lens)

    # ── Summary text ──────────────────────────────────────────────────────────
    header = (
        f"VIO Trajectory Evaluation Report\n"
        f"{'='*52}\n"
        f"  Generated  : {timestamp}\n"
        f"  Input dir  : {input_dir.resolve()}\n"
        f"  GT samples : {len(ts_gt)}\n"
        f"  Duration   : {duration_s:.1f} s\n"
        f"  GT path    : {total_dist:.2f} m\n\n"
    )
    eskf_block = format_block("ESKF vs Ground Truth", ate_pos_eskf, ate_rot_eskf, rpe_eskf)
    vio_block  = format_block("VIO  vs Ground Truth", ate_pos_vio,  ate_rot_vio,  rpe_vio)
    summary = header + eskf_block + "\n" + vio_block

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary)
    print("\n" + summary)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if _HAS_MPL:
        print("Saving plots...")

        # Trajectory: downsample ESKF (200 Hz → ~20 Hz) to avoid over-plotting
        step = max(1, len(ts_eskf) // max(len(ts_vio), 1))
        plot_trajectories(
            pos_gt,
            pos_eskf_al[::step],
            pos_vio_al,
            out_dir / "trajectory.png",
        )
        print("  trajectory.png")

        plot_ate_over_time(
            ts_eskf[::step], ate_pos_eskf["samples"][::step],
            ts_vio,          ate_pos_vio["samples"],
            ts_eskf[::step], ate_rot_eskf["samples"][::step],
            ts_vio,          ate_rot_vio["samples"],
            out_dir / "ate_over_time.png",
        )
        print("  ate_over_time.png")

        plot_rpe_boxplot(rpe_eskf, rpe_vio, out_dir / "rpe_boxplot.png")
        print("  rpe_boxplot.png")

    # ── Archive raw data into the log dir ────────────────────────────────────
    print("Archiving raw data...")
    for csv_path in glob.glob(str(input_dir / "*.csv")):
        shutil.copy2(csv_path, out_dir)
        print(f"  {Path(csv_path).name}")
    params_src = input_dir / "pipeline_params.yaml"
    if params_src.exists():
        shutil.copy2(params_src, out_dir)
        print("  pipeline_params.yaml")

    print(f"\nAll results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
