# Stereo Visual Odometry Pipeline (VIO — vision-only stage)

A ROS 2 stereo visual-odometry system for the [EuRoC MAV dataset](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f).
Feature extraction uses **SuperPoint + LightGlue** with a 4-way circular consistency check;
pose estimation uses stereo triangulation followed by PnP-RANSAC.

> **Status:** Vision-only. IMU fusion (Error-State Kalman Filter) is planned as future work.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Feature extraction — SuperPoint + LightGlue](#2-feature-extraction--superpoint--lightglue)
3. [4-way circular consistency check](#3-4-way-circular-consistency-check)
4. [Stereo triangulation](#4-stereo-triangulation)
5. [Pose estimation — PnP-RANSAC](#5-pose-estimation--pnp-ransac)
6. [Coordinate frames](#6-coordinate-frames)
7. [Project structure](#7-project-structure)
8. [Requirements and installation](#8-requirements-and-installation)
9. [Running the pipeline](#9-running-the-pipeline)
10. [ROS 2 topics and parameters](#10-ros-2-topics-and-parameters)
11. [Configuration file](#11-configuration-file)
12. [Future work — IMU fusion with ESKF](#12-future-work--imu-fusion-with-eskf)

---

## 1. Architecture overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EuRoC MAV Bag                                   │
│   /cam0/image_raw (20 Hz)       /cam1/image_raw (20 Hz)                 │
└────────────────┬──────────────────────────┬────────────────────────────┘
                 │  ApproximateTimeSynchronizer
                 ▼                          ▼
       ┌─────────────────────────────────────────────┐
       │           PoseEstimationNode                │
       │                                             │
       │  SuperPoint ──► LightGlue                   │
       │    stereo match     temporal match          │
       │          └──────────────────┘               │
       │          4-way circular check               │
       │                  │                          │
       │     circular_tracks (verified)              │
       │          │                                  │
       │   undistort keypoints                       │
       │          │                                  │
       │   triangulate 3D (prev stereo pair)         │
       │          │                                  │
       │   PnP-RANSAC + iterative refinement         │
       │          │                                  │
       │   motion sanity check                       │
       │          │                                  │
       │   T_world_cam0 ◄── accumulate               │
       │          │                                  │
       │   T_world_body = T_world_cam0 @ inv(T_b_c0) │
       └──────────┬──────────────────────────────────┘
                  │
        ┌─────────┼───────────────────┐
        ▼         ▼                   ▼
  /vio/pose  /vio/path        /vio/odometry

       ┌──────────────────────────────┐
       │   GroundTruthPublisherNode   │  ← /vicon0/pose (or similar)
       │  align to initial body frame │
       └──────────┬───────────────────┘
                  │
        /gt_pub/pose   /gt_pub/path   /gt_pub/odometry
```

---

## 2. Feature extraction — SuperPoint + LightGlue

### SuperPoint

SuperPoint ([DeTone et al., 2018](https://arxiv.org/abs/1712.07629)) is a self-supervised CNN
that jointly detects keypoints and computes their descriptors in a single forward pass.

Given a grayscale image **I** ∈ ℝ^{H×W}, the network produces:

- **Keypoints** p_i ∈ ℝ² (sub-pixel 2D image coordinates)
- **Descriptors** d_i ∈ ℝ^{256} (L2-normalised)
- **Confidence scores** s_i ∈ [0, 1]

The encoder is a VGG-style backbone; the decoder has two heads — one for interest-point
heatmaps and one for descriptor maps. Up to 2048 keypoints are extracted per image.

**Why SuperPoint?** Classical detectors (SIFT, ORB) rely on hand-crafted gradients.
SuperPoint learns to detect repeatable, matchable interest points from synthetic homographic
transformations, giving it strong performance under illumination change and viewpoint variation.

### LightGlue

LightGlue ([Lindenberger et al., 2023](https://arxiv.org/abs/2306.13643)) is a graph-neural-
network-based matcher that computes soft correspondences between two sets of SuperPoint
keypoints.

For two images A and B with keypoints {p_i^A, d_i^A} and {p_j^B, d_j^B}, LightGlue:

1. Encodes each keypoint with a positional encoding:

   ```
   f_i = MLP([d_i, PE(p_i)])
   ```

2. Applies L layers of attentional graph neural network (self + cross attention):

   ```
   f_i ← f_i + Σ_j α_{ij} · W · f_j
   ```

3. Computes a partial assignment matrix **S** ∈ ℝ^{(N+1)×(M+1)} (augmented with dustbin rows/
   columns for unmatched keypoints) via a differentiable Sinkhorn solver.

4. The final matches M = {(i, j) | argmax_j S_{ij} = j ∧ argmax_i S_{ij} = i} give mutual
   nearest-neighbour correspondences with confidence scores.

**Adaptive early exit:** LightGlue stops processing easy image pairs at earlier layers,
making it significantly faster than LoFTR or full SuperGlue on sequences with many similar
consecutive frames.

**Why LightGlue over traditional matchers?**
Nearest-neighbour matching on raw descriptors is O(NM) and requires careful ratio-test tuning.
LightGlue encodes geometric context (keypoint position relative to the image), is end-to-end
trained for matching quality, and explicitly handles unmatched points — giving cleaner
correspondences for downstream geometry.

### GPU acceleration

SuperPoint and LightGlue are PyTorch models. On a mid-range GPU (e.g., RTX 3060):

| Platform | ~throughput |
|---|---|
| CUDA GPU | 15–25 Hz (stereo + temporal pass) |
| CPU | 1–3 Hz |

The node automatically falls back to CPU if CUDA is unavailable, but real-time operation
requires a GPU. Set the `device` parameter to `"cuda"` to force GPU, or leave blank for
auto-detection.

---

## 3. 4-way circular consistency check

Raw LightGlue matches can still contain outliers. The pipeline performs a **4-way circular
check** across the stereo-temporal track quadruplet before any geometry is computed.

### The four match chains

For each consecutive frame pair (t−1, t), four LightGlue match sets are computed:

```
M_LL : L_{t-1} → L_t          (temporal left)
M_LR : L_t     → R_t          (stereo current)
M_RR : R_t     → R_{t-1}      (temporal right, reversed)
M_RL : L_{t-1} → R_{t-1}      (stereo previous)
```

### Consistency criterion

Starting from keypoint **i** in L_{t−1}, the chain is followed:

```
i  →[M_LL]→  j  →[M_LR]→  k  →[M_RR]→  l  →[M_RL^{-1}]→  î
```

The track is accepted only if the **loop-closure reprojection error** is below a pixel
threshold τ:

```
‖p^{L_{t-1}}_i  −  p^{L_{t-1}}_î‖₂  ≤  τ   (default τ = 2 px)
```

A track that passes carries four geometrically consistent 2D observations:

```
circular_tracks = {
  kpts_l_prev (N,2),   kpts_l_curr (N,2),
  kpts_r_prev (N,2),   kpts_r_curr (N,2)
}
```

This provides two stereo pairs and two temporal correspondences from a single verification
step, with no additional LightGlue calls required. The typical outlier rejection rate is
50–80%, leaving a clean set of tracks for triangulation and PnP.

---

## 4. Stereo triangulation

### Projection matrices

Given camera intrinsics **K₀**, **K₁** and the stereo extrinsic **T_{c₁c₀}** (transform
that maps cam0 coordinates to cam1 coordinates, derived from the Kalibr calibration):

```
T_{c₁c₀} = T_{bc₁}⁻¹ · T_{bc₀}
```

The projection matrices (cam0 as reference frame) are:

```
P₀ = K₀ · [I | 0]
P₁ = K₁ · [R_{c₁c₀} | t_{c₁c₀}]
```

### Undistortion

SuperPoint keypoints are detected on the original (distorted) images. Before any geometry,
all keypoints are mapped to ideal (undistorted) pixel coordinates using the radial-tangential
distortion model:

```
p_u = undistortPoints(p_d, K, [k₁, k₂, p₁, p₂], P=K)
```

This ensures a consistent pinhole model throughout.

### DLT triangulation

For a matched pair (p₀, p₁) in undistorted pixel coordinates, the homogeneous 3D point
**X** is recovered by solving the linear system (Direct Linear Transform):

```
[p₀ × P₀] · X = 0
[p₁ × P₁]
```

using `cv2.triangulatePoints`. Points with depth Z < 0.1 m or Z > 30 m (configurable) are
discarded to remove degenerate stereo observations.

---

## 5. Pose estimation — PnP-RANSAC

### Problem statement

Given N 3D landmarks {X_i} triangulated in cam0_{t−1} frame and their 2D observations
{p_i} in the cam0_t image, find the rigid transform T such that:

```
λ · p_i  =  K · (R · X_i + t)
```

This is the **Perspective-n-Point (PnP)** problem.

### RANSAC + EPnP

`cv2.solvePnPRansac` with `SOLVEPNP_EPNP` is used:

1. **EPnP** ([Lepetit et al., 2009](https://link.springer.com/article/10.1007/s11263-008-0152-6))
   expresses each 3D point as a weighted sum of four virtual control points, reducing PnP to a
   linear system solvable in O(N) time.

2. **RANSAC** (200 iterations, reprojection threshold 2 px, confidence 99.9%) randomly samples
   minimal sets of 4 correspondences, estimates a hypothesis with EPnP, and counts inliers.

### Iterative refinement

The RANSAC solution is used as an initial guess for Levenberg–Marquardt non-linear refinement
(`SOLVEPNP_ITERATIVE`) on the full inlier set, minimising total reprojection error:

```
{R*, t*} = argmin_{R,t} Σ_{i∈inliers} ‖p_i − π(K, R, t, X_i)‖²
```

### Layered outlier rejection

| Guard | Condition | Rejects |
|---|---|---|
| Depth filter | 0.1 m < Z < 30 m | Far/noisy stereo points |
| Minimum count | N_valid ≥ 6 | Degenerate configurations |
| Inlier ratio | n_inliers / N ≥ 0.4 | RANSAC on corrupt consensus |
| Max translation | ‖t‖ ≤ 0.5 m / frame | Pose jumps |
| Max rotation | angle ≤ 30° / frame | Orientation flips |

If any guard fires, the frame is **skipped** and the last accepted pose is held.

### Pose accumulation

`solvePnPRansac` returns **T_{curr,prev}** — the transform that maps cam0_{t-1} coordinates
to cam0_t. The world-frame cam0 pose is updated by right-multiplication of the inverse:

```
T_{world,cam0}^{t} = T_{world,cam0}^{t-1} · T_{curr,prev}⁻¹
```

The published body pose is then:

```
T_{world,body} = T_{world,cam0} · T_{bc0}⁻¹
```

---

## 6. Coordinate frames

### Kalibr convention (T_BS)

All extrinsic matrices use the Kalibr convention:

```
p_body = T_BS · p_sensor
```

So `T_BS` (named `T_b_c0` in code) transforms a point expressed in the **sensor** frame
into the **body** frame.

### World frame

The world frame is defined as the **initial body frame** (FLU — Forward, Left, Up):

```
+X  →  forward    (body x at t=0)
+Y  →  left       (body y at t=0)
+Z  →  up         (body z at t=0, approximately anti-gravity)
```

Initialising `T_{world,cam0} = T_{bc0}` (instead of identity) ensures that at t=0:

```
T_{world,body}^{0} = T_{bc0} · T_{bc0}⁻¹ = I
```

Both the odometry and the aligned ground truth start at the origin with identity
orientation, making trajectory comparison direct.

### Ground truth alignment

The Vicon ground truth is expressed in the Vicon world frame. To align it to the odometry
world frame (initial body frame):

```
T_align = T_{vicon,body_0}⁻¹

T_{world,body}^{GT}(t) = T_align · T_{vicon,body}(t)
```

At t=0 this evaluates to identity, matching the odometry initial condition.

---

## 7. Project structure

```
VIO_new/
├── requirements.txt
├── README.md
└── ros2_ws/
    └── src/
        └── vio_pipeline/
            ├── package.xml
            ├── setup.py
            ├── setup.cfg
            ├── config/
            │   └── euroc_params.yaml       # Camera + IMU calibration
            ├── launch/
            │   ├── full_pipeline.launch.py # Pose estimation + GT publisher
            │   └── features.launch.py      # Feature visualisation only
            └── vio_pipeline/
                ├── feature_extraction.py   # SuperPoint + LightGlue wrapper,
                │                           # 4-way circular check
                ├── feature_tracking_node.py# Debug visualisation node
                ├── vio_node.py             # PoseEstimationNode (main pipeline)
                └── ground_truth_publisher.py # Aligned GT republisher
```

### Node summary

| Node | Executable | Purpose |
|---|---|---|
| `PoseEstimationNode` | `pose_estimation_node` | Feature tracking → triangulation → PnP → publish pose |
| `FeatureTrackingNode` | `feature_tracking_node` | Standalone feature viz (debug) |
| `GroundTruthPublisherNode` | `ground_truth_publisher` | Align Vicon GT to odometry frame |

---

## 8. Requirements and installation

### System requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 22.04 / 24.04 |
| ROS 2 | Humble | Jazzy |
| Python | 3.10 | 3.11 |
| GPU | — (CPU fallback) | 2GB+ VRAM |


### 1. Install ROS 2

Follow the official ROS 2 installation guide for your distribution.

### 2. Install PyTorch (GPU)

```bash
# Check your CUDA version first:
nvidia-smi

# Install matching PyTorch build (example: CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU-only (real-time performance not guaranteed):
pip install torch torchvision
```

### 3. Install LightGlue

LightGlue is not on PyPI and must be installed from source:

```bash
pip install git+https://github.com/cvg/LightGlue.git
```

### 4. Install remaining Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Skip the `torch`/`torchvision` lines in `requirements.txt` if you already
> installed a CUDA build in step 2, to avoid overwriting with the CPU version.

### 5. Install ROS 2 package dependencies

```bash
cd ros2_ws
rosdep install --from-paths src -y --ignore-src
```

### 6. Build and source

```bash
cd ros2_ws
colcon build --packages-select vio_pipeline
source install/setup.bash
```

---

## 9. Running the pipeline

### Prepare the EuRoC bag

Download an EuRoC sequence (e.g. MH_01_easy) and convert to ROS 2 bag format, or use a
pre-converted bag. Place it at `bags/euroc_mav0/`.

### Terminal A — play the bag

```bash
ros2 bag play bags/euroc_mav0 --rate 0.5 --clock
```

`--rate 0.5` plays at half speed. Remove it (or increase) once you have GPU acceleration.
`--clock` publishes `/clock` so nodes use simulated time.

### Terminal B — launch the pipeline

```bash
# Full pipeline (pose estimation + ground truth)
ros2 launch vio_pipeline full_pipeline.launch.py

# Feature visualisation only (debug)
ros2 launch vio_pipeline features.launch.py
```

Override defaults with launch arguments:

```bash
ros2 launch vio_pipeline full_pipeline.launch.py \
    config_file:=/path/to/custom_params.yaml \
    use_sim_time:=true
```

### Terminal C — inspect output

```bash
# Live pose stream
ros2 topic echo /vio/pose

# Topic rates
ros2 topic hz /vio/pose
ros2 topic hz /gt_pub/pose

# RViz (add Path displays for /vio/path and /gt_pub/path)
rviz2
```

### Feature tracking only

```bash
ros2 launch vio_pipeline features.launch.py
# Then view in RViz: /features/viz, /features/temporal_viz
```

---

## 10. ROS 2 topics and parameters

### Published topics

| Topic | Type | Node | Description |
|---|---|---|---|
| `/vio/pose` | `PoseStamped` | PoseEstimation | Body pose in world frame |
| `/vio/path` | `Path` | PoseEstimation | Full trajectory history |
| `/vio/odometry` | `Odometry` | PoseEstimation | Body odometry (`child_frame_id=base_link`) |
| `/gt_pub/pose` | `PoseStamped` | GroundTruth | Aligned ground truth pose |
| `/gt_pub/path` | `Path` | GroundTruth | Ground truth trajectory |
| `/gt_pub/odometry` | `Odometry` | GroundTruth | Ground truth odometry |
| `/features/viz` | `Image` | FeatureTracking | Stereo match visualisation |
| `/features/temporal_viz` | `Image` | FeatureTracking | Temporal track visualisation |

### Subscribed topics

| Topic | Type | Node |
|---|---|---|
| `/cam0/image_raw` | `Image` | PoseEstimation, FeatureTracking |
| `/cam1/image_raw` | `Image` | PoseEstimation, FeatureTracking |
| `/vicon0/pose` (configurable) | `PoseStamped` | GroundTruth |

### PoseEstimationNode parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to `euroc_params.yaml` |
| `device` | `""` | `"cuda"` or `"cpu"` (blank = auto) |
| `min_tracks` | `10` | Min circular-checked tracks to attempt PnP |
| `circular_check_threshold` | `2.0` | Max loop-closure pixel error (px) |
| `max_depth` | `30.0` | Max triangulated point depth (m) |
| `min_inlier_ratio` | `0.4` | Min RANSAC inlier fraction |
| `max_translation` | `0.5` | Max accepted per-frame translation (m) |
| `max_rotation_deg` | `30.0` | Max accepted per-frame rotation (deg) |

### GroundTruthPublisherNode parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to `euroc_params.yaml` |
| `gt_topic` | `"/gt/pose"` | Input ground truth topic |

---

## 11. Configuration file

`config/euroc_params.yaml` stores the EuRoC VI-Sensor calibration:

```yaml
cam0:
  intrinsics: [fx, fy, cx, cy]          # Pinhole intrinsics (pixels)
  distortion: [k1, k2, p1, p2]          # Radial-tangential distortion
  resolution: [752, 480]
  T_BS: [...]                            # 4×4 row-major; cam0→body (Kalibr)

cam1:
  intrinsics: [fx, fy, cx, cy]
  distortion: [k1, k2, p1, p2]
  resolution: [752, 480]
  T_BS: [...]                            # cam1→body

stereo_baseline: 0.11007                 # ‖t_{c₀c₁}‖ in metres

imu:
  gyro_noise:  1.6968e-04               # rad/s/√Hz
  gyro_walk:   1.9393e-05               # rad/s²/√Hz
  accel_noise: 2.0000e-03               # m/s²/√Hz
  accel_walk:  3.0000e-03               # m/s³/√Hz
  rate_hz: 200
  T_BS: [identity]                       # IMU = body for EuRoC

gravity: [0.0, 0.0, -9.81]
```

---

## 12. Future work — IMU fusion with ESKF

The current pipeline is **vision-only**. Integrating IMU measurements will improve robustness
during fast motion and provide a well-defined gravity-aligned world frame. The planned approach
is an **Error-State Kalman Filter (ESKF)**.

### State vector

The nominal state at time t is:

```
x = [p, v, q, b_a, b_g]  ∈  ℝ³ × ℝ³ × SO(3) × ℝ³ × ℝ³
```

where **p** is position, **v** velocity, **q** orientation quaternion, **b_a** accelerometer
bias, **b_g** gyroscope bias. The error state **δx** lives in ℝ^{15}.

### IMU pre-integration (prediction step)

Between camera frames at times t_{k} and t_{k+1}, IMU measurements {a_m, ω_m} are integrated
to propagate the nominal state:

```
p_{k+1} = p_k + v_k·Δt + ½(R_k·(a_m − b_a) + g)·Δt²
v_{k+1} = v_k + (R_k·(a_m − b_a) + g)·Δt
R_{k+1} = R_k · Exp((ω_m − b_g)·Δt)
```

The error-state covariance **P** is propagated via the linearised dynamics:

```
P_{k+1} = F·P_k·Fᵀ + G·Q·Gᵀ
```

where **F** is the state transition Jacobian, **G** the noise input matrix, and **Q** the
IMU noise covariance (diagonal, loaded from `euroc_params.yaml`).

### Visual update step

When a new camera frame arrives, the PnP reprojection residual for each landmark provides
an observation:

```
z_i = p_i − π(K, T_{body→cam0}, T_{world,body}, X_i)
```

The linearised measurement Jacobian **H_i** maps the error state to residual space.
The Kalman gain and state correction follow the standard EKF update:

```
K  = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
δx = K·z
P  = (I − K·H)·P
```

After the update the nominal state is reset:
```
x ← x ⊕ δx,   P ← reset(P)
```

### Expected benefits over vision-only

| Limitation (current) | ESKF solution |
|---|---|
| Jumps during fast rotation | IMU provides smooth attitude between frames |
| World frame not gravity-aligned | Gravity from accelerometer init fixes z-up |
| Scale drift accumulates | IMU constrains velocity and scale |
| Tracking loss = lost pose | IMU propagates pose through dark/blurry frames |

The IMU parameters already present in `euroc_params.yaml` (noise densities, random walks) are
the exact inputs needed to build the process noise matrix **Q** for the ESKF.
