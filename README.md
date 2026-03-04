# Visual-Inertial Odometry Pipeline (KLT branch)

A ROS 2 stereo visual-inertial odometry system for the [EuRoC MAV dataset](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f).
Feature tracking uses **Shi-Tomasi corners + KLT optical flow** with a forward-backward
circular consistency check; pose estimation uses stereo triangulation followed by PnP-RANSAC;
IMU fusion is performed by an **Error-State Kalman Filter (ESKF)**.

> **Branch:** `KLT` — classical feature tracker, no deep-learning dependencies.
> The `main` branch uses SuperPoint + LightGlue.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Feature tracking — Shi-Tomasi + KLT](#2-feature-tracking--shi-tomasi--klt)
3. [Forward-backward circular consistency check](#3-forward-backward-circular-consistency-check)
4. [Stereo triangulation](#4-stereo-triangulation)
5. [Pose estimation — PnP-RANSAC](#5-pose-estimation--pnp-ransac)
6. [IMU processing](#6-imu-processing)
7. [Error-State Kalman Filter (ESKF)](#7-error-state-kalman-filter-eskf)
8. [Coordinate frames](#8-coordinate-frames)
9. [Project structure](#9-project-structure)
10. [Requirements and installation](#10-requirements-and-installation)
11. [Running the pipeline](#11-running-the-pipeline)
12. [ROS 2 topics and parameters](#12-ros-2-topics-and-parameters)
13. [Configuration file](#13-configuration-file)

---

## 1. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            EuRoC MAV Bag                                     │
│  /cam0/image_raw (20 Hz)   /cam1/image_raw (20 Hz)   /imu0 (200 Hz)         │
└──────┬──────────────────────────┬──────────────────────┬─────────────────────┘
       │  ApproximateTimeSynchronizer                     │
       ▼                          ▼                       ▼
┌──────────────────────────────────────┐    ┌────────────────────────┐
│         PoseEstimationNode           │    │   ImuProcessingNode    │
│                                      │    │                        │
│  Shi-Tomasi detect                   │    │  static bias init      │
│       │                              │    │  2nd-order Butterworth │
│  KLT temporal tracking (L→L)         │    │  LPF (gyro + accel)    │
│  + forward-backward consistency      │    │  dead-reckoning        │
│       │                              │    │  integrate             │
│  KLT stereo matching (L→R)           │    └────────┬───────────────┘
│  + forward-backward consistency      │             │ /imu/processed (200 Hz)
│       │                              │             │ /imu/odometry
│  circular_tracks (verified)          │             │ /imu/pose
│       │                              │             │ /imu/rpy
│  undistort keypoints                 │             │
│       │                              │             ▼
│  triangulate 3D (prev stereo pair)   │    ┌────────────────────────┐
│       │                              │    │      EskfNode          │
│  PnP-RANSAC + iterative refinement   │    │                        │
│       │                              │    │  IMU prediction 200 Hz │
│  motion sanity check                 │    │  VIO update    20 Hz   │
│       │                              │    │  Joseph-form update    │
│  T_world_cam0 ◄── accumulate         │    │  bias estimation       │
│       │                              │    └────────┬───────────────┘
│  T_world_body = T_world_cam0         │             │ /eskf/odometry
│              @ inv(T_b_c0)           │             │ /eskf/pose
└──────┬───────────────────────────────┘             │ /eskf/rpy
       │
   /vio/pose  /vio/path  /vio/odometry  /vio/rpy

┌──────────────────────────────────┐
│   GroundTruthPublisherNode       │  ← /gt/pose (Vicon)
│   align to initial body frame    │
└──────┬───────────────────────────┘
       │
  /gt_pub/pose   /gt_pub/path   /gt_pub/odometry
```

---

## 2. Feature tracking — Shi-Tomasi + KLT

### Corner detection — Shi-Tomasi

`cv2.goodFeaturesToTrack` detects corners by maximising the minimum eigenvalue of the
local gradient structure tensor. For each candidate pixel (x, y):

```
M = Σ_{patch} [ Ix²   Ix·Iy ]
               [ Ix·Iy  Iy² ]

score = λ_min(M)
```

A point is accepted if `score ≥ qualityLevel × max_score` and its nearest accepted
neighbour is at least `minDistance` pixels away (non-maximum suppression).

**Key parameters** (defaults in `FeatureExtractor.__init__`):

| Parameter | Default | Effect |
|---|---|---|
| `max_corners` | 100 | Cap on detected corners per frame |
| `quality_level` | 0.5 | Fraction of best score; higher → fewer, stronger corners |
| `min_distance` | 20 px | Spatial spread; larger → better-conditioned geometry |

**Re-detection:** When the tracked count falls below `max_corners // 2`, fresh corners are
detected in image regions not already covered by existing tracks (exclusion mask with radius
`min_distance`), up to the cap.

### Tracking — KLT Lucas-Kanade

`cv2.calcOpticalFlowPyrLK` minimises the photometric error in a local patch around each
corner across a Gaussian image pyramid:

```
E(u, v) = Σ_{(x,y)∈patch} [ I(x, y, t) − I(x+u, y+v, t+1) ]²
```

The pyramid (depth `max_level = 3`) allows tracking across large inter-frame displacements:
each level halves resolution, so the coarsest level handles motions up to
`win_size × 2^max_level` pixels.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `win_size` | (14, 14) | Patch size per pyramid level; larger handles faster motion |
| `max_level` | 3 | Pyramid depth; 0 = no pyramid |

---

## 3. Forward-backward circular consistency check

Raw KLT tracks are noisy near occlusion boundaries and textureless regions. Each track is
validated with a **forward-backward (circular) check**:

```
1. Track p₀ forward  (img₀ → img₁):  p₁ = KLT(img₀, img₁, p₀)
2. Track p₁ backward (img₁ → img₀):  p̂₀ = KLT(img₁, img₀, p₁)
3. Accept if ‖p₀ − p̂₀‖₂ < threshold   (default 2 px)
```

This is applied independently for **temporal** tracks (left_{t−1} → left_t) and **stereo**
matches (left_t → right_t).

Tracks that survive both checks yield index-aligned quadruples:

```
kpts_l_prev  (N,2)   kpts_l_curr  (N,2)
kpts_r_prev  (N,2)   kpts_r_curr  (N,2)
```

providing two stereo pairs and two temporal correspondences from a single verification step
with no additional feature-matching calls.

---

## 4. Stereo triangulation

### Projection matrices

Given camera intrinsics **K₀**, **K₁** and the stereo extrinsic **T_{c₁c₀}** derived from
the Kalibr calibration (`T_b_c1⁻¹ @ T_b_c0`):

```
P₀ = K₀ · [I | 0]
P₁ = K₁ · [R_{c₁c₀} | t_{c₁c₀}]
```

### Undistortion

Keypoints are detected on original (distorted) images. Before any geometry, all points are
mapped to ideal pixel coordinates using the radial-tangential model:

```
p_u = undistortPoints(p_d, K, [k₁, k₂, p₁, p₂], P=K)
```

### DLT triangulation

For each stereo-matched pair (p₀, p₁), the 3D point **X** is found by solving:

```
[p₀ × P₀] · X = 0
[p₁ × P₁]
```

via `cv2.triangulatePoints`. Points with depth Z < 0.1 m or Z > `max_depth` (default 30 m)
are discarded.

---

## 5. Pose estimation — PnP-RANSAC

### Problem

Given N 3D landmarks {X_i} triangulated from the **previous** stereo pair and their 2D
observations {p_i} in the **current** left image:

```
λ · p_i = K · (R · X_i + t)
```

### RANSAC + EPnP + iterative refinement

1. `cv2.solvePnPRansac` with `SOLVEPNP_EPNP` (200 iterations, 2 px threshold, 99.9%
   confidence) finds an initial consensus set.
2. Levenberg–Marquardt (`SOLVEPNP_ITERATIVE`) refines on the full inlier set.

### Layered outlier rejection

| Guard | Condition | Rejects |
|---|---|---|
| Depth filter | 0.1 m < Z < 30 m | Noisy/degenerate stereo points |
| Minimum count | N_valid ≥ 6 | Under-determined PnP |
| Inlier ratio | n_inliers / N ≥ 0.4 | Corrupt RANSAC consensus |
| Max translation | ‖t‖ ≤ 0.5 m/frame | Pose jumps |
| Max rotation | angle ≤ 30°/frame | Orientation flips |

### Pose accumulation

`solvePnPRansac` returns **T_{curr,prev}**. The world-frame cam0 pose is updated as:

```
T_{world,cam0}^{t} = T_{world,cam0}^{t-1} · T_{curr,prev}⁻¹
T_{world,body}      = T_{world,cam0} · T_{bc0}⁻¹
```

---

## 6. IMU processing

`ImuProcessingNode` preprocesses raw IMU data before ESKF fusion.

### Static initialisation

On startup the node collects `init_duration` seconds of static data to estimate:

- **Gyroscope bias** `b_g = mean(ω_raw)` — at rest, ideal ω = 0
- **Accelerometer bias** `b_a = mean(a_raw) − R_{bw} · g_reaction`
- **Initial orientation** — roll and pitch from gravity direction (yaw = 0, unobservable)

### Pipeline (per sample)

```
raw IMU
  │
  ├─ bias subtraction: a_corr = a_raw − b_a,  ω_corr = ω_raw − b_g
  │
  ├─ 2nd-order Butterworth LPF (biquad, bilinear transform)
  │     gyro:  cutoff 15 Hz
  │     accel: cutoff 10 Hz
  │
  ├─ dead-reckoning integration (attitude + velocity + position)
  │
  └─ publish /imu/processed, /imu/odometry, /imu/pose, /imu/rpy
```

The dead-reckoning output drifts without fusion — it is useful for debugging sensor health
and comparing against the ESKF output.

---

## 7. Error-State Kalman Filter (ESKF)

`EskfNode` fuses bias-corrected IMU measurements (200 Hz) with visual odometry pose
updates (20 Hz).

### State vector

```
x  = [p(3)  v(3)  q(4)  b_a(3)  b_g(3)]    nominal state (16-DOF)
δx = [δp(3) δv(3) δθ(3) δb_a(3) δb_g(3)]   error state   (15-DOF)
```

### Prediction (every IMU sample, dt ≈ 5 ms)

```
a_w   = R·(f − b_a) + g          world-frame acceleration
p    ← p + v·dt + ½·a_w·dt²
v    ← v + a_w·dt
q    ← q ⊗ Exp((ω − b_g)·dt)
P    ← F·P·Fᵀ + Qd
```

Process noise `Qd` is built from IMU noise densities in `euroc_params.yaml`
(accel σ_a, gyro σ_g, accel bias walk σ_ba, gyro bias walk σ_bg).

### Update (every VIO frame, dt ≈ 50 ms)

Innovation vector (6-DOF):

```
z = [ p_meas − p_nom ;  Log(R_nom.T @ R_meas) ]
```

Joseph-form update for numerical stability:

```
S      = H·P·Hᵀ + R_meas
K      = P·Hᵀ·S⁻¹
IKH    = I₁₅ − K·H
P     ← IKH·P·IKHᵀ + K·R_meas·Kᵀ
δx    = K·z   →   inject into nominal state
```

### Gravity estimation

On the first VIO pose the ESKF rotates the mean static accelerometer reading into the VIO
world frame to obtain `g_world`, avoiding reliance on the hardcoded `[0, 0, -9.81]` vector.

---

## 8. Coordinate frames

### Kalibr convention (T_BS)

```
p_body = T_BS · p_sensor
```

`T_BS` (code: `T_b_c0`) maps a point in the **sensor** frame to the **body** frame.

### World frame

Defined as the **initial body frame** (FLU — Forward, Left, Up):

```
+X → forward   (body x at t=0)
+Y → left      (body y at t=0)
+Z → up        (body z at t=0, approximately anti-gravity)
```

Initialising `T_{world,cam0} = T_{bc0}` ensures `T_{world,body}^{0} = I`, so both
odometry and aligned ground truth start at the origin with identity orientation.

### Ground truth alignment

```
T_align              = T_{vicon,body_0}⁻¹
T_{world,body}^GT(t) = T_align · T_{vicon,body}(t)
```

---

## 9. Project structure

```
Visual_Inertial_Odometry/
├── README.md
├── requirements.txt
└── ros2_ws/
    └── src/
        └── vio_pipeline/
            ├── package.xml
            ├── setup.py
            ├── setup.cfg
            ├── config/
            │   └── euroc_params.yaml          # Camera + IMU calibration
            ├── launch/
            │   ├── full_pipeline.launch.py    # Full VIO + ESKF + GT
            │   └── features.launch.py         # Feature visualisation only
            └── vio_pipeline/
                ├── feature_tracking_KLT.py    # Shi-Tomasi + KLT tracker,
                │                              # forward-backward consistency check
                ├── feature_tracking_node.py   # Standalone feature viz node
                ├── vio_node.py                # PoseEstimationNode
                ├── imu_processing_node.py     # Bias removal, LPF, dead-reckoning
                ├── eskf_node.py               # Error-State Kalman Filter
                ├── ground_truth_publisher.py  # Aligned Vicon GT republisher
                └── debug_logger_node.py       # CSV trajectory logger
```

### Node summary

| Node | Executable | Rate | Purpose |
|---|---|---|---|
| `PoseEstimationNode` | `pose_estimation_node` | 20 Hz | KLT tracking → triangulation → PnP → pose |
| `ImuProcessingNode` | `imu_processing_node` | 200 Hz | Bias removal, LPF, dead-reckoning |
| `EskfNode` | `eskf_node` | 200 Hz | IMU + VIO fusion |
| `FeatureTrackingNode` | `feature_tracking_node` | 20 Hz | Feature visualisation (debug) |
| `GroundTruthPublisherNode` | `ground_truth_publisher` | — | Aligned Vicon GT |
| `DebugLoggerNode` | `debug_logger_node` | — | CSV trajectory logger |

---

## 10. Requirements and installation

### System requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 22.04 / 24.04 |
| ROS 2 | Humble | Jazzy |
| Python | 3.10 | 3.11 |
| GPU | — | not required for KLT branch |

No deep-learning dependencies. The KLT branch runs in real-time on CPU.

### 1. Install ROS 2

Follow the [official ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ROS 2 package dependencies

```bash
cd ros2_ws
rosdep install --from-paths src -y --ignore-src
```

### 4. Build and source

```bash
cd ros2_ws
colcon build --packages-select vio_pipeline
source install/setup.bash
```

---

## 11. Running the pipeline

### Prepare the EuRoC bag

Download an EuRoC sequence (e.g. MH_01_easy) and convert to ROS 2 bag format.
Place it at `bags/euroc_mav0/`.

The [euroc_to_rosbag2](scripts/) conversion script is included in the repository.

### Terminal A — play the bag

```bash
ros2 bag play bags/euroc_mav0 --rate 0.5 --clock
```

`--rate 0.5` plays at half speed; remove once timing is confirmed.
`--clock` publishes `/clock` so nodes use simulated time.

### Terminal B — launch the full pipeline

```bash
ros2 launch vio_pipeline full_pipeline.launch.py
```

Override config or timing:

```bash
ros2 launch vio_pipeline full_pipeline.launch.py \
    config_file:=/path/to/euroc_params.yaml \
    use_sim_time:=true
```

### Terminal C — inspect output

```bash
# Live fused pose (ESKF)
ros2 topic echo /eskf/pose

# Compare VIO-only vs fused
ros2 topic hz /vio/pose /eskf/pose

# Plot RPY
ros2 run rqt_plot rqt_plot \
    /eskf/rpy/vector/x \
    /eskf/rpy/vector/y \
    /eskf/rpy/vector/z \
    /imu/rpy/vector/x \
    /imu/rpy/vector/y \
    /imu/rpy/vector/z

# RViz — add Path displays for:
#   /vio/path, /eskf/... path (via odometry), /gt_pub/path
rviz2
```

### Feature visualisation only

```bash
ros2 launch vio_pipeline features.launch.py
# /features/viz         — stereo matches (left | right)
# /features/temporal_viz — KLT motion trails
```

---

## 12. ROS 2 topics and parameters

### Published topics

| Topic | Type | Node | Description |
|---|---|---|---|
| `/vio/pose` | `PoseStamped` | PoseEstimation | Camera-only body pose |
| `/vio/path` | `Path` | PoseEstimation | Camera-only trajectory |
| `/vio/odometry` | `Odometry` | PoseEstimation | Camera-only odometry |
| `/vio/rpy` | `Vector3Stamped` | PoseEstimation | Roll/pitch/yaw in degrees |
| `/eskf/odometry` | `Odometry` | ESKF | Fused pose + velocity |
| `/eskf/pose` | `PoseStamped` | ESKF | Fused pose |
| `/eskf/rpy` | `Vector3Stamped` | ESKF | Fused roll/pitch/yaw in degrees |
| `/imu/processed` | `Imu` | ImuProcessing | Bias-corrected + filtered IMU |
| `/imu/odometry` | `Odometry` | ImuProcessing | IMU dead-reckoning |
| `/imu/pose` | `PoseStamped` | ImuProcessing | IMU dead-reckoning pose |
| `/imu/rpy` | `Vector3Stamped` | ImuProcessing | IMU dead-reckoning RPY (degrees) |
| `/gt_pub/pose` | `PoseStamped` | GroundTruth | Aligned Vicon pose |
| `/gt_pub/path` | `Path` | GroundTruth | Aligned Vicon trajectory |
| `/gt_pub/odometry` | `Odometry` | GroundTruth | Aligned Vicon odometry |
| `/features/viz` | `Image` | FeatureTracking | Stereo match visualisation |
| `/features/temporal_viz` | `Image` | FeatureTracking | KLT motion trails |

### Subscribed topics

| Topic | Type | Subscribers |
|---|---|---|
| `/cam0/image_raw` | `Image` | PoseEstimation, FeatureTracking |
| `/cam1/image_raw` | `Image` | PoseEstimation, FeatureTracking |
| `/imu0` | `Imu` | ImuProcessing, ESKF (raw buffer) |
| `/imu/processed` | `Imu` | ESKF |
| `/vio/odometry` | `Odometry` | ESKF |
| `/gt/pose` (configurable) | `PoseStamped` | GroundTruth |

### PoseEstimationNode parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to `euroc_params.yaml` |
| `min_tracks` | `10` | Minimum circular-checked tracks to attempt PnP |
| `circular_check_threshold` | `2.0` | Max forward-backward pixel error (px) |
| `max_depth` | `30.0` | Max triangulated point depth (m) |
| `min_inlier_ratio` | `0.4` | Min RANSAC inlier fraction |
| `max_translation` | `0.5` | Max accepted per-frame translation (m) |
| `max_rotation_deg` | `30.0` | Max accepted per-frame rotation (deg) |

### ImuProcessingNode parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to `euroc_params.yaml` |
| `imu_topic` | `"/imu0"` | Raw IMU input topic |
| `init_duration` | `2.0` | Static initialisation window (s) |
| `gyro_lpf_cutoff` | `20.0` | Gyro low-pass filter cutoff (Hz) |
| `accel_lpf_cutoff` | `20.0` | Accel low-pass filter cutoff (Hz) |
| `imu_rate_hz` | `200` | Expected IMU sample rate |

### EskfNode parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to `euroc_params.yaml` (IMU noise figures) |
| `meas_pos_std` | `0.05` | VIO position noise 1-σ (m). Lower → trust VIO more |
| `meas_ang_std` | `0.02` | VIO rotation noise 1-σ (rad). Lower → trust VIO more |
| `init_pos_std` | `1.0` | Initial position uncertainty (m) |
| `init_vel_std` | `0.5` | Initial velocity uncertainty (m/s) |
| `init_att_std` | `0.1` | Initial attitude uncertainty (rad, ≈5.7°) |
| `init_ba_std` | `0.02` | Initial accel-bias uncertainty (m/s²) |
| `init_bg_std` | `5e-4` | Initial gyro-bias uncertainty (rad/s) |
| `max_dt` | `0.1` | Max IMU dt before sample is discarded (s) |

---

## 13. Configuration file

`config/euroc_params.yaml` stores the EuRoC VI-Sensor calibration used by all nodes:

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
  gyro_noise:  1.6968e-04               # rad/s/√Hz   — process noise Q
  gyro_walk:   1.9393e-05               # rad/s²/√Hz
  accel_noise: 2.0000e-03               # m/s²/√Hz
  accel_walk:  3.0000e-03               # m/s³/√Hz
  rate_hz: 200
  T_BS: [identity]                       # IMU = body for EuRoC

gravity: [0.0, 0.0, -9.81]              # overridden by ESKF gravity estimator
```
