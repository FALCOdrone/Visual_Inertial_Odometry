# Visual-Inertial Odometry Pipeline

A ROS 2 stereo visual-inertial odometry system that fuses **Shi-Tomasi + KLT optical flow** feature tracking with either an **Error-State Kalman Filter (ESKF)** or a **sliding-window Factor Graph Optimization (FGO) backend** for 6-DOF pose estimation. The pipeline processes synchronized stereo images and IMU data at 200 Hz, with optional simulated GPS streams (`/gps/fix`, `/gps/enu`). It is evaluated on the [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) dataset using Absolute Trajectory Error (ATE) with Umeyama SE(3) alignment and Relative Pose Error (RPE) over multiple segment lengths.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Feature Tracking -- Shi-Tomasi + KLT](#2-feature-tracking----shi-tomasi--klt)
3. [Stereo Triangulation and PnP-RANSAC](#3-stereo-triangulation-and-pnp-ransac)
4. [IMU Processing](#4-imu-processing)
5. [Error-State Kalman Filter (ESKF)](#5-error-state-kalman-filter-eskf)
6. [Factor Graph Optimization (FGO) Backend](#6-factor-graph-optimization-fgo-backend)
7. [GPS Fusion](#7-gps-fusion)
8. [Coordinate Frames](#8-coordinate-frames)
9. [Trajectory Evaluation](#9-trajectory-evaluation)
10. [Project Structure](#10-project-structure)
11. [Installation](#11-installation)
12. [Running the Pipeline](#12-running-the-pipeline)
13. [ROS 2 Topics and Parameters](#13-ros-2-topics-and-parameters)
14. [Configuration Files](#14-configuration-files)

---

## 1. Architecture Overview

The pipeline supports two fusion backends, selected at launch time with `use_fgo:=true|false`:

- **ESKF (default):** Extended Kalman Filter operating at IMU rate (200 Hz) with VIO corrections at ~20 Hz.
- **FGO (WIP):** Sliding-window factor graph with IMU preintegration and VO relative-pose factors, optimized via Levenberg-Marquardt at keyframe rate (~20 Hz).

When `use_fgo:=true`, the ESKF node is not launched; the FGO backend replaces it.

```
                          EuRoC / UZH Bag
  /cam0/image_raw (20 Hz)   /cam1/image_raw (20 Hz)   /imu0 (200 Hz)   /gt/pose
         |                          |                       |               |
         v                          v                       |               |
  +----------------------------------+                      |               |
  | StereoRectifierNode (optional)   |                      |               |
  | undistort + rectify stereo pair  |                      |               |
  +------+---------------------------+                      |               |
         |                          |                       |               |
         |   ApproximateTimeSynchronizer                     |               |
         v                          v                       v               v
  +------------------------------------+    +------------------------+  +---------------------------+
  |     PoseEstimationNode (20 Hz)     |    | ImuProcessingNode      |  | GroundTruthPublisherNode  |
  |                                    |    |          (200 Hz)      |  |                           |
  |  Shi-Tomasi corner detection       |    | static bias init       |  | Vicon -> VIO world align  |
  |  KLT temporal tracking (L->L)     |    | Butterworth LPF        |  +------+--------------------+
  |  + forward-backward check         |    | dead-reckoning         |         |
  |  KLT stereo matching (L->R)       |    +------+-----------------+   /gt_pub/pose
  |  + forward-backward check         |           |                    /gt_pub/path
  |  undistort keypoints              |    /imu/processed (200 Hz)     /gt_pub/odometry
  |  stereo triangulation             |           |                         |
  |  PnP-RANSAC + LM refinement       |           v                         v
  |  keyframe selection               |                                +---------------------------+
  |  dynamic pose covariance          |    +========================+  | GpsSimulatorNode          |
  |  pose accumulation                |    ||  ESKF  OR  FGO       ||  |  (optional, ~5-20 Hz)     |
  +------+-----------------------------+    ||  (select at launch)  ||  |                           |
         |                                 |========================|  | /gt_pub/pose -> corrupt   |
  /vio/pose                                |                        |  | -> /gps/fix, /gps/enu    |
  /vio/path                                | ESKF: IMU pred 200 Hz  |  +---------------------------+
  /vio/odometry  -------->                 |   VIO update    20 Hz  |
  /vio/rpy                                 |   GPS update    ~5 Hz  |<-- /gps/enu (ESKF only)
                                           |   Joseph-form cov      |
                                           | -or-                   |
                                           | FGO: IMU preintegration|
                                           |   VO relative factors  |
                                           |   LM optimization      |
                                           |   Schur marginalization|
                                           +------+-----------------+
                                                  |
                                           /eskf/odometry (200 Hz)    -- ESKF mode
                                           /fgo/odometry  (~20 Hz)    -- FGO mode
                                           /{eskf,fgo}/pose
                                           /{eskf,fgo}/path
                                                  |
                                                  v
                              +-------------------------------+     +-------------------------+
                              | TfPublisherNode (optional)    |     | DebugLoggerNode         |
                              |                               |     |                         |
                              | /tf_static: base_link->cam0,  |     | Writes CSV logs:        |
                              |             base_link->cam1,  |     |   imu_raw.csv           |
                              |             base_link->imu0   |     |   imu_processed.csv     |
                              | /tf: map->base_link           |     |   pose_vio.csv          |
                              +-------------------------------+     |   pose_eskf.csv         |
                                                                    |   pose_imu_dr.csv       |
                                                                    |   pose_gt.csv           |
                                                                    +-------------------------+
```

---

## 2. Feature Tracking -- Shi-Tomasi + KLT

The feature frontend is implemented in `feature_tracking_KLT.py` as the `FeatureExtractor` class. It performs corner detection, temporal tracking, and stereo matching -- all validated by a forward-backward circular consistency check.

### 2.1 Shi-Tomasi Corner Detection

Corners are detected using `cv2.goodFeaturesToTrack`, which evaluates the **structure tensor** (also called the second-moment matrix or autocorrelation matrix) at each pixel. For a patch centered at pixel `(x, y)`:

```
        [ sum(Ix^2)    sum(Ix*Iy) ]
M(x,y) = [                          ]
        [ sum(Ix*Iy)   sum(Iy^2)  ]
```

where `Ix` and `Iy` are the image gradients computed over a `blockSize x blockSize` window (default: 7x7). The summation is weighted by a box or Gaussian kernel.

The **Shi-Tomasi score** is the minimum eigenvalue of M:

```
score(x, y) = lambda_min(M)
```

This differs from the Harris detector, which uses `det(M) - k * trace(M)^2`. The Shi-Tomasi criterion directly measures how well-conditioned the corner is for tracking -- a high minimum eigenvalue means the gradient pattern is distinctive in all directions.

**Acceptance criteria:**

1. **Quality gate:** A candidate is accepted only if `score >= qualityLevel * max_score_in_image`. With `qualityLevel = 0.5`, only corners at least half as strong as the best corner survive.
2. **Non-maximum suppression:** Among accepted candidates, corners closer than `minDistance` pixels are suppressed, keeping only the strongest. This ensures spatial spread across the image.
3. **Count cap:** At most `maxCorners` corners are returned.

**Re-detection strategy:** When the number of successfully tracked features drops below `max_corners // 2` (i.e., below 50 with the default of 100), fresh Shi-Tomasi corners are detected on the current left image. An **exclusion mask** prevents duplicates: circles of radius `min_distance` are drawn around every existing tracked point, and detection is restricted to the unmasked region. New detections are appended up to the `max_corners` cap.

#### Shi-Tomasi Parameter Table

| Parameter | Code name | Default | Description |
|---|---|---|---|
| Max corners | `maxCorners` | 100 | Maximum corners detected per frame |
| Quality level | `qualityLevel` | 0.5 | Minimum score as fraction of best score |
| Min distance | `minDistance` | 20 px | Minimum pixel gap between accepted corners |
| Block size | `blockSize` | 7 | Structure tensor window size (pixels) |

### 2.2 KLT Lucas-Kanade Tracking

Temporal correspondences are established by `cv2.calcOpticalFlowPyrLK`, which minimises the **sum-of-squared-differences (SSD) photometric error** between a template patch in the previous frame and a search patch in the current frame:

```
E(d) = sum_{(x,y) in W} [ I_prev(x, y) - I_curr(x + dx, y + dy) ]^2
```

where `W` is the window of size `winSize` and `d = (dx, dy)` is the displacement vector to be found.

The minimisation uses a first-order Taylor expansion of the image brightness:

```
I_curr(x + dx, y + dy) ~ I_curr(x, y) + Ix*dx + Iy*dy
```

Leading to the normal equations:

```
[ sum(Ix^2)    sum(Ix*Iy) ] [ dx ]   [ sum(Ix * dI) ]
[                          ] [    ] = [               ]
[ sum(Ix*Iy)   sum(Iy^2)  ] [ dy ]   [ sum(Iy * dI) ]
```

where `dI = I_prev(x,y) - I_curr(x,y)`. This is identical to the structure tensor M, which is why Shi-Tomasi corners (high `lambda_min(M)`) track well -- the system is well-conditioned.

**Gaussian pyramid:** To handle large inter-frame displacements, KLT operates on a coarse-to-fine image pyramid. At each level the image is downsampled by 2x, so the coarsest level `L` can capture displacements up to approximately `winSize * 2^L` pixels. The displacement estimate from a coarser level is propagated to the next finer level as an initial guess.

**Termination criteria:** The iterative refinement at each level stops when either the displacement update is smaller than `epsilon = 0.01` pixels, or `maxCount = 30` iterations are reached.

#### KLT Parameter Table

| Parameter | Code name | Default | Description |
|---|---|---|---|
| Window size | `winSize` | (21, 21) | Patch size for photometric matching |
| Pyramid depth | `maxLevel` | 3 | Number of pyramid levels (0 = no pyramid) |
| Max iterations | `criteria.maxCount` | 30 | Iteration limit per pyramid level |
| Epsilon | `criteria.epsilon` | 0.01 | Convergence threshold (pixels) |

### 2.3 Forward-Backward Circular Consistency Check

Raw KLT tracks are unreliable near occlusion boundaries, textureless regions, and motion blur. Each track is validated with a **circular consistency check** that measures round-trip reprojection error:

```
1. Forward track:   p1 = KLT(img0 -> img1, p0)
2. Backward track:  p0_hat = KLT(img1 -> img0, p1)
3. Forward-backward error:  fb_err = ||p0 - p0_hat||_2
4. Accept if:  status_fwd AND status_bwd AND (fb_err < threshold)
```

The default threshold is `pixel_threshold = 2.0` pixels. Both the forward and backward KLT must report success (status flag) AND the round-trip error must be below threshold.

This check is applied **twice per frame**:

1. **Temporal matches** (left_{t-1} -> left_t): validates inter-frame feature tracks.
2. **Stereo matches** (left_t -> right_t): validates left-right correspondences for triangulation.

Only features that pass **both** checks (temporal AND stereo) in the previous frame produce the `circular_tracks` output that feeds into triangulation and PnP. This yields index-aligned quadruples:

```
kpts_l_prev (N, 2)    kpts_l_curr (N, 2)
kpts_r_prev (N, 2)    [for triangulation in the previous stereo pair]
```

---

## 3. Stereo Triangulation and PnP-RANSAC

Implemented in `vio_node.py` as `PoseEstimationNode`. This node subscribes to synchronized stereo images, runs the feature tracker, and estimates frame-to-frame pose via a triangulation-then-PnP pipeline.

### 3.1 Stereo Rectification

The `StereoRectifierNode` (enabled by default with `use_rectifier:=true`) applies `cv2.stereoRectify` and `cv2.initUndistortRectifyMap` to produce row-aligned, undistorted stereo images published on `/cam0/image_rect` and `/cam1/image_rect`. The rectification maps are computed once at startup from the cam0/cam1 intrinsics, distortion coefficients, and the extrinsic transform `T_c1_c0`.

When rectification is active, the downstream `PoseEstimationNode` subscribes to the rectified topics and can enforce an **epipolar constraint** on stereo matches: matched keypoints with vertical disparity exceeding `max_epipolar_err` (default: 2.0 px) are rejected.

### 3.2 Undistortion

Keypoints are detected on the original (distorted) images for tracking robustness. Before any geometric computation, all keypoints are mapped to ideal (undistorted) pixel coordinates using the **radial-tangential distortion model**:

```
r^2 = x_n^2 + y_n^2       (where x_n, y_n are normalised image coordinates)

x_distorted = x_n * (1 + k1*r^2 + k2*r^4) + 2*p1*x_n*y_n + p2*(r^2 + 2*x_n^2)
y_distorted = y_n * (1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*y_n^2) + 2*p2*x_n*y_n
```

The undistortion is computed via `cv2.undistortPoints(pts, K, dist, P=K)`, which inverts the distortion model and reprojects back to pixel coordinates using the same intrinsic matrix K.

Separate intrinsics and distortion coefficients are used for cam0 and cam1 as specified in the config YAML.

### 3.3 Projection Matrices

The stereo geometry is defined by the relative transform `T_c1_c0` from cam0 to cam1, derived from the Kalibr extrinsics:

```
T_c1_c0 = inv(T_b_c1) @ T_b_c0
```

The 3x4 projection matrices are:

```
P0 = K0 @ [I_3 | 0]              (cam0 is the reference frame)
P1 = K1 @ [R_c1_c0 | t_c1_c0]    (cam1 relative to cam0)
```

### 3.4 DLT Triangulation

For each stereo-matched pair `(p_L, p_R)` of undistorted 2D keypoints, the 3D point `X` in the cam0 frame is computed using `cv2.triangulatePoints`:

```
A @ X = 0

where A is built from:
  row 0: p_L[0] * P0[2,:] - P0[0,:]
  row 1: p_L[1] * P0[2,:] - P0[1,:]
  row 2: p_R[0] * P1[2,:] - P1[0,:]
  row 3: p_R[1] * P1[2,:] - P1[1,:]
```

The solution is the right singular vector of A corresponding to the smallest singular value. The result is converted from homogeneous coordinates: `X_3D = X[0:3] / X[3]`.

**Depth filtering:** Points with `Z < 0.1 m` (behind camera or degenerate) or `Z > max_depth` (default 30 m, noisy at large baselines) are rejected before PnP.

### 3.5 PnP-RANSAC with Layered Outlier Rejection

Given N 3D landmarks `{X_i}` triangulated from the **previous** stereo pair and their 2D observations `{p_i}` in the **current** left image, the perspective-n-point problem solves for the relative camera pose:

```
lambda_i * p_i = K @ (R @ X_i + t)
```

**Step 1 -- EPnP + RANSAC:**

```python
cv2.solvePnPRansac(
    pts3d, pts2d, K, None,
    iterationsCount=200,
    reprojectionError=2.0,      # pixels
    confidence=0.999,
    flags=cv2.SOLVEPNP_EPNP
)
```

EPnP (Efficient PnP) parametrises the 3D points as a weighted sum of 4 virtual control points, reducing the problem to a low-dimensional eigenvalue computation. RANSAC draws minimal subsets, scores by reprojection error, and returns the inlier set.

**Step 2 -- Levenberg-Marquardt refinement:**

```python
cv2.solvePnP(
    pts3d[inliers], pts2d[inliers], K, None,
    rvec, tvec,
    useExtrinsicGuess=True,
    flags=cv2.SOLVEPNP_ITERATIVE
)
```

Using the RANSAC solution as initial guess, the iterative solver minimises reprojection error over all inliers using Levenberg-Marquardt nonlinear least squares.

**Layered outlier rejection** (checked sequentially, any failure skips the frame):

| Guard | Condition | Default | Purpose |
|---|---|---|---|
| Depth filter | `0.1 m < Z < max_depth` | `max_depth = 30.0` m | Reject degenerate triangulations |
| Minimum count | `N_valid >= 6` | 6 points | PnP needs >= 4; extra for robustness |
| PnP success | `solvePnPRansac` returns `ok` and `inliers` | -- | Degenerate geometry |
| Inlier count | `n_inliers >= 6` | 6 | Insufficient consensus |
| Inlier ratio | `n_inliers / N >= min_inlier_ratio` | 0.4 | Corrupt RANSAC consensus |
| Max translation | `||t|| <= max_translation` | 0.5 m/frame | Reject pose jumps |
| Max rotation | `angle <= max_rotation_deg` | 30.0 deg/frame | Reject orientation flips |

### 3.6 Keyframe Selection

Not every VIO frame triggers a backend update. The `PoseEstimationNode` publishes a pose only when at least one of these conditions is met since the last keyframe:

| Gate | Threshold | Default |
|---|---|---|
| Translation | accumulated Euclidean distance >= threshold | `kf_min_translation = 0.03` m |
| Rotation | accumulated angular change >= threshold | `kf_min_rotation_deg = 2.0` deg |
| Frame count | frames since last keyframe >= max | `kf_max_frames = 5` (0.25 s at 20 Hz) |

This reduces redundant updates during stationary periods while ensuring the filter does not starve during slow motion.

### 3.7 Dynamic Pose Covariance

The `PoseEstimationNode` fills the `pose.covariance` field of its `Odometry` message based on the RANSAC inlier ratio:

```
sigma = base_std / inlier_ratio
covariance_diag = sigma^2
```

At a typical inlier ratio of ~0.6, this yields per-axis position uncertainty of ~0.05 m (matching `meas_pos_std`). Both the ESKF and FGO backends extract this covariance from the incoming VIO message to weight updates appropriately.

### 3.8 Pose Accumulation

`solvePnPRansac` returns `T_curr_prev` (the current camera pose relative to the previous camera frame). The world-frame poses are accumulated as:

```
T_world_cam0(t) = T_world_cam0(t-1) @ inv(T_curr_prev)

T_world_body(t) = T_world_cam0(t) @ inv(T_b_c0)
```

The initial `T_world_cam0` is set to `T_body_to_ros @ T_b_c0`, where `T_body_to_ros` is a fixed rotation mapping the EuRoC body frame (x-up, y-right, z-forward) to the ROS FLU convention (x-forward, y-left, z-up).

---

## 4. IMU Processing

Implemented in `imu_processing_node.py` as `ImuProcessingNode`. This node preprocesses raw IMU data before ESKF/FGO fusion.

### 4.1 Static Bias Initialisation

On startup the node collects `init_duration` seconds of static IMU data (default: 2.0 s, overridden to 0.0 s in the launch file). During this window:

**Gyroscope bias:**
```
b_g = mean(omega_raw)
```
At rest the angular velocity should be zero; any nonzero mean is bias.

**Initial orientation** (roll + pitch from gravity, yaw = 0):
```
g_hat = mean(a_raw) / ||mean(a_raw)||
q_init = rotation that maps g_hat to [0, 0, 1]
```
Yaw is unobservable from the accelerometer alone.

**Accelerometer bias:**
```
b_a = mean(a_raw) - R_bw @ [0, 0, +|g|]^T
```
where `R_bw = R_wb^T` is the body-to-world rotation from the initial orientation estimate, and `|g| = 9.81 m/s^2`.

### 4.2 Butterworth Low-Pass Filter

A **2nd-order Butterworth biquad** filter is applied independently to each axis of gyro and accelerometer data. The filter coefficients are derived analytically via the bilinear transform with frequency pre-warping:

```
K    = tan(pi * fc / fs)          (pre-warped analog cutoff)
norm = 1 + sqrt(2)*K + K^2
b0   = K^2 / norm,   b1 = 2*b0,   b2 = b0
a1   = 2*(K^2 - 1) / norm
a2   = (1 - sqrt(2)*K + K^2) / norm
```

The filter is implemented as Direct Form II:
```
w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
```

Delay states are seeded from the mean of the static init window to avoid startup transient.

**Cutoff frequencies** (as set in `pipeline_params.yaml`):

| Signal | Cutoff | IMU rate | Purpose |
|---|---|---|---|
| Gyroscope | 15 Hz | 200 Hz | Remove vibration noise |
| Accelerometer | 10 Hz | 200 Hz | Remove motor vibration |

### 4.3 Dead-Reckoning Pipeline

After bias subtraction and filtering, each sample is integrated:

```
Orientation:  q <- q * Exp(omega * dt)       (SO(3) exponential map)
Acceleration: a_world = R_wb @ f_body + g     (gravity removal)
Velocity:     v <- v + a_world * dt
Position:     p <- p + v*dt + 0.5*a_world*dt^2
```

This dead-reckoning output drifts without fusion. It is published on `/imu/odometry` for debugging sensor health.

---

## 5. Error-State Kalman Filter (ESKF)

Implemented in `eskf_node.py` as `EskfNode`. Fuses bias-corrected IMU measurements (200 Hz) with visual odometry pose updates (~20 Hz) and optional GPS position updates (~5-20 Hz). This is the **default** fusion backend; use `use_fgo:=true` to switch to the FGO backend instead.

### 5.1 State Vector

The ESKF maintains a **nominal state** and a separate **error state**:

```
Nominal state (16-DOF):
  x = [ p(3)   v(3)   q(4)    b_a(3)   b_g(3) ]
       position velocity quaternion accel-bias gyro-bias

Error state (15-DOF):
  dx = [ dp(3)  dv(3)  dtheta(3)  db_a(3)  db_g(3) ]
        position velocity  SO(3)     accel-bias gyro-bias
```

The quaternion (4 parameters with 1 constraint) is replaced by a 3-DOF rotation error `dtheta` in the error state, giving 15 free parameters. The error-state covariance P is 15x15.

### 5.2 IMU Prediction Step (every IMU sample, dt ~ 5 ms)

**Nominal state propagation:**

The `/imu/processed` measurements have already had the static initial biases removed by `ImuProcessingNode`. The ESKF's own `b_a` / `b_g` states track **residual** slow drift on top of that correction.

```
f_corr = accel - b_a          (corrected specific force, body frame)
w_corr = gyro  - b_g          (corrected angular rate, body frame)

a_world = R @ f_corr + g      (world-frame true acceleration)

p <- p + v*dt + 0.5*a_world*dt^2
v <- v + a_world*dt
q <- q * Exp(w_corr * dt)     (quaternion multiplication with SO(3) exponential)
q <- q / ||q||                (re-normalise)
b_a, b_g: unchanged
```

where `g` is the gravity vector in the world frame (estimated from static IMU data at initialisation, or `[0, 0, -9.81]` from config).

**Error-state transition matrix F (15x15):**

```
F = I_15 + Fc * dt

where Fc (continuous-time Jacobian):

  Fc[0:3, 3:6]   =  I_3            dp_dot = dv
  Fc[3:6, 6:9]   = -R @ [f_corr x] dv_dot contribution from dtheta
  Fc[3:6, 9:12]  = -R              dv_dot contribution from db_a
  Fc[6:9, 6:9]   = -[w_corr x]     dtheta_dot contribution from dtheta
  Fc[6:9, 12:15] = -I_3            dtheta_dot contribution from db_g
```

Here `[v x]` denotes the 3x3 skew-symmetric matrix of vector `v`.

**Discrete process noise Qd (15x15):**

```
Qd = Qc * dt       (first-order hold approximation)

Qd[3:6,   3:6]   = sigma_a^2  * dt * I_3    (accel white noise -> velocity)
Qd[6:9,   6:9]   = sigma_g^2  * dt * I_3    (gyro white noise -> attitude)
Qd[9:12,  9:12]  = sigma_ba^2 * dt * I_3    (accel bias random walk)
Qd[12:15, 12:15] = sigma_bg^2 * dt * I_3    (gyro bias random walk)
```

The noise densities `sigma_a`, `sigma_g`, `sigma_ba`, `sigma_bg` are loaded from the config YAML (e.g., for EuRoC: `sigma_a = 2.0e-3`, `sigma_g = 1.6968e-4`, `sigma_ba = 3.0e-3`, `sigma_bg = 1.9393e-5`).

**Covariance propagation:**
```
P <- F @ P @ F^T + Qd
```

### 5.3 VIO Update Step (every VIO frame, dt ~ 50 ms)

**Innovation vector (6-DOF):**

```
z_p = p_meas - p_nom                           (position error, 3-DOF)
z_theta = Log(R_nom^T @ R_meas)                (rotation error as rotation vector, 3-DOF)
z = [ z_p ; z_theta ]                          (6x1)
```

The rotation error is computed in the **body frame** via the SO(3) logarithmic map:

```
dR = R_nom^T @ R_meas
cos(theta) = clip((trace(dR) - 1) / 2, -1, 1)
theta = arccos(cos(theta))
phi = (theta / (2*sin(theta))) * [dR[2,1]-dR[1,2]; dR[0,2]-dR[2,0]; dR[1,0]-dR[0,1]]
```

**Measurement Jacobian H (6x15):**

```
H = [ I_3  0_3  0_3  0_3  0_3 ]    (rows 0-2: position)
    [ 0_3  0_3  I_3  0_3  0_3 ]    (rows 3-5: orientation error state)
```

This structure means position observations affect `dp` (columns 0-2) and orientation observations affect `dtheta` (columns 6-8).

**Measurement noise R (6x6):**

When the incoming VIO message carries nonzero covariance (dynamic pose covariance from section 3.7), that covariance is used directly. Otherwise:

```
R = diag([ meas_pos_std^2,  meas_pos_std^2,  meas_pos_std^2,
           meas_ang_std^2,  meas_ang_std^2,  meas_ang_std^2 ])
```

**Joseph-form covariance update** (numerically stable):

```
S   = H @ P @ H^T + R              (6x6 innovation covariance)
K   = P @ H^T @ inv(S)             (15x6 Kalman gain)
dx  = K @ z                        (15x1 error-state estimate)

IKH = I_15 - K @ H
P   <- IKH @ P @ IKH^T + K @ R @ K^T
P   <- (P + P^T) / 2               (enforce exact symmetry)
```

The Joseph form `IKH @ P @ IKH^T + K @ R @ K^T` is preferred over the standard `(I - KH) @ P` because it guarantees positive semi-definiteness even with finite-precision arithmetic.

**Nominal state injection:**

```
p   <- p + dx[0:3]
v   <- v + dx[3:6]
q   <- q * Exp(dx[6:9]),  then normalise
b_a <- b_a + dx[9:12]
b_g <- b_g + dx[12:15]
```

### 5.4 GPS Update Step (3-DOF, position only)

**Innovation:**
```
z = p_gps - p_nom       (3x1, ENU position error)
```

**Measurement Jacobian H_gps (3x15):**
```
H_gps = [ I_3  0_3  0_3  0_3  0_3 ]    (position rows only)
```

**Measurement noise R_gps (3x3):**
```
R_gps = diag([ gps_pos_std_h^2,  gps_pos_std_h^2,  gps_pos_std_v^2 ])
```

**Mahalanobis gating (chi-squared test):**

Before applying the GPS update, the Normalised Innovation Squared (NIS) is computed:

```
S   = H_gps @ P @ H_gps^T + R_gps     (3x3)
NIS = z^T @ inv(S) @ z                 (scalar)
```

If `NIS > gps_gate_chi2`, the GPS sample is rejected. This suppresses multipath spikes and outage recovery fixes that would corrupt the velocity and attitude states. The chi-squared thresholds for 3 degrees of freedom:

| Confidence | chi^2 threshold |
|---|---|
| 95% | 7.815 |
| 99% | 11.345 |

The default gate is `7.815` (standalone mode) or `0.0` (disabled, RTK mode -- see section 7).

If the gate passes, the standard Joseph-form update is applied with all 15 error states injected (position, velocity, attitude, and biases).

### 5.5 Gravity Estimation

On the first VIO pose, the ESKF estimates the gravity vector in the VIO world frame from the raw `/imu0` accelerometer data collected during the static init window:

```
f_static = mean(raw_accel_buffer)           (body frame)
g_world  = R_init @ (-f_static)            (rotate to VIO world frame)
```

where `R_init` is the rotation matrix from the first VIO quaternion. This avoids reliance on the hardcoded `[0, 0, -9.81]` assumption and handles arbitrary sensor orientations. Up to 2000 raw samples (10 s at 200 Hz) are buffered for this purpose.

---

## 6. Factor Graph Optimization (FGO) Backend

Status: **Work in progress (WIP)**.

Implemented across three modules:

- `fgo_backend_node.py` -- ROS 2 node that orchestrates keyframe management and publishes optimized poses
- `imu_preintegrator.py` -- on-manifold IMU preintegration (Forster et al., TRO 2017)
- `factor_graph.py` -- sliding-window factor graph with LM optimization and Schur-complement marginalization
- `so3_utils.py` -- shared SO(3) rotation utilities (exponential/logarithmic maps, Jacobians, quaternion conversions)

The FGO backend replaces the ESKF when launched with `use_fgo:=true`. It subscribes to the same topics (`/imu/processed`, `/vio/odometry`, `/imu0` for gravity estimation) and publishes on `/fgo/odometry`, `/fgo/pose`, `/fgo/path`. Because this backend is still WIP, expect interface/behavior changes.

### 6.1 Keyframe State

Each keyframe node in the factor graph carries a 15-DOF state on the SO(3) x R^12 manifold:

```
x_k = [ R(3x3)   p(3)   v(3)   b_a(3)   b_g(3) ]
       rotation  position velocity accel-bias gyro-bias
```

Updates are applied via the **retract** operation (manifold oplus):

```
R_new   = R @ Exp(delta_theta)
p_new   = p + delta_p
v_new   = v + delta_v
b_a_new = b_a + delta_ba
b_g_new = b_g + delta_bg
```

### 6.2 IMU Preintegration

Between consecutive keyframes, all IMU measurements are preintegrated into compact rotation, velocity, and position deltas. This avoids re-integrating the full IMU stream during optimization.

**Preintegrated measurements** (accumulated in the body frame of keyframe `i`):

```
delta_R_ij = prod_k Exp((omega_k - b_g) * dt_k)
delta_v_ij = sum_k delta_R_ik @ (a_k - b_a) * dt_k
delta_p_ij = sum_k delta_v_ik * dt_k + 0.5 * delta_R_ik @ (a_k - b_a) * dt_k^2
```

**Covariance propagation (9x9):** The preintegrator tracks the uncertainty of `[delta_theta, delta_v, delta_p]` using a first-order discrete propagation:

```
F_d = I_9 + A * dt       (discrete transition matrix)
Sigma <- F_d @ Sigma @ F_d^T + B @ Q_d @ B^T
```

where `A` contains the continuous-time Jacobians (rotation-velocity coupling via `skew(f_corr)`) and `Q_d = diag(sigma_g^2, sigma_a^2) * dt`.

**First-order bias correction:** Rather than re-preintegrating when biases change during optimization, the bias Jacobians `d_R/d_bg`, `d_v/d_ba`, `d_v/d_bg`, `d_p/d_ba`, `d_p/d_bg` are accumulated during integration and used for a linear correction:

```
delta_R_corr = delta_R @ Exp(d_R_d_bg @ db_g)
delta_v_corr = delta_v + d_v_d_ba @ db_a + d_v_d_bg @ db_g
delta_p_corr = delta_p + d_p_d_ba @ db_a + d_p_d_bg @ db_g
```

**IMU noise scaling:** Following the practice of VINS-Mono and other optimization-based VIO systems, the IMU white noise is scaled by `imu_noise_scale` (default: 20x) before preintegration. Datasheet noise densities are too tight for the discrete-time integration approximation, causing IMU factors to dominate over VO factors in the optimization.

### 6.3 Factor Types

**IMU preintegration factor** (15-dimensional residual connecting keyframes `i` and `j`):

```
r_R   = Log(delta_R_corr^T @ R_i^T @ R_j)
r_v   = R_i^T @ (v_j - v_i - g*dt) - delta_v_corr
r_p   = R_i^T @ (p_j - p_i - v_i*dt - 0.5*g*dt^2) - delta_p_corr
r_ba  = b_a_j - b_a_i
r_bg  = b_g_j - b_g_i
```

The information matrix is block-diagonal: `[inv(Sigma_preint)(9x9), info_ba(3x3), info_bg(3x3)]`, where the bias blocks encode random-walk uncertainty `1/(sigma_b^2 * dt)`.

**VO relative-pose factor** (6-dimensional residual):

```
r_R = Log(dR_meas^T @ R_i^T @ R_j)
r_p = dR_meas^T @ (R_i^T @ (p_j - p_i) - dp_meas)
```

The information matrix is extracted from the VIO message covariance (dynamic, based on inlier ratio) or falls back to `diag(vo_pos_std^2, vo_ang_std^2)`.

**Prior factor:** On the first keyframe, a diagonal prior constrains all 15 state dimensions. After marginalization, the prior becomes a dense linearized factor from the Schur complement (see section 6.5).

### 6.4 Levenberg-Marquardt Optimization

The graph is optimized by solving the normal equations at each iteration:

```
(H + lambda * diag(H)) @ delta = -b

where H = sum_factors J^T @ Omega @ J     (Hessian approximation)
      b = sum_factors J^T @ Omega @ r     (gradient)
```

The solver features:
- **Adaptive damping:** `lambda` is halved on cost decrease, quintupled on increase, clamped to `[1e-8, 1e6]`.
- **Per-component step clamping:** rotation limited to 0.1 rad, position to 1.0 m, velocity to 2.0 m/s, biases to 0.01/0.001 per iteration. This prevents divergence from large initial guess errors.
- **Convergence criterion:** `||delta|| < 1e-6` or `lm_max_iter` iterations reached (default: 8).

For a window of W=10 keyframes, the system is 150x150 and solved via dense `numpy.linalg.solve`.

### 6.5 Schur-Complement Marginalization

When the window exceeds `window_size` keyframes, the oldest keyframe is marginalized:

```
| H_mm  H_mr | | dx_m |   | b_m |
| H_rm  H_rr | | dx_r | = | b_r |

H_prior = H_rr - H_rm @ inv(H_mm) @ H_mr
b_prior = b_r  - H_rm @ inv(H_mm) @ b_m
```

The prior Hessian is factored via eigendecomposition `H_prior = V @ diag(lambda) @ V^T` into a square-root form `J_prior = diag(sqrt(lambda)) @ V^T`, with a corresponding residual `r0 = J_prior^{-T} @ b_prior`. This linearized prior is re-evaluated at each optimization step relative to its linearization point using the manifold `local()` operator.

### 6.6 Gravity Estimation

Like the ESKF, the FGO backend estimates gravity from raw `/imu0` accelerometer data collected before the first VIO keyframe. It waits for `min_gravity_samples` (default: 100) raw IMU samples, then computes:

```
g_world = R_first_vio @ (-mean(raw_accel))
```

---

## 7. GPS Fusion

### 7.1 GPS Simulator (`gps_simulator_node.py`)

The `GpsSimulatorNode` creates realistic GPS measurements by corrupting the ground-truth position with a multi-component error model:

```
p_gps = p_true + bias + white_noise + multipath
```

**Error sources:**

| Component | Model | Parameters |
|---|---|---|
| White noise | `N(0, diag(sigma_h^2, sigma_h^2, sigma_v^2))` iid per sample | `noise_h_m`, `noise_v_m` |
| Bias drift | Gauss-Markov: `b_dot = -b/tau + q_b * xi` | `bias_time_const_s`, `bias_walk_h_m_s`, `bias_walk_v_m_s` |
| Multipath | Sporadic `N(0, (scale*sigma_h)^2)` with probability `p_multi` | `multipath_prob`, `multipath_scale` |
| Outages | Poisson-distributed signal loss events | `outage_prob_per_s`, `outage_duration_s` |

The Gauss-Markov bias is discretised as:
```
alpha = exp(-dt / tau)
bias[k+1] = alpha * bias[k] + q * sqrt(dt) * randn()
```

The corrupted ENU position is converted to WGS-84 latitude/longitude/altitude using ECEF as an intermediate frame, then published as `NavSatFix` on `/gps/fix` and as `PointStamped` on `/gps/enu`.

During outages, a `NavSatFix` with `STATUS_NO_FIX` and NaN coordinates is published.

### 7.2 GPS Profiles (`gps_profiles.yaml`)

Two pre-configured profiles model different receiver grades (based on Septentrio mosaic-G5 datasheet):

| Profile | Horizontal 1-sigma | Vertical 1-sigma | Update rate | Gate |
|---|---|---|---|---|
| `standalone` | 1.2 m | 1.9 m | 20 Hz | chi^2(3, 0.95) = 7.815 |
| `rtk` | 0.01 m (~1 cm) | 0.02 m (~2 cm) | 20 Hz | Disabled (0.0) |

Each profile has two sections:
- **`simulator`**: noise parameters injected by `GpsSimulatorNode`
- **`eskf`**: measurement noise assumed by the ESKF (slightly inflated above simulator noise to absorb residual Gauss-Markov bias)

The RTK profile disables the innovation gate because the EKF covariance converges to ~0.0025 m^2 after VIO updates, making the gate overly restrictive for valid cm-level corrections.

GPS fusion is implemented only in ESKF mode (the FGO backend does not include GPS factors).

Current launch behavior note: in `ros2_ws/src/vio_pipeline/launch/full_pipeline.launch.py`, the ESKF parameter `use_gps` is currently hardcoded to `False`. As a result, `use_gps:=true` launches `GpsSimulatorNode`, but ESKF GPS measurement updates are still disabled unless that launch file is adjusted.

---

## 8. Coordinate Frames

### 8.1 Kalibr T_BS Convention

All extrinsic transforms follow the Kalibr convention:

```
p_body = T_BS @ p_sensor
```

`T_BS` is the 4x4 homogeneous transform from the **sensor** frame (S) to the **body** frame (B). In TF2 terms, `T_BS` is published as `parent=base_link, child=sensor_frame`.

### 8.2 World Frame

The world frame is the **initial body frame** with a fixed rotation to the ROS FLU convention:

```
T_body_to_ros = [ 0  0  1  0 ]     body-z (fwd) -> world-x (fwd)
                [ 0 -1  0  0 ]     body-y (right) -> world-y (left)
                [ 1  0  0  0 ]     body-x (up)   -> world-z (up)
                [ 0  0  0  1 ]
```

This ensures:
- `+X` forward, `+Y` left, `+Z` up (ROS convention)
- Gravity vector `g = [0, 0, -9.81]` points downward in the world frame
- Both VIO and ground truth start at identity pose when properly aligned

### 8.3 Ground Truth Alignment

The `GroundTruthPublisherNode` aligns Vicon poses to the VIO world frame using the first ground-truth sample:

```
T_align = T_body_to_ros @ inv(T_vicon_body_0)
T_world_body_GT(t) = T_align @ T_vicon_body(t)
```

At `t=0` this yields identity, so both trajectories share the same starting pose.

---

## 9. Trajectory Evaluation

The `evaluate_trajectory.py` script performs offline trajectory scoring using CSV logs written by `DebugLoggerNode`.

### 9.1 Metrics

**ATE (Absolute Trajectory Error) with Umeyama SE(3) alignment:**

The estimated trajectory is first aligned to ground truth using a rigid body transform (rotation + translation, no scale):

```
R_align, t_align = argmin sum_i || R @ p_est_i + t - p_gt_i ||^2
```

Solved via SVD of the cross-covariance matrix with determinant correction to ensure a proper rotation.

Position ATE:
```
err_i = ||p_est_aligned_i - p_gt_i||
RMSE  = sqrt(mean(err_i^2))
```

Rotation ATE (geodesic angle):
```
theta_i = 2 * arccos(|w of q_gt_inv * q_est|)
```

**RPE (Relative Pose Error) over fixed path-length segments:**

For each segment length `d` in `{0.5, 1.0, 2.0, 5.0}` metres, pairs `(i, j)` are found where the ground-truth path length between them equals `d`. The RPE is:

```
rpe_i = || (p_est_j - p_est_i) - (p_gt_j - p_gt_i) ||
```

Reported as percentage drift: `rpe / segment_length * 100%`.

### 9.2 Output

Results are saved to `logs/<YYYY-MM-DD_HH-MM-SS>/`:

| File | Description |
|---|---|
| `summary.txt` | ATE (RMSE, mean, std, max) + RPE table for both ESKF and VIO |
| `trajectory.png` | Top-down XY and side-view XZ trajectory comparison |
| `ate_over_time.png` | Per-sample position and rotation ATE vs elapsed time |
| `rpe_boxplot.png` | RPE distribution boxplots at each segment length |

### 9.3 Usage

```bash
python evaluate_trajectory.py --input tmp/ --logs logs/
```

Requires `matplotlib` for plots (optional -- if missing, only `summary.txt` is generated).

---

## 10. Project Structure

```
Visual_Inertial_Odometry/
|-- README.md
|-- requirements.txt
|-- compare_logs.py                       # Compare/sort multiple logs/* runs
|-- evaluate_trajectory.py                 # Offline ATE + RPE scoring
|-- bag_utils/
|   |-- euroc_to_bag.py                    # EuRoC MAV -> ROS 2 bag converter
|   |-- uzh_to_bag.py                      # UZH FPV indoor -> ROS 2 bag converter
|                                          #   (with fisheye undistortion at write time)
|-- ros2_ws/
|   |-- src/
|       |-- vio_pipeline/
|           |-- package.xml
|           |-- setup.py
|           |-- setup.cfg
|           |-- config/
|           |   |-- euroc_params.yaml      # EuRoC VI-Sensor calibration
|           |   |-- uzh_indoor_params.yaml # UZH FPV camera + IMU calibration
|           |   |-- gps_profiles.yaml      # GPS receiver simulation profiles
|           |   |-- pipeline_params.yaml   # Centralized tunable parameters
|           |-- launch/
|           |   |-- full_pipeline.launch.py # Full VIO + ESKF/FGO + GT + optional GPS/TF
|           |   |-- features.launch.py      # Feature visualisation only
|           |-- vio_pipeline/
|               |-- __init__.py
|               |-- feature_tracking_KLT.py    # Shi-Tomasi + KLT tracker
|               |-- feature_tracking_node.py   # Standalone feature viz node
|               |-- vio_node.py                # PoseEstimationNode
|               |-- stereo_rectifier_node.py   # Stereo undistortion + rectification
|               |-- imu_processing_node.py     # Bias removal, LPF, dead-reckoning
|               |-- eskf_node.py               # Error-State Kalman Filter
|               |-- fgo_backend_node.py        # Factor Graph Optimization backend
|               |-- factor_graph.py            # Sliding-window graph + LM + Schur marginalisation
|               |-- imu_preintegrator.py       # On-manifold IMU preintegration
|               |-- so3_utils.py               # SO(3) exponential/log maps, Jacobians
|               |-- ground_truth_publisher.py  # Aligned Vicon GT republisher
|               |-- gps_simulator_node.py      # GPS error model simulator
|               |-- tf_publisher_node.py       # TF2 static + dynamic broadcaster
|               |-- debug_logger_node.py       # CSV trajectory logger
|-- tmp/                                       # CSV output from DebugLoggerNode
|-- logs/                                      # Evaluation output from evaluate_trajectory.py
```

### Node Summary

| Node | Executable | Rate | Purpose |
|---|---|---|---|
| `PoseEstimationNode` | `pose_estimation_node` | 20 Hz | KLT tracking, triangulation, PnP pose |
| `StereoRectifierNode` | `stereo_rectifier_node` | 20 Hz | Stereo undistortion and rectification |
| `ImuProcessingNode` | `imu_processing_node` | 200 Hz | Bias removal, LPF, dead-reckoning |
| `EskfNode` | `eskf_node` | 200 Hz | IMU + VIO fusion backend (GPS-capable, see Section 7 note) |
| `FgoBackendNode` | `fgo_backend_node` | ~20 Hz | IMU preintegration + VO factor graph (alternative backend) |
| `GroundTruthPublisherNode` | `ground_truth_publisher` | variable | Aligned Vicon ground truth |
| `GpsSimulatorNode` | `gps_simulator_node` | 5-20 Hz | Simulated GPS with realistic errors |
| `TfPublisherNode` | `tf_publisher_node` | 200 Hz | TF2 tree (static extrinsics + dynamic map->base_link) |
| `DebugLoggerNode` | `debug_logger_node` | passive | CSV logging of all pipeline topics |
| `FeatureTrackingNode` | `feature_tracking_node` | 20 Hz | Feature visualisation (debug only) |

### Library Modules (not ROS nodes)

| Module | Purpose |
|---|---|
| `feature_tracking_KLT.py` | `FeatureExtractor` class -- Shi-Tomasi detection, KLT tracking, circular consistency |
| `factor_graph.py` | `SlidingWindowGraph`, `KeyframeState`, factor dataclasses, LM optimizer, Schur marginalization |
| `imu_preintegrator.py` | `ImuPreintegrator` -- on-manifold preintegration with covariance and bias Jacobians |
| `so3_utils.py` | `exp_so3`, `log_so3`, `right_jacobian_so3`, `inv_right_jacobian_so3`, quaternion conversions |

---

## 11. Installation

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 24.04 |
| ROS 2 | Humble | Jazzy |
| Python | 3.10 | 3.12 |
| GPU | not required | not required |

Core runtime is CPU-friendly (`numpy`, OpenCV, ROS 2 Python). This repo's `requirements.txt` currently also lists `torch` / `torchvision` and a commented LightGlue install path for separate experimentation workflows.

### Python Dependencies

- `numpy`
- `opencv-python` (cv2)
- `PyYAML`
- `matplotlib` (optional, for `evaluate_trajectory.py` plots)
- `rosbag2_py` (for bag conversion scripts)
- `rich` (optional, for formatted `compare_logs.py` tables)
- `torch`, `torchvision` (listed in `requirements.txt`, optional for the core KLT pipeline)

### Build Steps

```bash
# 1. Source ROS 2
source /opt/ros/jazzy/setup.bash   # or humble

# 2. Install ROS 2 package dependencies
cd ros2_ws
rosdep install --from-paths src -y --ignore-src

# 3. Build the package
colcon build --packages-select vio_pipeline

# 4. Source the workspace
source install/setup.bash
```

### Dataset Preparation

Convert raw datasets to ROS 2 bags:

```bash
# EuRoC MAV
python bag_utils/euroc_to_bag.py \
  --dataset dataset/MH_01_easy/MH_01_easy/mav0 \
  --output bags/mh_01_easy

# UZH FPV indoor (applies fisheye undistortion at write time)
python bag_utils/uzh_to_bag.py \
    --dataset dataset/uzh_indoor_9 \
    --output bags/uzh_indoor_9
```

Both scripts validate existing bags and only rebuild topics with incorrect message counts. Use `--force` to rebuild from scratch.

---

## 12. Running the Pipeline

### Terminal A -- Play the bag

```bash
ros2 bag play bags/mh_01_easy --rate 0.5 --clock
```

- `--rate 0.5` plays at half speed (remove once timing is confirmed)
- `--clock` publishes `/clock` so nodes use simulated time

### Terminal B -- Launch the pipeline

**EuRoC with ESKF (default):**
```bash
ros2 launch vio_pipeline full_pipeline.launch.py
```

**EuRoC with FGO backend:**
```bash
ros2 launch vio_pipeline full_pipeline.launch.py use_fgo:=true
```

**UZH indoor:**
```bash
ros2 launch vio_pipeline full_pipeline.launch.py \
    config_file:=/path/to/uzh_indoor_params.yaml
```

**With GPS simulator enabled (standalone profile):**
```bash
ros2 launch vio_pipeline full_pipeline.launch.py \
    use_gps:=true \
    gps_mode:=standalone
```

**With GPS simulator enabled (RTK profile):**
```bash
ros2 launch vio_pipeline full_pipeline.launch.py \
    use_gps:=true \
    gps_mode:=rtk
```

At the moment, these launch commands enable GPS simulation topics; ESKF GPS correction is still disabled by the hardcoded `"use_gps": False` launch parameter described in Section 7.

**With TF publisher:**
```bash
ros2 launch vio_pipeline full_pipeline.launch.py \
    use_tf_publisher:=true
```

**All launch arguments:**

| Argument | Default | Description |
|---|---|---|
| `config_file` | `euroc_params.yaml` (from package share) | Path to dataset calibration YAML |
| `pipeline_params_file` | `pipeline_params.yaml` (from package share) | Path to centralized tunable parameters |
| `use_sim_time` | `true` | Use `/clock` from bag playback |
| `use_fgo` | `false` | Use FGO sliding-window backend instead of ESKF |
| `use_rectifier` | `true` | Launch `StereoRectifierNode` for stereo rectification |
| `use_gps` | `false` | Launch `GpsSimulatorNode` (ESKF GPS update currently disabled in launch code) |
| `gps_mode` | `standalone` | GPS profile: `standalone` or `rtk` |
| `use_tf_publisher` | `false` | Launch `TfPublisherNode` for TF2 tree |

### Terminal C -- Inspect output

```bash
# Live fused pose (ESKF mode)
ros2 topic echo /eskf/pose

# Live fused pose (FGO mode)
ros2 topic echo /fgo/pose

# Compare rates
ros2 topic hz /vio/pose /eskf/pose

# Visualise feature tracks (requires rqt_image_view)
ros2 run rqt_image_view rqt_image_view /features/temporal_viz
```

### Evaluate trajectory

After the bag finishes and CSVs are written to `tmp/`:

```bash
python evaluate_trajectory.py --input tmp/ --logs logs/
```

### Compare multiple runs

`compare_logs.py` parses `logs/*/summary.txt` and ranks runs by ATE.

```bash
python compare_logs.py --logs-dir logs --sort eskf_ate
```

---

## 13. ROS 2 Topics and Parameters

### Published Topics

| Topic | Type | Node | Description |
|---|---|---|---|
| `/vio/pose` | `PoseStamped` | PoseEstimation | Visual odometry body pose |
| `/vio/path` | `Path` | PoseEstimation | Visual odometry trajectory |
| `/vio/odometry` | `Odometry` | PoseEstimation | Visual odometry (frame: map, child: base_link) |
| `/vio/rpy` | `Vector3Stamped` | PoseEstimation | Roll/pitch/yaw in degrees |
| `/features/temporal_viz` | `Image` | PoseEstimation | KLT motion trail visualisation |
| `/cam0/image_rect` | `Image` | StereoRectifier | Rectified left image |
| `/cam1/image_rect` | `Image` | StereoRectifier | Rectified right image |
| `/eskf/odometry` | `Odometry` | ESKF | Fused pose + velocity (200 Hz) |
| `/eskf/pose` | `PoseStamped` | ESKF | Fused pose |
| `/eskf/path` | `Path` | ESKF | Fused trajectory |
| `/eskf/rpy` | `Vector3Stamped` | ESKF | Fused roll/pitch/yaw in degrees |
| `/fgo/odometry` | `Odometry` | FGO | Optimized pose + velocity (~20 Hz) |
| `/fgo/pose` | `PoseStamped` | FGO | Optimized pose |
| `/fgo/path` | `Path` | FGO | Optimized trajectory |
| `/imu/processed` | `Imu` | ImuProcessing | Bias-corrected + LPF IMU (200 Hz) |
| `/imu/odometry` | `Odometry` | ImuProcessing | IMU dead-reckoning (drifts) |
| `/gt_pub/pose` | `PoseStamped` | GroundTruth | Aligned ground-truth pose |
| `/gt_pub/path` | `Path` | GroundTruth | Aligned ground-truth trajectory |
| `/gt_pub/odometry` | `Odometry` | GroundTruth | Aligned ground-truth odometry |
| `/gt_pub/rpy` | `Vector3Stamped` | GroundTruth | Ground-truth RPY in degrees |
| `/gps/fix` | `NavSatFix` | GpsSimulator | Simulated GPS fix (LLA + covariance) |
| `/gps/enu` | `PointStamped` | GpsSimulator | Corrupted ENU position (debug) |
| `/tf_static` | `TransformStamped` | TfPublisher | Static sensor extrinsics |
| `/tf` | `TransformStamped` | TfPublisher | Dynamic map -> base_link |
| `/features/viz` | `Image` | FeatureTracking | Stereo match visualisation (debug node) |

### Subscribed Topics

| Topic | Type | Subscribers |
|---|---|---|
| `/cam0/image_raw` | `Image` | PoseEstimation, StereoRectifier, FeatureTracking |
| `/cam1/image_raw` | `Image` | PoseEstimation, StereoRectifier, FeatureTracking |
| `/imu0` | `Imu` | ImuProcessing, ESKF (raw gravity buffer), FGO (raw gravity buffer) |
| `/imu/processed` | `Imu` | ESKF, FGO |
| `/vio/odometry` | `Odometry` | ESKF, FGO, DebugLogger |
| `/eskf/odometry` | `Odometry` | TfPublisher, DebugLogger |
| `/fgo/odometry` | `Odometry` | DebugLogger |
| `/gt/pose` | `PoseStamped` | GroundTruth (configurable via `gt_topic` param) |
| `/gt_pub/pose` | `PoseStamped` | GpsSimulator |
| `/gt_pub/odometry` | `Odometry` | DebugLogger |
| `/imu/odometry` | `Odometry` | DebugLogger |

### PoseEstimationNode Parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to calibration YAML |
| `min_tracks` | `10` | Minimum circular-checked tracks to attempt PnP |
| `circular_check_threshold` | `2.0` | Max forward-backward pixel error (px) |
| `max_depth` | `30.0` | Max triangulated point depth (m) |
| `min_inlier_ratio` | `0.4` | Min RANSAC inlier fraction to accept pose |
| `max_translation` | `0.5` | Max accepted per-frame translation (m) |
| `max_rotation_deg` | `30.0` | Max accepted per-frame rotation (deg) |
| `kf_min_translation` | `0.03` | Min accumulated translation for keyframe (m) |
| `kf_min_rotation_deg` | `2.0` | Min accumulated rotation for keyframe (deg) |
| `kf_max_frames` | `5` | Max frames between keyframes |
| `pose_cov_pos_base` | `0.03` | Base position std for dynamic covariance (m) |
| `pose_cov_ang_base` | `0.03` | Base rotation std for dynamic covariance (rad) |
| `max_epipolar_err` | `2.0` | Max stereo epipolar line error (px, rectified mode) |

### ImuProcessingNode Parameters

| Parameter | Default | Launch override |
|---|---|---|
| `config_path` | `""` | from `config_file` arg |
| `imu_topic` | `"/imu0"` | `"/imu0"` |
| `init_duration` | `2.0` | `0.0` |
| `gyro_lpf_cutoff` | `50.0` | `15.0` |
| `accel_lpf_cutoff` | `30.0` | `10.0` |
| `imu_rate_hz` | `200` | from config YAML |

### EskfNode Parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | IMU noise figures |
| `meas_pos_std` | `0.05` | VIO position noise 1-sigma (m) |
| `meas_ang_std` | `0.05` | VIO rotation noise 1-sigma (rad) |
| `init_pos_std` | `1.0` | Initial position uncertainty (m) |
| `init_vel_std` | `0.3` | Initial velocity uncertainty (m/s) |
| `init_att_std` | `0.05` | Initial attitude uncertainty (rad) |
| `init_ba_std` | `0.02` | Initial accel-bias uncertainty (m/s^2) |
| `init_bg_std` | `5.0e-4` | Initial gyro-bias uncertainty (rad/s) |
| `max_dt` | `0.1` | Max IMU dt before discard (s) |
| `use_gps` | `false` | Enable GPS measurement updates |
| `gps_pos_std_h` | `2.0` | GPS horizontal 1-sigma (m) |
| `gps_pos_std_v` | `4.0` | GPS vertical 1-sigma (m) |
| `gps_gate_chi2` | `7.815` | Mahalanobis gate threshold (0 = disabled) |

### FgoBackendNode Parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | IMU noise figures |
| `window_size` | `10` | Number of keyframes in sliding window |
| `lm_max_iter` | `8` | Max Levenberg-Marquardt iterations per keyframe |
| `lm_lambda_init` | `1.0e-3` | Initial LM damping factor |
| `imu_noise_scale` | `20.0` | IMU white noise inflation factor for preintegration |
| `min_gravity_samples` | `100` | Min raw IMU samples for gravity estimation |
| `vo_pos_std` | `0.05` | Fallback VO position noise (m) |
| `vo_ang_std` | `0.05` | Fallback VO rotation noise (rad) |
| `prior_pos_std` | `0.01` | Prior on first keyframe position (m) |
| `prior_vel_std` | `1.0` | Prior on first keyframe velocity (m/s) |
| `prior_att_std` | `0.01` | Prior on first keyframe attitude (rad) |
| `prior_ba_std` | `0.02` | Prior on accel bias (m/s^2) |
| `prior_bg_std` | `5.0e-4` | Prior on gyro bias (rad/s) |

### GpsSimulatorNode Parameters

| Parameter | Default | Description |
|---|---|---|
| `update_rate_hz` | `5.0` | GPS publishing rate (Hz) |
| `ref_lat_deg` | `47.3977419` | WGS-84 reference latitude (deg) |
| `ref_lon_deg` | `8.5455938` | WGS-84 reference longitude (deg) |
| `ref_alt_m` | `486.0` | WGS-84 reference altitude (m) |
| `noise_h_m` | `1.5` | Horizontal position 1-sigma (m) |
| `noise_v_m` | `3.0` | Vertical position 1-sigma (m) |
| `bias_time_const_s` | `60.0` | Gauss-Markov bias time constant (s) |
| `bias_walk_h_m_s` | `0.1` | Horizontal bias walk sigma (m/sqrt(s)) |
| `bias_walk_v_m_s` | `0.15` | Vertical bias walk sigma (m/sqrt(s)) |
| `multipath_prob` | `0.02` | Per-sample multipath probability |
| `multipath_scale` | `4.0` | Multipath error scale factor (x sigma_h) |
| `outage_prob_per_s` | `0.0` | Probability of outage per second |
| `outage_duration_s` | `3.0` | Duration of each outage (s) |
| `seed` | `-1` | RNG seed (-1 = random) |

### TfPublisherNode Parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to calibration YAML (for sensor T_BS extrinsics) |

### GroundTruthPublisherNode Parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to calibration YAML |
| `gt_topic` | `"/gt/pose"` | Input ground-truth topic |

### DebugLoggerNode Parameters

| Parameter | Default | Description |
|---|---|---|
| `output_dir` | `<project_root>/tmp` | Directory for CSV output files |
| `pipeline_params_file` | `""` | Path to pipeline_params.yaml (copied to output for reproducibility) |

---

## 14. Configuration Files

### `pipeline_params.yaml`

**Centralized tunable parameters** for all pipeline nodes. The launch file reads this single file and distributes values to each node. Edit here to change behaviour -- no need to touch launch files or Python nodes.

Sections:

| Section | Controls |
|---|---|
| `eskf` | ESKF measurement noise, initial covariance, max dt |
| `imu` | IMU preprocessing: init duration, LPF cutoffs |
| `feature_tracking` | Shi-Tomasi + KLT parameters |
| `vio` | PnP thresholds, keyframe selection, dynamic covariance |
| `fgo` | FGO window size, LM optimizer, IMU noise scaling, priors |

### `euroc_params.yaml`

EuRoC VI-Sensor calibration (ADIS16448 IMU + stereo cameras, 200 Hz IMU / 20 Hz stereo, 752x480):

```yaml
cam0:
  intrinsics: [458.654, 457.296, 367.215, 248.375]  # fx fy cx cy
  distortion: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]  # k1 k2 p1 p2
  resolution: [752, 480]
  rate_hz: 20
  T_BS: [...]   # 4x4 row-major cam0 -> body (Kalibr convention)

cam1:
  intrinsics: [457.587, 456.134, 379.999, 255.238]
  distortion: [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]
  resolution: [752, 480]
  rate_hz: 20
  T_BS: [...]   # 4x4 row-major cam1 -> body

stereo_baseline: 0.11007  # metres

imu:
  gyro_noise:  1.6968e-04   # rad/s/sqrt(Hz)
  gyro_walk:   1.9393e-05   # rad/s^2/sqrt(Hz)
  accel_noise: 2.0000e-03   # m/s^2/sqrt(Hz)
  accel_walk:  3.0000e-03   # m/s^3/sqrt(Hz)
  rate_hz: 200
  T_BS: [identity]          # IMU = body frame for EuRoC

gravity: [0.0, 0.0, -9.81]  # overridden by ESKF/FGO gravity estimator
```

### `uzh_indoor_params.yaml`

UZH FPV indoor dataset calibration (Snapdragon stereo + IMU, 500 Hz IMU / 30 Hz stereo, 640x480). Images are pre-undistorted with the fisheye model during bag creation (`uzh_to_bag.py`), so distortion coefficients are all zeros.

Key differences from EuRoC:
- Lower resolution (640x480 vs 752x480)
- Higher IMU rate (500 Hz vs 200 Hz)
- Different stereo baseline (0.07962 m vs 0.11007 m)
- Higher IMU noise figures (gyro: 0.05 vs 1.7e-4 rad/s/sqrt(Hz))

### `gps_profiles.yaml`

Contains two GPS receiver profiles (`standalone` and `rtk`), each with `simulator` and `eskf` sections. The simulator section defines the noise injected by `GpsSimulatorNode`; the `eskf` section defines the measurement noise assumed by the ESKF filter.

Select a profile at launch:
```bash
ros2 launch vio_pipeline full_pipeline.launch.py use_gps:=true gps_mode:=standalone
ros2 launch vio_pipeline full_pipeline.launch.py use_gps:=true gps_mode:=rtk
```

These commands currently configure the simulator profile; enabling ESKF GPS updates additionally requires changing the hardcoded ESKF `use_gps` launch parameter in `full_pipeline.launch.py`.

To add a new profile, add a top-level key to `gps_profiles.yaml` with the same `simulator` / `eskf` structure and reference it via `gps_mode:=<name>`.
