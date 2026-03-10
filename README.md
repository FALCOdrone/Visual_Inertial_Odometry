# Visual-Inertial Odometry Pipeline (Factor-Graph branch)

A ROS 2 stereo visual-inertial odometry system for the [EuRoC MAV dataset](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f).

Feature tracking uses **Shi-Tomasi corners + KLT optical flow** with a forward-backward circular consistency check. Pose estimation uses stereo triangulation followed by PnP-RANSAC. IMU fusion is offered via two complementary backends:

- **Error-State Kalman Filter (ESKF)** — linear filter, always running, low latency
- **Sliding-Window Factor Graph (FGO)** — nonlinear batch optimizer, optional, higher accuracy

> **Branch:** `Factor-Graph` — adds on-manifold IMU preintegration and a Levenberg-Marquardt sliding-window optimizer on top of the `KLT` branch.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Feature tracking — Shi-Tomasi + KLT](#2-feature-tracking--shi-tomasi--klt)
3. [Forward-backward circular consistency check](#3-forward-backward-circular-consistency-check)
4. [Stereo triangulation](#4-stereo-triangulation)
5. [Pose estimation — PnP-RANSAC](#5-pose-estimation--pnp-ransac)
6. [IMU processing](#6-imu-processing)
7. [Error-State Kalman Filter (ESKF)](#7-error-state-kalman-filter-eskf)
8. [IMU preintegration on manifold](#8-imu-preintegration-on-manifold)
9. [Sliding-Window Factor Graph (FGO)](#9-sliding-window-factor-graph-fgo)
10. [FGO backend node](#10-fgo-backend-node)
11. [Coordinate frames](#11-coordinate-frames)
12. [Project structure](#12-project-structure)
13. [Requirements and installation](#13-requirements-and-installation)
14. [Running the pipeline](#14-running-the-pipeline)
15. [ROS 2 topics and parameters](#15-ros-2-topics-and-parameters)
16. [Configuration file](#16-configuration-file)

---

## 1. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            EuRoC MAV Bag                                     │
│  /cam0/image_raw (20 Hz)   /cam1/image_raw (20 Hz)   /imu0 (200 Hz)         │
└──────┬──────────────────────────┬──────────────────────┬─────────────────────┘
       │  ApproximateTimeSynchronizer                     │
       ▼                          ▼                       ▼
┌───────────────────────────────────────┐     ┌─────────────────────────┐
│          PoseEstimationNode           │     │    ImuProcessingNode    │
│                                       │     │                         │
│  Shi-Tomasi detect                    │     │  static bias init       │
│       │                               │     │  2nd-order Butterworth  │
│  KLT temporal tracking (L→L)          │     │  LPF (gyro + accel)    │
│  + forward-backward consistency       │     │  dead-reckoning         │
│       │                               │     └──────────┬──────────────┘
│  KLT stereo matching (L→R)            │                │ /imu/processed (200 Hz)
│  + forward-backward consistency       │                │
│       │                               │                ▼
│  undistort keypoints                  │     ┌─────────────────────────┐
│       │                               │     │       EskfNode          │
│  triangulate 3D (prev stereo pair)    │     │                         │
│       │                               │     │  IMU predict  200 Hz    │
│  PnP-RANSAC + iterative refinement    │     │  VIO update    20 Hz    │
│       │                               │     │  Joseph-form update     │
│  motion sanity check                  │     │  bias estimation        │
│       │                               │     └──────────┬──────────────┘
│  T_world_body accumulated             │                │ /eskf/odometry
└──────┬────────────────────────────────┘                │ /eskf/pose
       │                                                  │ /eskf/path
   /vio/odometry  /vio/pose  /vio/path                   │
       │                                                  │
       └──────────────────┐   ┌──────────────────────────┘
                          ▼   ▼
              ┌──────────────────────────────┐
              │       FgoBackendNode         │  (optional: use_fgo:=true)
              │                              │
              │  gravity estimation          │
              │  IMU preintegration  200 Hz  │
              │  keyframe selection   20 Hz  │
              │  LM optimization  background │
              │  Schur marginalization       │
              │  bias-corrected propagation  │
              └──────────────┬───────────────┘
                             │ /fgo/odometry
                             │ /fgo/pose
                             │ /fgo/path

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
M = Σ_{patch} [ Ix²    Ix·Iy ]
               [ Ix·Iy  Iy²  ]

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

The pyramid (depth `max_level = 3`) allows tracking across large inter-frame displacements.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `win_size` | (14, 14) | Patch size per pyramid level; larger handles faster motion |
| `max_level` | 3 | Pyramid depth; 0 = no pyramid |

---

## 3. Forward-backward circular consistency check

Raw KLT tracks are validated with a **forward-backward (circular) check**:

```
1. Track p₀ forward  (img₀ → img₁):  p₁  = KLT(img₀, img₁, p₀)
2. Track p₁ backward (img₁ → img₀):  p̂₀ = KLT(img₁, img₀, p₁)
3. Accept if ‖p₀ − p̂₀‖₂ < threshold   (default 2 px)
```

Applied independently for **temporal** tracks (left_{t-1} → left_t) and **stereo**
matches (left_t → right_t). Surviving tracks yield index-aligned quadruples:

```
kpts_l_prev  (N,2)   kpts_l_curr  (N,2)
kpts_r_prev  (N,2)   kpts_r_curr  (N,2)
```

---

## 4. Stereo triangulation

### Projection matrices

```
P₀ = K₀ · [I | 0]
P₁ = K₁ · [R_{c₁c₀} | t_{c₁c₀}]
```

where `T_{c₁c₀} = T_b_c1⁻¹ @ T_b_c0` from the Kalibr calibration.

### Undistortion

Keypoints are undistorted before any geometry using the radial-tangential model:

```
p_u = undistortPoints(p_d, K, [k₁, k₂, p₁, p₂], P=K)
```

### DLT triangulation

For each stereo-matched pair (p₀, p₁) the 3D point **X** satisfies:

```
[p₀ × P₀] · X = 0
[p₁ × P₁]
```

solved via `cv2.triangulatePoints`. Points with depth Z < 0.1 m or Z > `max_depth` (30 m)
are discarded.

---

## 5. Pose estimation — PnP-RANSAC

### Problem

Given N 3D landmarks {X_i} in the previous camera frame and their 2D projections {p_i}
in the current left image:

```
λ · p_i = K · (R · X_i + t)
```

### RANSAC + EPnP + iterative refinement

1. `cv2.solvePnPRansac` with `SOLVEPNP_EPNP` (200 iters, 2 px threshold, 99.9% confidence).
2. Levenberg-Marquardt (`SOLVEPNP_ITERATIVE`) refines on the inlier set.

### Layered outlier rejection

| Guard | Condition | Rejects |
|---|---|---|
| Depth filter | 0.1 m < Z < 30 m | Noisy/degenerate stereo points |
| Minimum count | N_valid ≥ 6 | Under-determined PnP |
| Inlier ratio | n_inliers / N ≥ 0.4 | Corrupt RANSAC consensus |
| Max translation | ‖t‖ ≤ 0.5 m/frame | Pose jumps |
| Max rotation | angle ≤ 30°/frame | Orientation flips |

### Pose accumulation

`solvePnP` returns **T_{curr,prev}**. The world-frame body pose is updated:

```
T_{world,cam0}^t = T_{world,cam0}^{t-1} · T_{curr,prev}⁻¹
T_{world,body}   = T_{world,cam0} · T_{b_c0}⁻¹
```

---

## 6. IMU processing

`ImuProcessingNode` preprocesses raw `/imu0` before fusion.

### Static initialisation

Collects `init_duration` seconds of static data to estimate:

- **Gyroscope bias** `b_g = mean(ω_raw)` — ideal ω = 0 at rest
- **Accelerometer bias** from gravity direction
- **Initial orientation** — roll and pitch from mean specific force

### Per-sample pipeline

```
raw /imu0
  ├─ bias subtraction:  a_corr = a_raw − b_a,   ω_corr = ω_raw − b_g
  ├─ 2nd-order Butterworth LPF (biquad, bilinear transform)
  │     gyro:  cutoff 15 Hz
  │     accel: cutoff 10 Hz
  ├─ dead-reckoning integration (attitude + velocity + position)
  └─ publish /imu/processed
```

---

## 7. Error-State Kalman Filter (ESKF)

`EskfNode` fuses bias-corrected IMU (200 Hz) with visual odometry pose updates (20 Hz).

### State vector

```
x  = [p(3)  v(3)  q(4)   b_a(3)  b_g(3)]   nominal state (16-DOF)
δx = [δp(3) δv(3) δθ(3)  δb_a(3) δb_g(3)]  error state   (15-DOF)
```

### Prediction (every IMU sample, dt ≈ 5 ms)

```
a_w   = R·(f − b_a) + g
p    ← p + v·dt + ½·a_w·dt²
v    ← v + a_w·dt
q    ← q ⊗ Exp((ω − b_g)·dt)

F = I₁₅ + Fc·dt   (first-order discrete transition)
Fc[0:3, 3:6]  =  I              δṗ = δv
Fc[3:6, 6:9]  = -R·[f×]        δv̇ ← −R·[f×]·δθ
Fc[3:6, 9:12] = -R              δv̇ ← −R·δb_a
Fc[6:9, 6:9]  = -[ω×]          δθ̇ = −[ω×]·δθ
Fc[6:9,12:15] = -I              δθ̇ ← −δb_g

Qd[3:6,  3:6 ] = σ_a²·dt·I
Qd[6:9,  6:9 ] = σ_g²·dt·I
Qd[9:12, 9:12] = σ_ba²·dt·I
Qd[12:15,12:15]= σ_bg²·dt·I

P ← F·P·Fᵀ + Qd
```

### Update (every VIO frame, dt ≈ 50 ms)

Innovation:

```
z = [ p_meas − p_nom ;  Log(R_nom.T @ R_meas) ]   ∈ ℝ⁶
```

Measurement Jacobian:

```
H[0:3, 0:3] = I₃   (position)
H[3:6, 6:9] = I₃   (orientation error in body frame)
```

Joseph-form update (numerically stable):

```
S      = H·P·Hᵀ + R_meas
K      = P·Hᵀ·S⁻¹
IKH    = I₁₅ − K·H
P     ← IKH·P·IKHᵀ + K·R_meas·Kᵀ
δx    = K·z   →   inject into nominal state
```

### Gravity estimation

On the first VIO pose the ESKF rotates the mean static accelerometer reading into the VIO
world frame:

```
g_world = R_init · (−mean(a_raw_static))
```

---

## 8. IMU preintegration on manifold

`imu_preintegrator.py` implements on-manifold preintegration following
**Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry," TRO 2017**.

### Motivation

A standard IMU integration from keyframe i to keyframe j depends on the absolute state at i:

```
p_j = p_i + v_i·Δt + R_i·Δp   (where Δp depends on R_i)
```

When the optimizer changes the state at i (as it does every LM iteration), the whole
integration would need to be recomputed. Preintegration decouples the relative measurements
from the absolute state by integrating in the **body frame at keyframe i**, making the
factor independent of the linearisation state.

### Preintegrated measurements

Accumulated between keyframes i and j over N IMU samples:

```
ΔR = ∏ₖ Exp(ω̃_k · δt)
Δv = Σₖ ΔR_{0..k-1} · ã_k · δt
Δp = Σₖ [Δv_{0..k-1} · δt + ½ · ΔR_{0..k-1} · ã_k · δt²]
```

where `ω̃_k = ω_k − b_g` and `ã_k = a_k − b_a` are the bias-corrected measurements.

### Discrete integration recursion (`integrate` method)

For each IMU sample, in strict order:

**Step 1 — Update bias Jacobians** using the rotation **before** this step:

```
J_p_ba +=  J_v_ba · dt − ½ · ΔR · dt²
J_p_bg +=  J_v_bg · dt − ½ · ΔR · [ã]× · J_R_bg · dt²
J_v_ba += −ΔR · dt
J_v_bg += −ΔR · [ã]× · J_R_bg · dt
J_R_bg  = δR.T · J_R_bg − Jr · dt
```

where `δR = Exp(ω̃·dt)` and `Jr` is the right Jacobian of SO(3).

**Step 2 — Update preintegrated state** (p before v before R):

```
Δp  += Δv · dt + ½ · ΔR · ã · dt²
Δv  += ΔR · ã · dt
ΔR   = ΔR · δR
```

**Step 3 — Propagate covariance** using the rotation **after** this step:

State transition F (9×9) for [Δp, Δv, Δθ]:

```
F[0:3, 3:6] = I · dt
F[0:3, 6:9] = −½ · ΔR_new · [ã]× · dt²
F[3:6, 6:9] = −ΔR_new · [ã]× · dt
F[6:9, 6:9] = δR.T
```

Noise input matrix G (9×6) mapping [accel noise, gyro noise] to state error:

```
G[0:3, 0:3] = −½ · ΔR_new · dt²
G[3:6, 0:3] = −ΔR_new · dt
G[6:9, 3:6] = −Jr · dt
```

Discrete covariance update:

```
Q_d = G · diag(σ_a², σ_a², σ_a², σ_g², σ_g², σ_g²) · G.T
Cov ← F · Cov · F.T + Q_d
```

### First-order bias correction

When the optimizer updates the biases by a small increment δb = b_new − b_old, the
preintegrated measurements are corrected without re-integration:

```
Δp_corr = Δp + J_p_ba · δb_a + J_p_bg · δb_g
Δv_corr = Δv + J_v_ba · δb_a + J_v_bg · δb_g
ΔR_corr = ΔR · Exp(J_R_bg · δb_g)
```

If the bias change is large (‖δb_a‖ > 0.01 or ‖δb_g‖ > 0.001), the stored raw IMU buffer
is replayed with the new linearisation biases (`reintegrate`).

---

## 9. Sliding-Window Factor Graph (FGO)

`factor_graph.py` implements a sliding-window Maximum-a-Posteriori estimator.

### State per keyframe

Each keyframe holds a 16-DOF manifold state, parameterised by a 15-DOF tangent vector:

```
x    = [p(3), v(3), q(4), b_a(3), b_g(3)]    on-manifold  (16-DOF)
δx   = [δp(3), δv(3), δθ(3), δb_a(3), δb_g(3)]  tangent space (15-DOF)
```

The **retraction** maps a tangent vector back onto the manifold:

```
p   ← p + δp
v   ← v + δv
q   ← normalize(q ⊗ Exp_quat(δθ))   (right perturbation on SO(3))
b_a ← b_a + δb_a
b_g ← b_g + δb_g
```

### Factor types

The graph assembles a Gauss-Newton Hessian by summing contributions from five factor types.
For each factor with residual **r** and information matrix **Σ⁻¹**:

```
H +=  J.T · Σ⁻¹ · J
b +=  J.T · Σ⁻¹ · r
cost += ½ · r.T · Σ⁻¹ · r
```

#### Factor 1: IMU preintegration (9-DOF)

Connects consecutive keyframes i and j. Residual expressed in the body frame at i:

```
r_p = R_i.T · (p_j − p_i − v_i·Δt − ½·g·Δt²) − Δp_corr
r_v = R_i.T · (v_j − v_i − g·Δt) − Δv_corr
r_R = Log(ΔR_corr.T · R_i.T · R_j)
r   = [r_p(3); r_v(3); r_R(3)]
```

Jacobians J_i (9×15) and J_j (9×15) are computed by forward finite differences.

Information matrix from the preintegrated covariance, with a floor to prevent the IMU
factor from dominating the VO factor:

```
Cov_floor = diag(1e-4, 1e-4, 1e-4,       # σ_p ≥ 1 cm
                  0.09, 0.09, 0.09,        # σ_v ≥ 0.3 m/s
                  7.6e-5, 7.6e-5, 7.6e-5) # σ_θ ≥ 0.5°
Cov_safe   = max(Cov, Cov_floor) + 1e-8·I
Σ⁻¹        = inv(Cov_safe),  enforced symmetric
```

#### Factor 2: VO absolute-pose anchor (6-DOF)

Anchors a keyframe to its visual odometry measurement:

```
r_t = p_kf − p_meas
r_R = Log(R_meas.T · R_kf)
r   = [r_t(3); r_R(3)]

J_pose[0:3, 0:3] = I₃   (∂r_t / ∂δp)
J_pose[3:6, 6:9] = I₃   (∂r_R / ∂δθ)
Σ = diag(σ_t², σ_t², σ_t², σ_r², σ_r², σ_r²)
```

#### Factor 3: Velocity prior (3-DOF)

Soft anchor on velocity — prevents the velocity null-space from diverging on the first
keyframe before any IMU factor provides a relative velocity constraint:

```
r_v = v_kf − v_meas
J[3:6] = I₃,   Σ = σ_v²·I₃
```

#### Factor 4: Bias random-walk (6-DOF between consecutive keyframes)

Constrains how fast biases may change between frames (continuous Brownian-motion model):

```
r_ba = b_a_j − b_a_i,   info_ba = 1 / (σ_ba² · Δt)
r_bg = b_g_j − b_g_i,   info_bg = 1 / (σ_bg² · Δt)
```

Affects the [9:12] (accel bias) and [12:15] (gyro bias) tangent blocks of both keyframes,
with opposing signs on the off-diagonal cross-terms.

#### Factor 5: Marginalization prior

Dense linear prior accumulated from previous marginalizations (see below).

### Levenberg-Marquardt optimization

At each keyframe insertion, the optimizer runs in a background thread:

```
λ ← λ_init

for iter in [0 .. max_iters):
    H, b, cost = assemble_hessian()
    diag_H     = max(diag(H), 1e-6)          # avoid zero diagonal
    H_damp     = H + λ·diag(diag_H) + 1e-8·I
    dx         = solve(H_damp, −b)
    if not finite(dx):  λ ← min(10λ, 1e8); continue
    backup state
    retract all keyframes by dx
    new_cost   = assemble_hessian()[2]
    if new_cost < cost:  λ ← max(λ/3, 1e-8)   # accept
    else:  restore; λ ← min(10λ, 1e8)           # reject
```

### Schur complement marginalization

When the window overflows, the oldest keyframe is marginalized rather than discarded.
This preserves its information as a **dense linear prior** on the remaining keyframes.

**Steps:**

1. Assemble the full Hessian H and gradient b (including any existing prior).

2. Partition the state into:
   - **α** (to marginalize): oldest keyframe `[0 : KF_DIM]`
   - **β** (to keep): all remaining keyframes `[KF_DIM : N·KF_DIM]`

3. Compute the Schur complement:

```
H_αα += 1e-8·I           (regularization before inversion)
H_prior = H_ββ − H_βα · H_αα⁻¹ · H_αβ
b_prior = b_β  − H_βα · H_αα⁻¹ · b_α
```

4. Store `H_prior`, `b_prior`, and the IDs of the remaining keyframes.

5. Remove the oldest keyframe, its IMU factor, and its pose/velocity factors.

**Key property:** The new prior **replaces** the old one (not accumulates), because
`assemble_hessian` already included the old prior in H when computing H_αα and H_ββ.
Accumulating would double-count the old information.

---

## 10. FGO backend node

`FgoBackendNode` (`fgo_backend_node.py`) is the ROS 2 wrapper that drives the above.

### State machine

```
UNINIT ──── first /vio/odometry ──── RUNNING
```

During UNINIT, raw `/imu0` accelerometer data is buffered for gravity estimation.
On the first VO measurement, the gravity vector is estimated:

```
g_world = R_init · (−mean(a_raw_static))
```

and the graph is seeded with an initial keyframe anchored tightly to the VO pose
(σ_t = 1 mm, σ_r = 0.5°) and a zero-velocity prior (σ_v = 0.3 m/s).

### IMU callback — 200 Hz nominal pose propagation

Between keyframes, the IMU is used to propagate a high-rate nominal pose for smooth output:

```
R      = quat_to_rot(q_prop)
a_world= R · (a − b_a) + g_world
p_prop+= v_prop · dt + ½ · a_world · dt²
v_prop+= a_world · dt
q_prop = normalize(q_prop ⊗ Exp_quat((ω − b_g) · dt))
```

Simultaneously the same sample is fed into the active `ImuPreintegrator` for the next
keyframe's IMU factor.

### Keyframe selection — VO callback

A new keyframe is triggered when **any** of these conditions holds:

| Criterion | Default threshold |
|---|---|
| Translation from last keyframe | > 0.05 m |
| Rotation from last keyframe | > 5° |
| Time since last keyframe | > 0.25 s |

### Post-optimization state re-anchoring

After the LM optimizer runs, the propagated state is recomputed to reflect the refined
keyframe state while preserving the IMU delta accumulated since that keyframe:

```
R_kf   = quat_to_rot(latest_kf.q)
dp, dv, dR = preint.bias_corrected_measurement(latest_kf.b_a, latest_kf.b_g)
Δt     = preint.dt_total

p_prop = latest_kf.p + latest_kf.v·Δt + ½·g·Δt² + R_kf·Δp
v_prop = latest_kf.v + g·Δt + R_kf·Δv
q_prop = normalize(rot_to_quat(R_kf · ΔR))
```

This re-anchoring ensures that the 200 Hz output remains consistent with the optimized
trajectory without waiting for the next VO frame.

---

## 11. Coordinate frames

### Kalibr convention (T_BS)

```
p_body = T_BS · p_sensor
```

`T_b_c0` maps a point in the cam0 frame to the body (IMU) frame.

### World frame

Defined as the **initial body frame** (FLU — Forward, Left, Up):

```
+X → forward   (body x at t=0)
+Y → left      (body y at t=0)
+Z → up        (body z at t=0, approximately anti-gravity)
```

Initialising `T_{world,cam0} = T_{b_c0}` at t=0 ensures `T_{world,body}⁰ = I`.

### Ground truth alignment

```
T_align              = T_{vicon,body_0}⁻¹
T_{world,body}^GT(t) = T_align · T_{vicon,body}(t)
```

---

## 12. Project structure

```
Visual_Inertial_Odometry/
├── README.md
├── requirements.txt
├── bag_utils/
│   ├── euroc_to_bag.py          # Convert EuRoC CSV → ROS 2 bag
│   └── uzh_to_bag.py
└── ros2_ws/
    └── src/
        └── vio_pipeline/
            ├── package.xml
            ├── setup.py
            ├── setup.cfg
            ├── config/
            │   └── euroc_params.yaml       # Camera + IMU calibration
            ├── launch/
            │   ├── full_pipeline.launch.py # Full VIO + ESKF [+ FGO]
            │   └── features.launch.py      # Feature visualisation only
            └── vio_pipeline/
                ├── feature_tracking_KLT.py     # Shi-Tomasi + KLT tracker
                ├── feature_tracking_node.py    # Standalone feature viz node
                ├── vio_node.py                 # PoseEstimationNode
                ├── imu_processing_node.py      # Bias removal, LPF, dead-reckoning
                ├── eskf_node.py                # Error-State Kalman Filter
                ├── imu_preintegrator.py        # On-manifold IMU preintegration (NEW)
                ├── factor_graph.py             # Sliding-window LM optimizer (NEW)
                ├── fgo_backend_node.py         # FGO ROS 2 node (NEW)
                ├── ground_truth_publisher.py   # Aligned Vicon GT republisher
                ├── tf_publisher_node.py        # TF transforms publisher
                └── debug_logger_node.py        # CSV trajectory logger
```

### Node summary

| Node | Executable | Rate | Purpose |
|---|---|---|---|
| `PoseEstimationNode` | `pose_estimation_node` | 20 Hz | KLT tracking → triangulation → PnP → pose |
| `ImuProcessingNode` | `imu_processing_node` | 200 Hz | Bias removal, LPF, dead-reckoning |
| `EskfNode` | `eskf_node` | 200 Hz | Linear IMU + VIO fusion |
| `FgoBackendNode` | `fgo_backend_node` | 200 Hz out / LM bg | Nonlinear sliding-window optimizer |
| `GroundTruthPublisherNode` | `ground_truth_publisher` | — | Aligned Vicon GT |
| `TfPublisherNode` | `tf_publisher_node` | — | TF frame transforms |
| `DebugLoggerNode` | `debug_logger_node` | — | CSV trajectory logger |
| `FeatureTrackingNode` | `feature_tracking_node` | 20 Hz | Feature visualisation (debug) |

---

## 13. Requirements and installation

### System requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 22.04 / 24.04 |
| ROS 2 | Humble | Jazzy |
| Python | 3.10 | 3.11 |
| GPU | — | not required |

No deep-learning dependencies. The full pipeline runs in real-time on CPU.

### 1. Install ROS 2

Follow the [official ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `opencv-python`, `pyyaml`, `scipy` (for `message_filters`).

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

## 14. Running the pipeline

### Prepare the EuRoC bag

Download an EuRoC sequence (e.g. `MH_01_easy`) and convert to ROS 2 bag format:

```bash
python3 bag_utils/euroc_to_bag.py  \
    --data-dir /path/to/MH_01_easy/mav0 \
    --output   bags/euroc_mav0
```

### Terminal A — play the bag

```bash
ros2 bag play bags/euroc_mav0 --rate 0.5 --clock
```

`--rate 0.5` plays at half speed; remove once timing is confirmed.
`--clock` publishes `/clock` so all nodes use simulated time.

### Terminal B — launch with ESKF only (default)

```bash
ros2 launch vio_pipeline full_pipeline.launch.py
```

### Terminal B — launch with FGO backend enabled

```bash
ros2 launch vio_pipeline full_pipeline.launch.py use_fgo:=true
```

Both the ESKF (`/eskf/...`) and FGO (`/fgo/...`) nodes run simultaneously when
`use_fgo:=true`. They are independent — both subscribe to `/vio/odometry` and
`/imu/processed` and publish to separate namespaces.

### Terminal C — inspect output

```bash
# Live fused pose (ESKF)
ros2 topic echo /eskf/pose

# Live fused pose (FGO)
ros2 topic echo /fgo/pose

# Compare output rates
ros2 topic hz /vio/pose /eskf/pose /fgo/pose

# RViz — add Path displays for:
#   /vio/path, /eskf/path, /fgo/path, /gt_pub/path
rviz2

# FGO bias estimates (log every 10 keyframes)
ros2 topic echo /rosout | grep "FGO opt"
```

### Feature visualisation only

```bash
ros2 launch vio_pipeline features.launch.py
# /features/temporal_viz — KLT motion trails
```

---

## 15. ROS 2 topics and parameters

### Published topics

| Topic | Type | Node | Description |
|---|---|---|---|
| `/vio/pose` | `PoseStamped` | PoseEstimation | Camera-only body pose |
| `/vio/path` | `Path` | PoseEstimation | Camera-only trajectory |
| `/vio/odometry` | `Odometry` | PoseEstimation | Camera-only odometry |
| `/vio/rpy` | `Vector3Stamped` | PoseEstimation | Roll/pitch/yaw (degrees) |
| `/eskf/odometry` | `Odometry` | ESKF | Fused pose + velocity |
| `/eskf/pose` | `PoseStamped` | ESKF | Fused pose |
| `/eskf/rpy` | `Vector3Stamped` | ESKF | Fused RPY (degrees) |
| `/eskf/path` | `Path` | ESKF | ESKF trajectory |
| `/fgo/odometry` | `Odometry` | FGO | Optimized fused pose + velocity |
| `/fgo/pose` | `PoseStamped` | FGO | Optimized pose |
| `/fgo/path` | `Path` | FGO | FGO trajectory |
| `/imu/processed` | `Imu` | ImuProcessing | Bias-corrected + filtered IMU |
| `/imu/odometry` | `Odometry` | ImuProcessing | IMU dead-reckoning |
| `/gt_pub/pose` | `PoseStamped` | GroundTruth | Aligned Vicon pose |
| `/gt_pub/path` | `Path` | GroundTruth | Aligned Vicon trajectory |
| `/gt_pub/odometry` | `Odometry` | GroundTruth | Aligned Vicon odometry |

### Subscribed topics

| Topic | Type | Subscribers |
|---|---|---|
| `/cam0/image_raw` | `Image` | PoseEstimation |
| `/cam1/image_raw` | `Image` | PoseEstimation |
| `/imu0` | `Imu` | ImuProcessing, ESKF (gravity buffer), FGO (gravity buffer) |
| `/imu/processed` | `Imu` | ESKF, FGO |
| `/vio/odometry` | `Odometry` | ESKF, FGO |

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
| `init_duration` | `0.0` | Static initialisation window (s) |
| `gyro_lpf_cutoff` | `15.0` | Gyro low-pass filter cutoff (Hz) |
| `accel_lpf_cutoff` | `10.0` | Accel low-pass filter cutoff (Hz) |
| `imu_rate_hz` | `200` | Expected IMU sample rate |

### EskfNode parameters

| Parameter | Default | Description |
|---|---|---|
| `config_path` | `""` | Path to `euroc_params.yaml` |
| `meas_pos_std` | `0.1` | VIO position noise 1-σ (m) |
| `meas_ang_std` | `0.2` | VIO rotation noise 1-σ (rad) |
| `init_pos_std` | `1.0` | Initial position uncertainty (m) |
| `init_vel_std` | `0.5` | Initial velocity uncertainty (m/s) |
| `init_att_std` | `0.1` | Initial attitude uncertainty (rad) |
| `init_ba_std` | `0.02` | Initial accel-bias uncertainty (m/s²) |
| `init_bg_std` | `5e-4` | Initial gyro-bias uncertainty (rad/s) |
| `max_dt` | `0.1` | Max IMU dt before sample is discarded (s) |

### FgoBackendNode parameters

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `10` | Number of keyframes in the sliding window |
| `lm_max_iters` | `5` | Levenberg-Marquardt iterations per keyframe |
| `kf_trans_thresh` | `0.05` | Keyframe translation threshold (m) |
| `kf_rot_thresh_deg` | `5.0` | Keyframe rotation threshold (deg) |
| `kf_time_thresh` | `0.25` | Keyframe time threshold (s) |
| `sigma_a` | `2.0e-3` | Accel noise density (m/s²/√Hz) |
| `sigma_g` | `1.6968e-4` | Gyro noise density (rad/s/√Hz) |
| `sigma_ba` | `3.0e-3` | Accel bias random walk (m/s³/√Hz) |
| `sigma_bg` | `1.9393e-5` | Gyro bias random walk (rad/s²/√Hz) |
| `max_dt` | `0.1` | Max IMU dt before sample is discarded (s) |

---

## 16. Configuration file

`config/euroc_params.yaml` stores the EuRoC VI-Sensor calibration used by all nodes:

```yaml
cam0:
  intrinsics: [458.654, 457.296, 367.215, 248.375]   # fx fy cx cy
  distortion: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
  resolution: [752, 480]
  T_BS: [...]                           # 4×4 row-major; cam0 → body (Kalibr)

cam1:
  intrinsics: [457.587, 456.134, 379.999, 255.238]
  distortion: [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]
  resolution: [752, 480]
  T_BS: [...]                           # cam1 → body

stereo_baseline: 0.11007               # ‖t_{c₀c₁}‖ in metres

imu:
  gyro_noise:  1.6968e-04              # rad/s/√Hz   — gyroscope white noise
  gyro_walk:   1.9393e-05              # rad/s²/√Hz  — gyroscope bias walk
  accel_noise: 2.0000e-03             # m/s²/√Hz    — accelerometer white noise
  accel_walk:  3.0000e-03             # m/s³/√Hz    — accelerometer bias walk
  rate_hz: 200
  T_BS: [identity]                     # IMU = body frame for EuRoC

gravity: [0.0, 0.0, -9.81]            # overridden at runtime by gravity estimator
```
