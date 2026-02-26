# Visual-Inertial Odometry (VIO)

A modular **Visual-Inertial Odometry** pipeline built on **ROS 2 (Humble)**.  
It fuses stereo camera images with IMU measurements through an Extended Kalman Filter to estimate 6-DoF pose in real time, and includes keyframe-based mapping with loop closure for drift correction.

---

## Pipeline Overview

```
┌────────────┐   stereo images    ┌──────────┐  visual odom   ┌──────────┐
│  Rosbag    │ ─────────────────► │ Frontend │ ─────────────► │ Backend  │
│  Playback  │   IMU data         │ (Stereo  │                │  (EKF    │
│            │ ─────────────────► │   VO)    │                │  Fusion) │
└────────────┘                    └──────────┘                └────┬─────┘
      │                                                            │
      │  Vicon GT                                                  │ /odom/vio
      ▼                                                            │
┌─────────────┐                                                    │
│ Ground Truth│                                                    │
│  Publisher  │                                                    │
└──────┬──────┘                                                    │
       │                                                           │
       │  /ground_truth/odom                                       │
       │                                                           ▼
       ▼                                                 ┌───────────────────┐
┌───────────────────┐                                    │    Trajectory      │
│    Trajectory      │  ◄── /odom/vio                    │    Comparator      │
│  (CSV diagnostics) │                                   │  (CSV diagnostics) │
└───────────────────┘                                    └───────────────────┘
```

### Node Descriptions

| Node | Executable | Description |
|---|---|---|
| **Frontend** | `frontend` | Stereo visual odometry — detects ORB features, tracks them with optical flow, triangulates 3-D points from stereo, and estimates frame-to-frame motion via PnP RANSAC. |
| **Backend** | `backend` | EKF-based sensor fusion — propagates state with IMU (prediction step) and corrects with VO and loop closure poses (update steps). Publishes fused odometry and path. |
| **TF Broadcaster** | `tf_broadcaster` | Publishes the full TF frame tree (dynamic + static) so RViz and `tf2` tools can resolve all coordinate frames. |
| **Ground Truth Publisher** | `ground_truth_pub` | Converts Vicon motion-capture transforms into `Odometry` and `Path` messages calibrated to the body-frame origin. |
| **Trajectory Comparator** | `trajectory_comparator` | Time-aligns VIO and ground-truth trajectories, computes frame-consistent position & orientation errors, and writes a CSV log for offline analysis. |

---

## Coordinate Frames

This project strictly adheres to the **ROS FLU (Forward-Left-Up)** convention internally for all nodes:
*   **X-axis:** Forward
*   **Y-axis:** Left
*   **Z-axis:** Up

**Important Context on the EuRoC Dataset:**
The visual-inertial sensor system on the EuRoC MAV contains arbitrary internal rotations. The Camera natively provides data as `X-right, Y-down, Z-forward`, while the IMU natively provides measurements as `X-forward, Y-left, Z-up` (FLU). 

To ensure stability in the EKF (Backend), the **Frontend** applies a hardcoded rotation (`R_body_cam`) to map all visual tracking vectors directly into the native IMU FLU body frame. As the Vicon Ground Truth is natively reported in FLU, the Ground Truth Publisher maps these translation and rotation values out of the box with no mathematical swaps required.

### TF Frame Tree

```
odom  (fixed / world frame)
├── base_link          ← EKF-fused pose   (dynamic, from /odom/vio_ekf)
│   ├── imu_link       ← static, identity (IMU co-located with body)
│   └── camera_link    ← static, R_body_cam rotation
├── visual_odom        ← VO-only pose     (dynamic, from /vio/visual_odom)
└── ground_truth       ← Vicon GT pose    (dynamic, from /ground_truth/odom)
```

The **TF Broadcaster** node publishes all of the above transforms. Static transforms (`base_link → imu_link`, `base_link → camera_link`) are published once on `/tf_static`. Dynamic transforms are published at the incoming message rate on `/tf`.

---

## Dataset

This project uses the [**EuRoC MAV Dataset**](https://projects.asl.ethz.ch/datasets/euroc-mav/) recorded on-board an AscTec Firefly hexacopter.

### Download

Download the **ASL / Machine Hall** or **Vicon Room** sequences in **ROS bag** format from the official page:

> **🔗 [EuRoC MAV Dataset — Download Page](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#downloads)**

The bags must be converted to **MCAP** format for ROS 2 compatibility. You can use [`rosbags`](https://pypi.org/project/rosbags/) to convert:

```bash
pip install rosbags
rosbags-convert <input.bag> --dst <output_dir>
```

### Included Sequences

Three difficulty levels from the Vicon Room 1 collection are used:

| Sequence | File | Size | Duration | Description |
|---|---|---|---|---|
| **Easy 1** | `easy1/easy1.mcap` | ~2.0 GB | ~147 s | Slow, smooth flight |
| **Medium 1** | `med1/med1.mcap` | ~1.2 GB | — | Moderate speed and dynamics |
| **Hard 1** | `hard1/hard1.mcap` | ~1.5 GB | — | Fast, aggressive maneuvers |

### Bag Topics

| Topic | Message Type | Description |
|---|---|---|
| `/cam0/image_raw` | `sensor_msgs/msg/Image` | Left camera (grayscale, 20 Hz) |
| `/cam1/image_raw` | `sensor_msgs/msg/Image` | Right camera (grayscale, 20 Hz) |
| `/imu0` | `sensor_msgs/msg/Imu` | IMU accelerometer + gyroscope (~200 Hz) |
| `/fcu/imu` | `sensor_msgs/msg/Imu` | Flight controller IMU |
| `/fcu/motor_speed` | `asctec_hl_comm/msg/MotorSpeed` | Rotor speed telemetry |
| `/vicon/firefly_sbx/firefly_sbx` | `geometry_msgs/msg/TransformStamped` | Vicon motion-capture ground truth |

---

## Requirements

### System

- **OS:** Ubuntu 22.04 (native or WSL2)
- **ROS 2:** Humble Hawksbill
- **Python:** 3.10+

### ROS 2 Packages

```
ros-humble-cv-bridge
ros-humble-message-filters
ros-humble-sensor-msgs
ros-humble-geometry-msgs
ros-humble-nav-msgs
ros-humble-rosbag2-storage-mcap
```

### Python Libraries

```
numpy
scipy
opencv-python   (or opencv-contrib-python)
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/FALCOdrone/Visual_Inertial_Odometry.git
cd Visual_Inertial_Odometry
```

### 2. Install ROS 2 Dependencies

```bash
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-message-filters \
                 ros-humble-rosbag2-storage-mcap
```

### 3. Install Python Dependencies

```bash
pip install numpy scipy opencv-python
```

### 4. Build the Workspace

```bash
cd vio_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 5. Place Dataset Bags

Download and place the MCAP bags in the project root so the directory structure matches:

```
Visual_Inertial_Odometry/
├── easy1/
│   ├── easy1.mcap
│   └── metadata.yaml
├── med1/
│   ├── med1.mcap
│   └── metadata.yaml
└── hard1/
    ├── hard1.mcap
    └── metadata.yaml
```

---

## Usage

### Launch the Full Pipeline

The provided launch file starts **all five nodes** and begins rosbag playback with a 2-second delay to let the nodes initialize:

```bash
# Source the workspace
source vio_ws/install/setup.bash

# Run with default settings (easy1 sequence)
ros2 launch vio_pkg vio_launch.py

# Specify a different bag and playback rate
ros2 launch vio_pkg vio_launch.py bag_path:=/path/to/hard1 bag_rate:=0.5
```

#### Launch Arguments

| Argument | Default | Description |
|---|---|---|
| `bag_path` | `/mnt/d/GITHUB/VIO/easy1` | Absolute path to the rosbag directory |
| `bag_rate` | `1.0` | Playback speed multiplier (e.g. `0.5` = half speed) |

### Run Individual Nodes

You can also start nodes independently for development:

```bash
# Terminal 1 – Frontend
ros2 run vio_pkg frontend

# Terminal 2 – Backend
ros2 run vio_pkg backend

# Terminal 3 – TF Broadcaster
ros2 run vio_pkg tf_broadcaster

# Terminal 4 – Ground Truth Publisher
ros2 run vio_pkg ground_truth_pub

# Terminal 5 – Trajectory Comparator
ros2 run vio_pkg trajectory_comparator

# Terminal 6 – Play bag
ros2 bag play /path/to/easy1 --clock --rate 1.0
```

### Visualize in RViz2

```bash
rviz2
```

Add the following displays:
- **Odometry** → topic `/odom/vio` (VIO estimate)
- **Path** → topic `/vio/path` (VIO trajectory)
- **Path** → topic `/ground_truth/path` (Ground truth trajectory)
- **TF** → shows all coordinate frames and their relationships


---

## File Structure

```
Visual_Inertial_Odometry/
│
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── .gitignore
│
├── easy1/                             # EuRoC — Easy sequence (git-ignored)
│   ├── easy1.mcap
│   └── metadata.yaml
├── med1/                              # EuRoC — Medium sequence (git-ignored)
│   ├── med1.mcap
│   └── metadata.yaml
├── hard1/                             # EuRoC — Hard sequence (git-ignored)
│   ├── hard1.mcap
│   └── metadata.yaml
│
└── vio_ws/                            # ROS 2 workspace
    └── src/
        └── vio_pkg/                   # Main ROS 2 Python package
            ├── package.xml            # Package manifest
            ├── setup.py               # Entry-point definitions
            ├── setup.cfg
            ├── launch/
            │   └── vio_launch.py      # Launch file (all nodes + bag play)
            ├── vio_pkg/               # Source code
            │   ├── __init__.py
            │   ├── frontend.py        # Stereo visual odometry node
            │   ├── backend.py         # EKF sensor-fusion node
            │   ├── tf_broadcaster.py  # TF frame tree broadcaster
            │   ├── ground_truth_pub.py# Vicon → Odometry converter
            │   └── trajectory_comparator.py  # Error analysis & CSV logger
            ├── test/                  # Test stubs
            └── resource/
                └── vio_pkg
```

---

## ROS 2 Topic Graph

### Published Topics

| Topic | Type | Publisher |
|---|---|---|
| `/visual_odom` | `PoseWithCovarianceStamped` | Frontend |
| `/odom/vio` | `Odometry` | Backend |
| `/vio/path` | `Path` | Backend |
| `/ground_truth/odom` | `Odometry` | Ground Truth Publisher |
| `/ground_truth/path` | `Path` | Ground Truth Publisher |
| `/tf` | `TFMessage` | TF Broadcaster (dynamic) |
| `/tf_static` | `TFMessage` | TF Broadcaster (static) |


### Subscribed Topics

| Topic | Subscriber(s) |
|---|---|
| `/cam0/image_raw` | Frontend |
| `/cam1/image_raw` | Frontend |
| `/imu0` | Backend |
| `/visual_odom` | Backend |
| `/vicon/firefly_sbx/firefly_sbx` | Ground Truth Publisher |
| `/odom/vio_ekf` | Trajectory Comparator, TF Broadcaster |
| `/ground_truth/odom` | Trajectory Comparator, TF Broadcaster |
| `/vio/visual_odom` | Backend, TF Broadcaster |

---

## Output

The **Trajectory Comparator** node writes a timestamped CSV file with per-sample diagnostics:

```
vio_comparison_YYYYMMDD_HHMMSS.csv
```

Columns include:
- `time_s` — elapsed time
- `gt_x/y/z`, `vio_x/y/z` — aligned ground truth and VIO positions
- `err_x/y/z`, `err_norm` — position error components and Euclidean norm
- `gt/vio_roll/pitch/yaw` — orientation in Euler angles (degrees)
- `err_roll/pitch/yaw` — orientation error

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.
