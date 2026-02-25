import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np

class VIOBackend(Node):
    def __init__(self):
        super().__init__('vio_backend')

        # --- EKF State Vector ---
        # Nominal State: [x, y, z, vx, vy, vz]
        # Orientation (self.quat) is kept separate.
        self.state = np.zeros(6) 
        self.quat = np.array([0.0, 0.0, 0.0, 1.0]) # x, y, z, w
        
        # Error State: [delta_p(3), delta_v(3), delta_theta(3), delta_bg(3), delta_ba(3)]
        self.bg = np.zeros(3) # Gyro bias
        self.ba = np.zeros(3) # Accel bias
        
        # Covariance Matrix P
        self.P = np.eye(15) * 0.01
        self.P[9:15, 9:15] *= 0.01 

        # Constants
        self.g = np.array([0.0, 0.0, -9.81])
        self.prev_imu_time = None

        # --- Gravity Alignment Initialization ---
        self.initialized = False
        self.init_accel_samples = []
        self.init_gyro_samples = []
        self.INIT_SAMPLE_COUNT = 50  # ~0.25s at 200Hz

        # --- ROS Communication ---
        self.odom_pub = self.create_publisher(Odometry, '/odom/vio', 10)
        self.path_pub = self.create_publisher(Path, '/odom/vio/path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"
        self.create_subscription(Imu, '/imu0', self.imu_callback, 100)
        self.create_subscription(PoseWithCovarianceStamped, '/vio/visual_odom', self.vo_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/vio/loop_correction', self.loop_correction_callback, 10)

        self.get_logger().info("VIO Backend Started with Dynamic Covariance + Loop Closure Support...")

    def skew_symmetric(self, v):
        """Creates a skew-symmetric matrix from a 3-element vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def _initialize_from_gravity(self):
        """
        Estimate initial orientation from stationary accelerometer readings.
        EKF operates in the INITIAL BODY FRAME so VO measurements match directly.
        Gravity is expressed in body frame coordinates.
        """
        acc_mean = np.mean(self.init_accel_samples, axis=0)
        gyro_mean = np.mean(self.init_gyro_samples, axis=0)

        # Initialize gyro bias from stationary readings
        self.bg = gyro_mean.copy()

        # Gravity in body frame: accelerometer reads reaction to gravity,
        # so actual gravity direction = -acc_mean
        g_body = -acc_mean
        g_magnitude = np.linalg.norm(g_body)
        # Use measured magnitude for gravity vector in body frame
        self.g = g_body * (9.81 / g_magnitude)

        # Compute R_init (body→world) for RViz display only
        g_body_norm = g_body / g_magnitude
        g_world_norm = np.array([0.0, 0.0, -1.0])
        v = np.cross(g_body_norm, g_world_norm)
        s = np.linalg.norm(v)
        c = np.dot(g_body_norm, g_world_norm)
        if s < 1e-6:
            self.R_init = np.eye(3) if c > 0 else np.diag([-1, -1, 1])
        else:
            vx = self.skew_symmetric(v)
            self.R_init = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

        # EKF starts in body frame: quaternion = identity
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.initialized = True
        self.get_logger().info(
            f"Gravity init (body frame). "
            f"g_body: [{self.g[0]:.2f}, {self.g[1]:.2f}, {self.g[2]:.2f}], "
            f"Gyro bias: [{gyro_mean[0]:.4f}, {gyro_mean[1]:.4f}, {gyro_mean[2]:.4f}]")

    def imu_callback(self, msg):
        """ PREDICTION STEP """
        acc_raw = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro_raw = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # --- Gravity Alignment Phase ---
        if not self.initialized:
            self.init_accel_samples.append(acc_raw)
            self.init_gyro_samples.append(gyro_raw)
            if len(self.init_accel_samples) >= self.INIT_SAMPLE_COUNT:
                self._initialize_from_gravity()
            return

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_imu_time is None:
            self.prev_imu_time = current_time
            return
        
        dt = current_time - self.prev_imu_time
        self.prev_imu_time = current_time

        # Robustness: Ignore if dt is negative or too large (dropped packets)
        if dt <= 0 or dt > 1.0:
            self.get_logger().warn(f"Invalid IMU dt: {dt}. Skipping prediction.")
            return

        # 2. Remove Biases
        acc = acc_raw - self.ba
        gyro = gyro_raw - self.bg

        # 3. Nominal State Update
        r_obj = R.from_quat(self.quat)
        rot_mat = r_obj.as_matrix()

        # p = p + v*dt + 0.5*a*dt^2
        acc_world = rot_mat @ acc + self.g
        self.state[0:3] += self.state[3:6] * dt + 0.5 * acc_world * dt**2
        
        # v = v + a*dt
        self.state[3:6] += acc_world * dt

        # q = q * delta_q
        angle_delta = gyro * dt
        delta_q = R.from_rotvec(angle_delta)
        self.quat = (r_obj * delta_q).as_quat()

        # 4. Error State Covariance Prediction
        # F = State Transition Matrix
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt  # Pos w.r.t Vel
        F[3:6, 6:9] = -rot_mat @ self.skew_symmetric(acc) * dt # Vel w.r.t Angle
        F[3:6, 12:15] = -rot_mat * dt # Vel w.r.t Accel Bias
        
        # IMPROVEMENT: Angle w.r.t Angle (Rotation error propagation)
        # delta_theta_k+1 = (I - [omega]x * dt) * delta_theta_k - delta_bg
        F[6:9, 6:9] = np.eye(3) - self.skew_symmetric(gyro) * dt
        F[6:9, 9:12] = -np.eye(3) * dt # Angle w.r.t Gyro Bias

        # Process Noise Q
        Q = np.zeros((15, 15))
        noise_acc = 0.05
        noise_gyro = 0.01
        noise_ba_walk = 0.0005
        noise_bg_walk = 0.00005
        
        Q[3:6, 3:6] = np.eye(3) * noise_acc**2 * dt**2
        Q[6:9, 6:9] = np.eye(3) * noise_gyro**2 * dt**2
        Q[9:12, 9:12] = np.eye(3) * noise_bg_walk**2 * dt
        Q[12:15, 12:15] = np.eye(3) * noise_ba_walk**2 * dt
        
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)  # Enforce symmetry

        self.publish_odometry(msg.header)

    def vo_callback(self, msg):
        """ UPDATE STEP: Position Correction """
        if not self.initialized:
            return
        
        # VO position is in initial body frame = EKF frame, use directly
        z_pos = np.array([msg.pose.pose.position.x, 
                          msg.pose.pose.position.y, 
                          msg.pose.pose.position.z])
        
        current_pos = self.state[0:3]
        error = z_pos - current_pos

        # IMPROVEMENT: Use the dynamic covariance from the frontend
        # msg.pose.covariance is a flat 36-float array. We need the 3x3 diagonal for position.
        # indices 0, 7, 14 correspond to cov(x,x), cov(y,y), cov(z,z)
        cov_x = msg.pose.covariance[0]
        cov_y = msg.pose.covariance[7]
        cov_z = msg.pose.covariance[14]
        
        if cov_x > 0:
            R_cov = np.diag([cov_x, cov_y, cov_z])
        else:
            # Fallback if frontend sends zeros
            R_cov = np.eye(3) * 0.1

        # H Matrix (Measurement maps to state)
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)

        # Kalman Update
        # S = H*P*H' + R
        S = H @ self.P @ H.T + R_cov
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().error("Singular Matrix in EKF Update. Skipping.")
            return

        # Error State
        delta_x = K @ error

        # Correct Nominal State
        self.state[0:3] += delta_x[0:3]
        self.state[3:6] += delta_x[3:6]
        
        delta_theta = delta_x[6:9]
        delta_q = R.from_rotvec(delta_theta)
        self.quat = (R.from_quat(self.quat) * delta_q).as_quat()
        self.quat = self.quat / np.linalg.norm(self.quat) # Normalize quaternion

        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # Update Covariance (Joseph form for numerical stability)
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Enforce symmetry

    def loop_correction_callback(self, msg):
        """Apply pose correction from loop closure using EKF-style blended update."""
        corrected_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        corrected_quat = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])

        # Use loop closure covariance from message
        lc_cov_pos = np.diag([
            msg.pose.covariance[0],
            msg.pose.covariance[7],
            msg.pose.covariance[14],
        ])
        lc_cov_pos = np.maximum(lc_cov_pos, np.eye(3) * 0.001)  # Floor

        # --- Position Update via EKF ---
        H_pos = np.zeros((3, 15))
        H_pos[0:3, 0:3] = np.eye(3)

        pos_error = corrected_pos - self.state[0:3]
        S = H_pos @ self.P @ H_pos.T + lc_cov_pos
        try:
            K = self.P @ H_pos.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().error("Singular matrix in loop correction. Skipping.")
            return

        delta_x = K @ pos_error
        self.state[0:3] += delta_x[0:3]
        self.state[3:6] += delta_x[3:6]

        # Orientation correction
        delta_theta = delta_x[6:9]
        delta_q = R.from_rotvec(delta_theta)
        self.quat = (R.from_quat(self.quat) * delta_q).as_quat()
        self.quat = self.quat / np.linalg.norm(self.quat)

        # Bias corrections
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # Update covariance (Joseph form)
        IKH = np.eye(15) - K @ H_pos
        self.P = IKH @ self.P @ IKH.T + K @ lc_cov_pos @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        pos_jump = np.linalg.norm(pos_error)
        self.get_logger().info(
            f"Loop correction blended: position adjustment = {pos_jump:.3f}m"
        )

    def publish_odometry(self, header):
        odom = Odometry()
        odom.header.stamp = header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        # Transform from body frame to world frame (Z-up) for RViz display
        pos_world = self.R_init @ self.state[0:3]
        q_world = (R.from_matrix(self.R_init) * R.from_quat(self.quat)).as_quat()

        odom.pose.pose.position.x = float(pos_world[0])
        odom.pose.pose.position.y = float(pos_world[1])
        odom.pose.pose.position.z = float(pos_world[2])
        
        odom.pose.pose.orientation.x = float(q_world[0])
        odom.pose.pose.orientation.y = float(q_world[1])
        odom.pose.pose.orientation.z = float(q_world[2])
        odom.pose.pose.orientation.w = float(q_world[3])

        pose_cov = np.zeros((6, 6))
        pose_cov[0:3, 0:3] = self.P[0:3, 0:3] 
        pose_cov[0:3, 3:6] = self.P[0:3, 6:9] 
        pose_cov[3:6, 0:3] = self.P[6:9, 0:3] 
        pose_cov[3:6, 3:6] = self.P[6:9, 6:9] 
        odom.pose.covariance = pose_cov.flatten().tolist()
        
        self.odom_pub.publish(odom)

        # --- Append to VIO Path ---
        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self.path_msg.header.stamp = header.stamp
        self.path_msg.poses.append(pose_stamped)
        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses = self.path_msg.poses[-5000:]
        self.path_pub.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VIOBackend()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()