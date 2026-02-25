import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct


class Keyframe:
    """Stores all data associated with a single keyframe."""

    __slots__ = [
        "id",
        "timestamp",
        "pose",
        "descriptors",
        "keypoints_2d",
        "landmarks_3d",
        "image",
    ]

    def __init__(self, kf_id, timestamp, pose, descriptors, keypoints_2d, landmarks_3d, image):
        self.id = kf_id
        self.timestamp = timestamp
        self.pose = pose                  # 4x4 global pose (body frame)
        self.descriptors = descriptors    # ORB descriptors (Nx32 uint8)
        self.keypoints_2d = keypoints_2d  # Nx2 pixel coords
        self.landmarks_3d = landmarks_3d  # Nx3 world-frame 3D points
        self.image = image                # Grayscale image for visualization


class VIOMapper(Node):
    def __init__(self):
        super().__init__("vio_mapper")
        self.bridge = CvBridge()

        # --- Camera Intrinsics (must match frontend) ---
        self.fx, self.fy = 458.654, 457.296
        self.cx, self.cy = 367.215, 248.375
        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )
        self.baseline = 0.11

        # --- Camera-to-Body rotation (must match frontend) ---
        self.R_body_cam = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )

        # --- ORB Detector ---
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

        # --- Keyframe Database ---
        self.keyframes = []
        self.next_kf_id = 0
        self.last_kf_pose = None

        # --- Keyframe Selection Thresholds ---
        self.min_translation = 0.3    # meters
        self.min_rotation_deg = 15.0  # degrees
        self.min_kf_interval = 5      # minimum frames between keyframes

        self.frame_count = 0

        # --- Stereo Matching Parameters ---
        self.lk_params = dict(
            winSize=(21, 21),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # --- ROS Communication ---
        self.keyframe_pub = self.create_publisher(
            PoseStamped, "/vio/keyframe", 10
        )
        self.map_points_pub = self.create_publisher(
            PointCloud2, "/vio/map_points", 10
        )

        # Subscribe to VO pose
        self.create_subscription(
            PoseWithCovarianceStamped, "/vio/visual_odom", self.vo_pose_callback, 10
        )

        # Subscribe to stereo images (synced)
        left_sub = message_filters.Subscriber(self, Image, "/cam0/image_raw")
        right_sub = message_filters.Subscriber(self, Image, "/cam1/image_raw")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], 10, 0.02
        )
        self.ts.registerCallback(self.image_callback)

        # Buffers for synchronization between pose and image callbacks
        self.latest_pose = None
        self.latest_pose_matrix = None

        self.get_logger().info("VIO Mapper Started. Waiting for data...")

    def vo_pose_callback(self, msg):
        """Cache the latest VO pose for keyframe selection."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        t = np.array([pos.x, pos.y, pos.z])
        q = np.array([ori.x, ori.y, ori.z, ori.w])
        rot = R.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t

        self.latest_pose = msg
        self.latest_pose_matrix = T

    def image_callback(self, left_msg, right_msg):
        """Process stereo images when a new VO pose is available."""
        if self.latest_pose_matrix is None:
            return

        self.frame_count += 1
        current_pose = self.latest_pose_matrix.copy()

        # --- Keyframe Selection ---
        if not self._should_create_keyframe(current_pose):
            return

        try:
            img_l = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
            img_r = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # --- Extract ORB Features ---
        kp_orb, descriptors = self.orb.detectAndCompute(img_l, None)

        if descriptors is None or len(kp_orb) < 20:
            self.get_logger().warn("Not enough ORB features for keyframe.")
            return

        # --- Triangulate 3D Points (Camera Frame) ---
        kp_pixels = np.array([k.pt for k in kp_orb], dtype=np.float32).reshape(-1, 1, 2)
        pts_cam, valid_mask = self._triangulate_stereo(img_l, img_r, kp_pixels)

        if np.sum(valid_mask) < 10:
            self.get_logger().warn("Not enough stereo matches for keyframe.")
            return

        # Filter to valid points only
        valid_idx = valid_mask.flatten()
        descriptors = descriptors[valid_idx]
        kp_pixels_valid = kp_pixels[valid_idx].reshape(-1, 2)
        pts_cam_valid = pts_cam[valid_idx]

        # --- Transform 3D Points to World Frame ---
        # pts_cam is in camera frame -> rotate to body -> then to world
        pts_body = (self.R_body_cam @ pts_cam_valid.T).T
        pts_world = (current_pose[:3, :3] @ pts_body.T + current_pose[:3, 3:4]).T

        # --- Create Keyframe ---
        kf = Keyframe(
            kf_id=self.next_kf_id,
            timestamp=left_msg.header.stamp,
            pose=current_pose,
            descriptors=descriptors,
            keypoints_2d=kp_pixels_valid,
            landmarks_3d=pts_world.astype(np.float32),
            image=img_l,
        )
        self.keyframes.append(kf)
        self.last_kf_pose = current_pose.copy()
        self.next_kf_id += 1
        self.frame_count = 0

        self.get_logger().info(
            f"Keyframe {kf.id} created: {len(descriptors)} features, "
            f"{len(self.keyframes)} total keyframes"
        )

        # --- Publish Keyframe Pose ---
        self._publish_keyframe_pose(kf, left_msg.header)

        # --- Publish Map Points ---
        self._publish_map_points(left_msg.header)

    def _should_create_keyframe(self, current_pose):
        """Decide whether to create a new keyframe based on motion thresholds."""
        # Always create the first keyframe
        if self.last_kf_pose is None:
            return True

        # Minimum frame interval
        if self.frame_count < self.min_kf_interval:
            return False

        # Translation check
        delta_t = np.linalg.norm(current_pose[:3, 3] - self.last_kf_pose[:3, 3])
        if delta_t > self.min_translation:
            return True

        # Rotation check
        R_rel = self.last_kf_pose[:3, :3].T @ current_pose[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        if np.degrees(angle) > self.min_rotation_deg:
            return True

        return False

    def _triangulate_stereo(self, img_l, img_r, kp_l):
        """Triangulate 3D points from stereo pair. Returns points in camera frame."""
        kp_l = kp_l.astype(np.float32)
        kp_r, st, _ = cv2.calcOpticalFlowPyrLK(
            img_l, img_r, kp_l, None, **self.lk_params
        )

        n_points = len(kp_l)
        pts_3d = np.zeros((n_points, 3), dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            if st[i][0] != 1:
                continue
            pl = kp_l[i].ravel()
            pr = kp_r[i].ravel()
            d = pl[0] - pr[0]

            if d > 1.0:
                z = (self.fx * self.baseline) / d
                x = (pl[0] - self.cx) * z / self.fx
                y = (pl[1] - self.cy) * z / self.fy
                # Reject points that are too far away
                if z < 30.0:
                    pts_3d[i] = [x, y, z]
                    valid[i] = True

        return pts_3d, valid

    def _publish_keyframe_pose(self, kf, header):
        """Publish keyframe pose as PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = kf.timestamp
        msg.header.frame_id = "odom"

        msg.pose.position.x = float(kf.pose[0, 3])
        msg.pose.position.y = float(kf.pose[1, 3])
        msg.pose.position.z = float(kf.pose[2, 3])

        q = R.from_matrix(kf.pose[:3, :3]).as_quat()
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

        self.keyframe_pub.publish(msg)

    def _publish_map_points(self, header):
        """Publish all landmarks as a PointCloud2 message."""
        all_points = []
        for kf in self.keyframes:
            all_points.append(kf.landmarks_3d)

        if not all_points:
            return

        points = np.vstack(all_points)

        # Subsample if too many points
        max_points = 10000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        msg = PointCloud2()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "odom"
        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_bigendian = False
        msg.is_dense = True

        buffer = []
        for p in points:
            buffer.append(struct.pack("fff", float(p[0]), float(p[1]), float(p[2])))
        msg.data = b"".join(buffer)

        self.map_points_pub.publish(msg)

    def get_keyframes(self):
        """Public accessor for the keyframe database (used by loop closure)."""
        return self.keyframes


def main(args=None):
    rclpy.init(args=args)
    node = VIOMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
