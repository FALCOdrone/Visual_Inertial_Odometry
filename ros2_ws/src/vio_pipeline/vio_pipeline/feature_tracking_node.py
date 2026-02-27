import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # type: ignore
from message_filters import Subscriber, ApproximateTimeSynchronizer  # type: ignore

import numpy as np
import cv2
import yaml

from vio_pipeline.feature_extraction import FeatureExtractor


def stamp_to_ns(stamp):
    """Convert ROS2 stamp to nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class FeatureTrackingNode(Node):
    def __init__(self):
        super().__init__("feature_tracking_node")

        self.declare_parameter("config_path", "")
        self.declare_parameter("max_matches", 50)
        self.declare_parameter("device", "")
        self.declare_parameter("circular_check_threshold", 2.0)

        config_path = self.get_parameter("config_path").value
        self.max_matches = self.get_parameter("max_matches").value
        device_param = self.get_parameter("device").value
        self.circular_threshold = self.get_parameter("circular_check_threshold").value

        if config_path:
            self.load_config(config_path)

        device_kwarg = {"device": device_param} if device_param else {}
        self.extractor = FeatureExtractor(**device_kwarg)

        if self.extractor.superpoint is None:
            self.get_logger().fatal(
                "SuperPoint/LightGlue models failed to load. "
                "Install with: pip install lightglue"
            )
            raise RuntimeError("lightglue not installed")

        # Previous-frame state for temporal matching
        self._prev_left_features = None
        self._prev_right_features = None

        self._setup_ros_topics()
        self.get_logger().info("FeatureTrackingNode initialized")

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config
        self.get_logger().info("Config loaded successfully")

    def _setup_ros_topics(self):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cam0_sub = Subscriber(self, Image, "/cam0/image_raw", qos_profile=qos)
        self.cam1_sub = Subscriber(self, Image, "/cam1/image_raw", qos_profile=qos)

        self.time_sync = ApproximateTimeSynchronizer(
            [self.cam0_sub, self.cam1_sub], queue_size=10, slop=0.1
        )
        self.time_sync.registerCallback(self.stereo_callback)

        self.viz_pub = self.create_publisher(Image, "/features/viz", 10)
        self.temporal_viz_pub = self.create_publisher(
            Image, "/features/temporal_viz", 10
        )

        self.get_logger().info("ROS topics initialized")

    def _image_msg_to_numpy(self, msg):
        """Convert sensor_msgs/Image to a grayscale numpy array."""
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ("mono8", "8UC1"):
            return img[:, :, 0]
        # bgr8 / rgb8 -> grayscale
        code = cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)

    def _numpy_bgr_to_image_msg(self, img, stamp, frame_id="cam0"):
        """Convert a BGR numpy image to sensor_msgs/Image."""
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        return msg

    def stereo_callback(self, cam0_msg, cam1_msg):
        """Process a synchronized stereo image pair."""
        ts_ns = stamp_to_ns(cam0_msg.header.stamp)

        left_img = self._image_msg_to_numpy(cam0_msg)
        right_img = self._image_msg_to_numpy(cam1_msg)

        result = self.extractor.process_full_frame(
            left_img,
            right_img,
            prev_left_features=self._prev_left_features,
            prev_right_features=self._prev_right_features,
            pixel_threshold=self.circular_threshold,
        )

        if result is None:
            self.get_logger().warn("Feature extraction returned no result")
            return

        left_features = result["left_features"]
        right_features = result["right_features"]
        stereo_matches = result["stereo_matches"]

        n_kpts_left = left_features["keypoints"].shape[1]
        n_kpts_right = right_features["keypoints"].shape[1]
        valid_stereo = int(np.sum(stereo_matches["matches"] >= 0))

        # --- Temporal / circular-check stats ---
        circular_tracks = result["circular_tracks"]
        temporal_matches = result["temporal_matches"]

        if temporal_matches is not None:
            n_temporal_left = int(
                np.sum(temporal_matches["temporal_left"]["matches"] >= 0)
            )
            n_circular = circular_tracks["count"] if circular_tracks else 0

            self.get_logger().info(
                f"ts={ts_ns} | "
                f"kpts_L={n_kpts_left} kpts_R={n_kpts_right} | "
                f"stereo={valid_stereo} "
                f"temporal={n_temporal_left} "
                f"circular={n_circular} "
                f"(thr={self.circular_threshold:.1f}px)"
            )
        else:
            self.get_logger().info(
                f"ts={ts_ns} | kpts_left={n_kpts_left} "
                f"kpts_right={n_kpts_right} stereo={valid_stereo} "
                f"[first frame — no temporal]"
            )

        # --- Stereo visualisation ---
        vis = self.extractor.visualize_matches(
            left_img, right_img, stereo_matches, max_matches=self.max_matches
        )
        if vis is not None:
            self.viz_pub.publish(
                self._numpy_bgr_to_image_msg(vis, cam0_msg.header.stamp)
            )

        # --- Temporal / circular-check visualisation ---
        if circular_tracks is not None:
            t_vis = self.extractor.visualize_temporal_tracks(
                left_img,
                circular_tracks,
                max_tracks=self.max_matches,
            )
            if t_vis is not None:
                self.temporal_viz_pub.publish(
                    self._numpy_bgr_to_image_msg(t_vis, cam0_msg.header.stamp)
                )

        # --- Store current frame as previous ---
        self._prev_left_features = result["left_features"]
        self._prev_right_features = result["right_features"]


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FeatureTrackingNode()
    except RuntimeError as e:
        print(f"[FATAL] Failed to start FeatureTrackingNode: {e}")
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
