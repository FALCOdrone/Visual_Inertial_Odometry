"""Stereo rectifier node.

Waits for camera_info on /cam0/camera_info and /cam1/camera_info to obtain
intrinsics and distortion coefficients, then reads stereo extrinsics (T_BS)
from the config YAML to compute rectification maps.  Republishes rectified
grayscale images at /cam0/image_rect and /cam1/image_rect.

Subscribed topics:
    /cam0/camera_info  (sensor_msgs/CameraInfo)  — intrinsics + distortion
    /cam1/camera_info  (sensor_msgs/CameraInfo)  — intrinsics + distortion
    /cam0/image_raw    (sensor_msgs/Image)
    /cam1/image_raw    (sensor_msgs/Image)

Published topics:
    /cam0/image_rect   (sensor_msgs/Image, mono8)
    /cam1/image_rect   (sensor_msgs/Image, mono8)
"""

import numpy as np
import cv2
import yaml

import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import (  # type: ignore
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
)
from sensor_msgs.msg import Image, CameraInfo  # type: ignore
from message_filters import Subscriber, ApproximateTimeSynchronizer  # type: ignore


class StereoRectifierNode(Node):
    def __init__(self):
        super().__init__("stereo_rectifier_node")

        self.declare_parameter("config_path", "")
        config_path = self.get_parameter("config_path").value
        if not config_path:
            raise RuntimeError("config_path parameter is required")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        self._cam0_info: CameraInfo | None = None
        self._cam1_info: CameraInfo | None = None
        self._map0x = self._map0y = self._map1x = self._map1y = None

        self._setup_topics()
        self.get_logger().info(
            "StereoRectifierNode waiting for /cam0/camera_info and /cam1/camera_info ..."
        )

    # ── Topic setup ────────────────────────────────────────────────────────────

    def _setup_topics(self):
        ci_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(CameraInfo, "/cam0/camera_info", self._ci0_cb, ci_qos)
        self.create_subscription(CameraInfo, "/cam1/camera_info", self._ci1_cb, ci_qos)

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._cam0_sub = Subscriber(self, Image, "/cam0/image_raw", qos_profile=img_qos)
        self._cam1_sub = Subscriber(self, Image, "/cam1/image_raw", qos_profile=img_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._cam0_sub, self._cam1_sub], queue_size=10, slop=0.01
        )
        self._sync.registerCallback(self._image_cb)

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._rect0_pub = self.create_publisher(Image, "/cam0/image_rect", pub_qos)
        self._rect1_pub = self.create_publisher(Image, "/cam1/image_rect", pub_qos)

    # ── CameraInfo callbacks ────────────────────────────────────────────────────

    def _ci0_cb(self, msg: CameraInfo):
        if self._cam0_info is None:
            self._cam0_info = msg
            self._try_build_maps()

    def _ci1_cb(self, msg: CameraInfo):
        if self._cam1_info is None:
            self._cam1_info = msg
            self._try_build_maps()

    def _try_build_maps(self):
        if self._cam0_info is None or self._cam1_info is None:
            return

        ci0 = self._cam0_info
        ci1 = self._cam1_info

        K0 = np.array(ci0.k, dtype=np.float64).reshape(3, 3)
        K1 = np.array(ci1.k, dtype=np.float64).reshape(3, 3)
        d0 = np.array(ci0.d, dtype=np.float64)
        d1 = np.array(ci1.d, dtype=np.float64)

        # Stereo extrinsics from config (not available in a single CameraInfo)
        cam0_cfg = self._config["cam0"]
        cam1_cfg = self._config["cam1"]
        T_b_c0 = np.array(cam0_cfg["T_BS"], dtype=np.float64).reshape(4, 4)
        T_b_c1 = np.array(cam1_cfg["T_BS"], dtype=np.float64).reshape(4, 4)
        T_c1_c0 = np.linalg.inv(T_b_c1) @ T_b_c0
        R = T_c1_c0[:3, :3]
        t = T_c1_c0[:3, 3]

        w, h = ci0.width, ci0.height
        R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
            K0, d0, K1, d1, (w, h), R, t, alpha=0
        )
        self._map0x, self._map0y = cv2.initUndistortRectifyMap(
            K0, d0, R0, P0, (w, h), cv2.CV_32FC1
        )
        self._map1x, self._map1y = cv2.initUndistortRectifyMap(
            K1, d1, R1, P1, (w, h), cv2.CV_32FC1
        )
        self.get_logger().info(
            f"Rectification maps built ({w}x{h}) — publishing /cam0/image_rect, /cam1/image_rect"
        )

    # ── Image callback ─────────────────────────────────────────────────────────

    def _image_cb(self, cam0_msg: Image, cam1_msg: Image):
        if self._map0x is None:
            return  # still waiting for camera_info

        img0 = self._to_gray(cam0_msg)
        img1 = self._to_gray(cam1_msg)
        rect0 = cv2.remap(img0, self._map0x, self._map0y, cv2.INTER_LINEAR)
        rect1 = cv2.remap(img1, self._map1x, self._map1y, cv2.INTER_LINEAR)
        self._rect0_pub.publish(
            self._to_image_msg(rect0, cam0_msg.header.stamp, "cam0_rect")
        )
        self._rect1_pub.publish(
            self._to_image_msg(rect1, cam1_msg.header.stamp, "cam1_rect")
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _to_gray(self, msg: Image) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ("mono8", "8UC1"):
            return img[:, :, 0]
        code = cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)

    def _to_image_msg(self, img: np.ndarray, stamp, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "mono8"
        msg.step = msg.width
        msg.data = img.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    try:
        node = StereoRectifierNode()
    except Exception as e:
        print(f"[FATAL] StereoRectifierNode failed to start: {e}")
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
