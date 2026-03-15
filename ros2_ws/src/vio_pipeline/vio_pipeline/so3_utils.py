"""
so3_utils.py
============
Shared SO(3) and quaternion utilities for the VIO pipeline.

Quaternion convention: [x, y, z, w] (Hamilton, scalar-last).
"""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion [x,y,z,w] (Shepperd method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, both [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ],
        dtype=np.float64,
    )


def exp_so3_quat(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential: rotation vector phi -> quaternion [x,y,z,w]."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        half = phi * 0.5
        return np.array([half[0], half[1], half[2], 1.0]) / np.sqrt(
            1.0 + 0.25 * angle * angle
        )
    axis = phi / angle
    s = np.sin(angle * 0.5)
    return np.array(
        [axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle * 0.5)],
        dtype=np.float64,
    )


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential: rotation vector phi -> 3x3 rotation matrix (Rodrigues)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: 3x3 rotation matrix -> rotation vector phi (3,)."""
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(theta - np.pi) < 1e-6:
        # Near pi: use eigenvector of R corresponding to eigenvalue 1
        # R = I + 2 sin(theta) K + 2 K^2  =>  near pi, R + I = 2 v v^T
        M = R + np.eye(3)
        col = np.argmax(np.sum(M * M, axis=0))
        v = M[:, col]
        v = v / np.linalg.norm(v)
        return v * theta
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3): J_r(phi), 3x3."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) - 0.5 * skew(phi)
    K = skew(phi / theta)
    return (np.eye(3, dtype=np.float64)
            - ((1.0 - np.cos(theta)) / (theta * theta)) * skew(phi)
            + ((theta - np.sin(theta)) / (theta**3)) * (skew(phi) @ skew(phi)))


def inv_right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3): J_r^{-1}(phi), 3x3."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + 0.5 * skew(phi)
    K = skew(phi)
    half_theta = theta * 0.5
    cot_half = np.cos(half_theta) / np.sin(half_theta) if abs(np.sin(half_theta)) > 1e-12 else 1.0 / half_theta
    return (np.eye(3, dtype=np.float64)
            + 0.5 * K
            + (1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))) * (K @ K))
