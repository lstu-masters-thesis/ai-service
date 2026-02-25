# headpose.py

import cv2
import numpy as np


def estimate_head_pose(landmarks, frame_shape):
    h, w, _ = frame_shape

    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),  # Chin
        (landmarks[33].x * w, landmarks[33].y * h),    # Left eye
        (landmarks[263].x * w, landmarks[263].y * h),  # Right eye
        (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth
        (landmarks[291].x * w, landmarks[291].y * h)   # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)

    return R


def estimate_distance(landmarks, frame_width, focal_length=900):
    left = landmarks[33]
    right = landmarks[263]

    x1 = left.x * frame_width
    x2 = right.x * frame_width

    ipd_pixels = abs(x2 - x1)

    real_ipd = 0.063  # 63 mm average

    if ipd_pixels < 1:
        return 0.6

    distance = (focal_length * real_ipd) / ipd_pixels
    return distance