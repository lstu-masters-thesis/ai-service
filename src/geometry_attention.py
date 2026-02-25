# geometry_attention.py

import numpy as np


def compute_attention(gaze_eye, R_head, distance, screen_width=0.5):
    """
    gaze_eye: np.array (3,)
    R_head: np.array (3,3)  <-- rotation from solvePnP
    distance: float (meters)
    screen_width: float (meters)
    """

    # ------------------------
    # Normalize gaze
    # ------------------------
    gaze_eye = gaze_eye / np.linalg.norm(gaze_eye)

    # ------------------------
    # IMPORTANT FIX:
    # OpenCV gives rotation object->camera
    # We need camera space, so use transpose (inverse)
    # ------------------------
    gaze_cam = R_head.T @ gaze_eye
    gaze_cam = gaze_cam / np.linalg.norm(gaze_cam)

    # ------------------------
    # Screen normal
    # IMPORTANT: webcam looks along +Z in our setup
    # ------------------------
    screen_normal = np.array([0, 0, 1])

    # ------------------------
    # Angle between gaze and screen normal
    # ------------------------
    dot = np.clip(np.dot(gaze_cam, screen_normal), -0.999999, 0.999999)
    theta = np.arccos(dot)

    # ------------------------
    # Adaptive attention cone
    # ------------------------
    alpha = np.arctan((screen_width / 2) / distance)

    attention = 1 if theta <= alpha else 0

    return attention, np.degrees(theta), np.degrees(alpha)