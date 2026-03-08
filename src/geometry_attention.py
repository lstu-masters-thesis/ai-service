# geometry_attention.py

import numpy as np


# Минимальный физически разумный полуугол конуса внимания (в градусах)
MIN_ALPHA_DEG = 25.0

# Дополнительный мягкий запас к теоретическому конусу (в градусах)
EXTRA_MARGIN_DEG = 10.0


def compute_attention(gaze_eye, R_head, distance, screen_width=0.5):
    """
    gaze_eye: np.array (3,)
    R_head: np.array (3,3)  <-- rotation from solvePnP
    distance: float (meters)
    screen_width: float (meters)

    ВАЖНО:
      - решение о внимании принимается по "расширенному" конусу:
        alpha_eff = max(alpha_theory, MIN_ALPHA_DEG) + EXTRA_MARGIN_DEG
      - в лог выводится "чистый" теоретический alpha (без запаса).
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
    # Защита от нелепых расстояний: если оценка D совсем маленькая,
    # считаем, что пользователь примерно на 0.4 м.
    if distance <= 0.2:
        distance = 0.4

    alpha_theory = np.arctan((screen_width / 2) / distance)
    alpha_deg = max(np.degrees(alpha_theory), MIN_ALPHA_DEG)
    alpha_rad = np.radians(alpha_deg)

    # Реальный конус для решения: теоретический + дополнительный запас
    alpha_eff_rad = alpha_rad + np.radians(EXTRA_MARGIN_DEG)

    attention = 1 if theta <= alpha_eff_rad else 0

    # В лог возвращаем "чистый" alpha без EXTRA_MARGIN_DEG,
    # чтобы видеть физический полуугол конуса.
    return attention, np.degrees(theta), alpha_deg