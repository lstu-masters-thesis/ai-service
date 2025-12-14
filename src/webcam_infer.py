# src/webcam_infer.py
import cv2
import torch
import numpy as np
import mediapipe as mp

from model import GazeCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load model
# ------------------------
model = GazeCNN().to(DEVICE)
model.load_state_dict(torch.load(
    "checkpoints/best_model.pt", map_location=DEVICE))
model.eval()

# ------------------------
# MediaPipe
# ------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 374, 380]

HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]

ARROW_SCALE = 200

# ------------------------
# Helpers
# ------------------------


def crop_eye(frame, landmarks, indices):
    h, w, _ = frame.shape
    xs = [int(landmarks[i].x * w) for i in indices]
    ys = [int(landmarks[i].y * h) for i in indices]
    x1, x2 = max(min(xs) - 5, 0), min(max(xs) + 5, w)
    y1, y2 = max(min(ys) - 5, 0), min(max(ys) + 5, h)
    eye = frame[y1:y2, x1:x2]
    return eye, (x1 + x2) // 2, (y1 + y2) // 2


def estimate_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]

    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in HEAD_POSE_IDX
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose
        (0.0, -63.6, -12.5),    # Chin
        (-43.3, 32.7, -26.0),   # Left eye
        (43.3, 32.7, -26.0),    # Right eye
        (-28.9, -28.9, -24.1),  # Left mouth
        (28.9, -28.9, -24.1)    # Right mouth
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


# ------------------------
# Webcam
# ------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # ---- Head pose
        R_head = estimate_head_pose(lm, frame.shape)
        if R_head is None:
            continue

        # ---- Both eyes
        gazes = []
        centers = []

        for eye_idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
            eye_img, cx, cy = crop_eye(frame, lm, eye_idx)
            if eye_img.size == 0:
                continue

            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            eye = cv2.resize(gray, (55, 35))
            eye = eye.astype(np.float32) / 255.0
            eye = torch.tensor(eye).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                gaze_eye = model(eye).cpu().numpy()[0]

            gazes.append(gaze_eye)
            centers.append((cx, cy))

        if len(gazes) == 2:
            gaze_eye = np.mean(gazes, axis=0)

            # ---- Head-corrected gaze
            gaze_cam = R_head @ gaze_eye
            gaze_cam = gaze_cam / np.linalg.norm(gaze_cam)

            cx, cy = centers[0]

            end_x = int(cx + gaze_cam[0] * ARROW_SCALE)
            end_y = int(cy - gaze_cam[1] * ARROW_SCALE)

            cv2.arrowedLine(
                frame, (cx, cy), (end_x, end_y),
                (0, 0, 255), 3, tipLength=0.3
            )

            cv2.putText(
                frame,
                f"gaze_cam: {gaze_cam.round(2)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Gaze Estimation (Head Pose Corrected)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
