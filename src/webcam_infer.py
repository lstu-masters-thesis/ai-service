import cv2
import mediapipe as mp
import numpy as np
import torch

from geometry_attention import compute_attention
from headpose import estimate_head_pose, estimate_distance
from model import GazeCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAZE_CHECKPOINT = "checkpoints/best_model.pt"

# Модель для регрессии вектора взгляда (используется в геометрическом модуле)
gaze_model = GazeCNN().to(DEVICE)
gaze_model.load_state_dict(torch.load(GAZE_CHECKPOINT, map_location=DEVICE))
gaze_model.eval()



mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# --- smoothing ---
prev_gaze = None
EMA_ALPHA = 0.6


def preprocess_eye(frame, landmarks, eye_indices):
    h, w, _ = frame.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min = max(min(x_coords) - 5, 0)
    x_max = min(max(x_coords) + 5, w)
    y_min = max(min(y_coords) - 5, 0)
    y_max = min(max(y_coords) + 5, h)

    if x_max <= x_min or y_max <= y_min:
        return None

    eye_crop = frame[y_min:y_max, x_min:x_max]

    if eye_crop.size == 0:
        return None

    eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    eye_crop = cv2.resize(eye_crop, (55, 35))

    eye_crop = eye_crop.astype(np.float32) / 255.0
    eye_crop = np.expand_dims(eye_crop, axis=(0, 1))

    return torch.from_numpy(eye_crop).to(DEVICE)


def main():
    cap = cv2.VideoCapture(0)

    global prev_gaze

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye_idx = [33, 133, 160, 159, 158, 157, 173]
            right_eye_idx = [263, 362, 387, 386, 385, 384, 398]

            left_eye = preprocess_eye(frame, landmarks, left_eye_idx)
            right_eye = preprocess_eye(frame, landmarks, right_eye_idx)

            if left_eye is None or right_eye is None:
                cv2.imshow("Attention Monitor", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # ------------------------------
            # 1) CNN: направление взгляда
            # ------------------------------
            with torch.no_grad():
                gaze_left = gaze_model(left_eye)
                gaze_right = gaze_model(right_eye)

                gaze_left = torch.nn.functional.normalize(gaze_left, dim=1)
                gaze_right = torch.nn.functional.normalize(gaze_right, dim=1)

                gaze_left = gaze_left.cpu().numpy()[0]
                gaze_right = gaze_right.cpu().numpy()[0]

            # Right eye is mirrored horizontally
            gaze_right[0] *= -1

            # average both eyes
            gaze_eye = (gaze_left + gaze_right) / 2.0

            norm = np.linalg.norm(gaze_eye)
            if norm > 1e-6:
                gaze_eye = gaze_eye / norm
            else:
                cv2.imshow("Attention Monitor", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Alignment UnityEyes → webcam
            gaze_eye[0] *= -1

            # Smoothing (EMA)
            if prev_gaze is None:
                prev_gaze = gaze_eye
            else:
                gaze_eye = EMA_ALPHA * gaze_eye + (1 - EMA_ALPHA) * prev_gaze
                gaze_eye = gaze_eye / np.linalg.norm(gaze_eye)
                prev_gaze = gaze_eye

            # ------------------------------
            # 2) Геометрия: голова + дистанция
            # ------------------------------
            R_head = estimate_head_pose(landmarks, frame.shape)
            distance = estimate_distance(landmarks, frame.shape[1])

            geo_attention = None
            theta = None
            alpha = None

            if R_head is not None:
                geo_attention, theta, alpha = compute_attention(
                    gaze_eye,
                    R_head,
                    distance,
                )


            # ------------------------------
            # 3) Финальное решение (по геометрии)
            # ------------------------------
            attention = geo_attention if geo_attention is not None else 0


            status = "ATTENTIVE" if attention == 1 else "NOT ATTENTIVE"

            cv2.putText(
                frame,
                f"{status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if attention else (0, 0, 255),
                2,
            )

            if theta is not None and alpha is not None:
                cv2.putText(
                    frame,
                    f"angle: {theta:.1f} / limit: {alpha:.1f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )


        cv2.imshow("Attention Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()