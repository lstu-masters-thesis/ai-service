# src/webcam_infer.py
import cv2
import torch
import numpy as np
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
# Webcam
# ------------------------
cap = cv2.VideoCapture(0)

ARROW_SCALE = 150  # длина стрелки (можно менять)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ------------------------
    # Preprocess input
    # ------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(gray, (55, 35))
    eye = eye.astype(np.float32) / 255.0
    eye = torch.tensor(eye).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # ------------------------
    # Inference
    # ------------------------
    with torch.no_grad():
        gaze = model(eye).cpu().numpy()[0]

    gx, gy, gz = gaze

    # ------------------------
    # Draw gaze arrow
    # ------------------------
    center_x = w // 2
    center_y = h // 2

    end_x = int(center_x + gx * ARROW_SCALE)
    end_y = int(center_y - gy * ARROW_SCALE)  # инверсия Y для OpenCV

    cv2.arrowedLine(
        frame,
        (center_x, center_y),
        (end_x, end_y),
        color=(0, 0, 255),
        thickness=4,
        tipLength=0.25
    )

    # ------------------------
    # Debug text
    # ------------------------
    cv2.putText(
        frame,
        f"Gaze: x={gx:.2f}, y={gy:.2f}, z={gz:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gaze Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
