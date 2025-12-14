# src/evaluate.py
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from model import GazeCNN
from dataset import GazeDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Config
# ------------------------
BATCH_SIZE = 32
ANGLE_THRESHOLD_DEG = 5.0   # для Accuracy / F1
CHECKPOINT_PATH = "checkpoints/best_model.pt"
DATA_PATH = "data/gaze.h5"

# ------------------------
# Utils
# ------------------------


def angular_error_deg(pred, gt):
    pred = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    gt = gt / np.linalg.norm(gt, axis=1, keepdims=True)

    dot = np.sum(pred * gt, axis=1)
    dot = np.clip(dot, -1.0, 1.0)

    return np.degrees(np.arccos(dot))


# ------------------------
# Evaluation
# ------------------------
def evaluate():
    dataset = GazeDataset(DATA_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = GazeCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_gts = []

    inference_times = []

    with torch.no_grad():
        for images, gaze_gt in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE)
            gaze_gt = gaze_gt.numpy()

            start = time.perf_counter()
            gaze_pred = model(images).cpu().numpy()
            end = time.perf_counter()

            inference_times.append(end - start)

            all_preds.append(gaze_pred)
            all_gts.append(gaze_gt)

    preds = np.vstack(all_preds)
    gts = np.vstack(all_gts)

    # ------------------------
    # MSE
    # ------------------------
    mse = np.mean((preds - gts) ** 2)

    # ------------------------
    # Angular error
    # ------------------------
    ang_err = angular_error_deg(preds, gts)
    mean_ang_err = np.mean(ang_err)

    # ------------------------
    # Accuracy / F1
    # ------------------------
    y_true = np.ones_like(ang_err)
    y_pred = (ang_err <= ANGLE_THRESHOLD_DEG).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # ------------------------
    # Inference time
    # ------------------------
    mean_infer_time = np.mean(inference_times) / BATCH_SIZE

    # ------------------------
    # Report
    # ------------------------
    print("\n===== Evaluation Results =====")
    print(f"MSE: {mse:.6f}")
    print(f"Mean angular error (deg): {mean_ang_err:.2f}")
    print(f"Accuracy (≤ {ANGLE_THRESHOLD_DEG}°): {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Mean inference time per sample: {mean_infer_time*1000:.2f} ms")


if __name__ == "__main__":
    evaluate()
