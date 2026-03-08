import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UnityEyesDataset
from model import GazeCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
CHECKPOINT_PATH = "checkpoints/best_model.pt"
DATA_PATH = "data/gaze.h5"


def angular_error_deg(pred, target):
    pred = F.normalize(pred, dim=1, eps=1e-6)
    target = F.normalize(target, dim=1, eps=1e-6)

    dot = torch.sum(pred * target, dim=1)
    dot = torch.clamp(dot, -0.999999, 0.999999)

    angle = torch.acos(dot)
    return torch.mean(angle).item() * 180 / math.pi


def evaluate():
    dataset = UnityEyesDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GazeCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_errors = []
    inference_times = []

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            start = time.perf_counter()
            preds = model(images)
            end = time.perf_counter()

            err = angular_error_deg(preds, targets)
            all_errors.append(err)
            inference_times.append(end - start)

    mean_err = float(np.mean(all_errors))
    mean_infer_time = float(np.mean(inference_times) / BATCH_SIZE)

    print("\n===== Gaze Regression Evaluation =====")
    print(f"Mean angular error: {mean_err:.2f}°")
    print(f"Mean inference time per sample: {mean_infer_time*1000:.2f} ms")


if __name__ == "__main__":
    evaluate()