import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm

from model import AttentionCNN
from dataset import AttentionDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
CHECKPOINT_PATH = "checkpoints/best_model.pt"
DATA_PATH = "data/gaze.h5"


def evaluate():
    dataset = AttentionDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_gts = []
    inference_times = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)

            start = time.perf_counter()
            logits = model(images)
            end = time.perf_counter()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            inference_times.append(end - start)

            all_preds.append(preds)
            all_gts.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)

    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mean_infer_time = np.mean(inference_times) / BATCH_SIZE

    print("\n===== Evaluation Results =====")
    print(f"MSE: {mse:.6f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Mean inference time per sample: {mean_infer_time*1000:.2f} ms")


if __name__ == "__main__":
    evaluate()