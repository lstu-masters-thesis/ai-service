# train.py

import os
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from dataset import UnityEyesDataset
from model import GazeCNN


# ------------------------
# Config
# ------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "data/gaze.h5"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-4                 # ↓ было 1e-3 → это ломало обучение
VAL_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 7


# ------------------------
# Stable Angular Loss
# ------------------------

def angular_loss(pred, target):
    """
    Numerically stable angular loss (radians)
    """

    # Safe normalization
    pred = F.normalize(pred, dim=1, eps=1e-6)
    target = F.normalize(target, dim=1, eps=1e-6)

    dot = torch.sum(pred * target, dim=1)

    # КРИТИЧНО: не [-1,1], а чуть уже
    dot = torch.clamp(dot, -0.999999, 0.999999)

    angle = torch.acos(dot)

    return torch.mean(angle)


def angular_error_deg(pred, target):
    pred = F.normalize(pred, dim=1, eps=1e-6)
    target = F.normalize(target, dim=1, eps=1e-6)

    dot = torch.sum(pred * target, dim=1)
    dot = torch.clamp(dot, -0.999999, 0.999999)

    angle = torch.acos(dot)
    return torch.mean(angle).item() * 180 / math.pi


# ------------------------
# Train
# ------------------------

def train():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading dataset...")
    full_dataset = UnityEyesDataset(DATA_PATH)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = GazeCNN().to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...\n")

    for epoch in range(EPOCHS):

        # ------------------------
        # TRAIN
        # ------------------------

        model.train()
        train_loss = 0
        train_angle = 0
        valid_batches = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(images)
            loss = angular_loss(preds, targets)

            # 🔥 защита от NaN
            if torch.isnan(loss):
                print("⚠ NaN batch skipped")
                continue

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            train_loss += loss.item()
            train_angle += angular_error_deg(preds.detach(), targets)
            valid_batches += 1

        if valid_batches == 0:
            print("Training collapsed (all batches NaN)")
            break

        train_loss /= valid_batches
        train_angle /= valid_batches

        # ------------------------
        # VALIDATION
        # ------------------------

        model.eval()
        val_loss = 0
        val_angle = 0
        val_batches = 0

        with torch.no_grad():
            for images, targets in val_loader:

                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                preds = model(images)
                loss = angular_loss(preds, targets)

                if torch.isnan(loss):
                    continue

                val_loss += loss.item()
                val_angle += angular_error_deg(preds, targets)
                val_batches += 1

        val_loss /= val_batches
        val_angle /= val_batches

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch+1}: "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Train Angle={train_angle:.2f}° | "
            f"Val Angle={val_angle:.2f}° | "
            f"LR={current_lr:.6f}"
        )

        # ------------------------
        # Early Stopping
        # ------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("✓ Best model saved\n")
        else:
            patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()