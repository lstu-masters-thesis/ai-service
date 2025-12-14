# src/train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm

from dataset import GazeDataset
from model import GazeCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    dataset = GazeDataset("data/gaze.h5")
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    model = GazeCNN().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()

    for epoch in range(20):
        model.train()
        total_loss = 0

        for images, gaze in tqdm(loader):
            images = images.to(DEVICE)
            gaze = gaze.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, gaze)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.6f}")

        torch.save(model.state_dict(), "checkpoints/best_model.pt")


if __name__ == "__main__":
    train()
