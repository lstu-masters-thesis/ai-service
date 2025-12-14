# src/dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class GazeDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(self.h5_path, 'r') as f:
            self.image_keys = list(f['image'].keys())
            self.look_vecs = f['look_vec'][:]  # ← читаем один раз

        self.file = None

    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
        return self.file

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        f = self._get_file()

        key = self.image_keys[idx]

        image = np.array(f['image'][key])          # (35, 55)
        look_vec = self.look_vecs[idx][:3]          # ← ВАЖНО

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return (
            torch.tensor(image),
            torch.tensor(look_vec, dtype=torch.float32)
        )
