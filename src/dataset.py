# dataset.py

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class UnityEyesDataset(Dataset):
    """
    Dataset for UnityEyes gaze.h5

    Returns:
        image:  Tensor (1, 35, 55), float32 in [0,1]
        look_vec: Tensor (3,), unit vector (float32)
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path

        # Читаем метаданные один раз
        with h5py.File(self.h5_path, "r") as f:
            self.image_keys = list(f["image"].keys())
            self.look_vecs = f["look_vec"][:, :3].astype(np.float32)

        self.file = None  # ленивое открытие для DataLoader workers

    def _get_file(self):
        # Важно для num_workers > 0
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        return self.file

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        f = self._get_file()

        # --- Image ---
        key = self.image_keys[idx]
        image = np.array(f["image"][key])  # (35, 55), uint8

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # (1, 35, 55)

        # --- Look vector ---
        look_vec = self.look_vecs[idx]

        # Нормализация (на всякий случай)
        norm = np.linalg.norm(look_vec)
        if norm > 0:
            look_vec = look_vec / norm

        return (
            torch.from_numpy(image),
            torch.from_numpy(look_vec),
        )

