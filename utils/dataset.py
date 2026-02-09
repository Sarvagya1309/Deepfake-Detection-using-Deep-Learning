"""
Dataset utilities for deepfake detection.

Handles:
- Frame-level feature datasets
- Video-level grouping (to avoid data leakage)
- Scaling consistency between train/val/test
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class FeatureDataset(Dataset):
    """
    PyTorch Dataset for frame-level deepfake features.

    Each sample:
    - X: scaled feature vector
    - y: binary label (0 = real, 1 = fake)
    """

    def __init__(self, dataframe, scaler):
        """
        Args:
            dataframe (pd.DataFrame): must include 'label' column
            scaler (sklearn Scaler): fitted on training data only
        """
        drop_cols = [c for c in ['label', 'filename', 'video_id'] if c in dataframe.columns]
        features = dataframe.drop(columns=drop_cols).values.astype('float32')

        self.X = scaler.transform(features).astype('float32')
        self.y = dataframe['label'].values.astype('int64')
        self.length = len(self.y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def extract_video_id(filename):
    """
    Extracts video identifier from frame filename.

    Example:
        celeb_real_001_frame023.jpg → celeb_real_001
    """
    base = str(filename).split('/')[-1]
    base = base.replace('.jpg', '').replace('.png', '')

    if "_frame" in base:
        return base.rsplit("_frame", 1)[0]

    return base
