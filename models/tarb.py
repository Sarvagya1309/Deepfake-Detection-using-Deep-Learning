
"""
Multi-Frame Temporal Attention Residual Block (MF-TARB)

Models temporal inconsistencies between consecutive face frames
to enhance deepfake detection features.
"""

import torch
import torch.nn as nn


class MultiFrameTemporalAttentionResidualBlock(nn.Module):
    def __init__(self, feature_dim=2048, reduction=256, num_past=5):
        super().__init__()
        self.num_past = num_past

        self.fc1 = nn.Linear(feature_dim, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduction, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_t, past_features):
        """
        Args:
            f_t (Tensor): current frame feature [B, D]
            past_features (List[Tensor]): previous frame features, length = num_past
        """
        assert len(past_features) == self.num_past, \
            f"Expected {self.num_past} past features, got {len(past_features)}"

        # Temporal difference modeling
        deltas = [torch.abs(f_t - f_p) for f_p in past_features]
        delta_mean = torch.stack(deltas, dim=0).mean(dim=0)

        # Attention estimation
        attn = self.sigmoid(self.fc2(self.relu(self.fc1(delta_mean))))

        # Residual fusion of temporal and current features
        past_mean = torch.stack(past_features, dim=0).mean(dim=0)
        fused = attn * f_t + (1.0 - attn) * past_mean

        return fused


