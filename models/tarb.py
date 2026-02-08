
import torch
import torch.nn as nn

class MultiFrameTemporalAttentionResidualBlock(nn.Module):
    """
    Temporal Attention over multiple previous frames.
    Computes attention based on difference between current and each past frame,
    then fuses them adaptively.
    """
    def __init__(self, feature_dim=2048, reduction=256, num_past=5):
        super().__init__()
        self.num_past = num_past
        self.fc1 = nn.Linear(feature_dim, reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(reduction, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_t, past_features):
        """
        f_t: current feature [B, 2048]
        past_features: list of previous frame features [B, 2048]
        """
        # Aggregate temporal differences
        deltas = [torch.abs(f_t - f_p) for f_p in past_features]
        delta_mean = torch.stack(deltas, dim=0).mean(dim=0)  # average temporal delta

        # Compute attention weights
        attn = self.sigmoid(self.fc2(self.relu(self.fc1(delta_mean))))
        fused = attn * f_t + (1 - attn) * torch.stack(past_features, dim=0).mean(dim=0)
        return fused

