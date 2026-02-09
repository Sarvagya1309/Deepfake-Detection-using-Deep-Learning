
"""
ResNet50 backbone used as a frozen feature extractor
for deepfake detection.
"""

import torch
from torchvision.models import resnet50


def get_resnet50(device):
    """
    Returns a pretrained ResNet50 with the classification
    head removed (2048-D feature output).
    """
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model.to(device)


