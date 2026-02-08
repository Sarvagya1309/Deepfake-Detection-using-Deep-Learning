# models/resnet50_backbone.py
import torch
from torchvision.models import resnet50

def get_resnet50(device):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)

