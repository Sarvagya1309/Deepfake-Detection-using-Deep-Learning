"""
Feature extraction using ResNet50 + Multi-Frame Temporal Attention Residual Block (MF-TARB).

This script converts face frames into temporal-enhanced feature vectors
for deepfake detection.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

from models.tarb import MultiFrameTemporalAttentionResidualBlock

# ---------------------- DEVICE ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- DATASET PATHS ----------------------
dataset_root = Path("/content/celeb-df-frames/150_frames")
real_path = dataset_root / "real"
fake_path = dataset_root / "fake"

# ---------------------- IMAGE PREPROCESSING ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------- RESNET50 BACKBONE ----------------------
resnet = resnet50(pretrained=True)
resnet.fc = nn.Identity()  # remove classifier head
resnet = resnet.to(device)
resnet.eval()

# Freeze backbone parameters
for param in resnet.parameters():
    param.requires_grad = False

# ---------------------- MF-TARB MODULE ----------------------
NUM_PAST = 5  # number of previous frames for temporal context

tarb = MultiFrameTemporalAttentionResidualBlock(
    feature_dim=2048,
    reduction=256,
    num_past=NUM_PAST
).to(device)
tarb.eval()

# ---------------------- FEATURE EXTRACTION FUNCTION ----------------------
def extract_resnet_feature(image_path: Path) -> torch.Tensor:
    """
    Extracts ResNet50 feature for a single frame.

    Args:
        image_path (Path): path to face frame image
    Returns:
        Tensor: 2048-D feature vector
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = resnet(tensor).squeeze(0)

    return feature

# ---------------------- TEMPORAL FEATURE CONSTRUCTION ----------------------
data = []
frames_per_video = 150

for label, path in [("real", real_path), ("fake", fake_path)]:
    label_int = 0 if label == "real" else 1
    image_files = sorted(path.glob("*.jpg"))

    num_videos = len(image_files) // frames_per_video

    for v in tqdm(range(num_videos), desc=f"Processing {label}"):
        start = v * frames_per_video
        end = start + frames_per_video
        video_frames = image_files[start:end]

        # Initialize temporal buffer
        feature_buffer = [
            extract_resnet_feature(video_frames[i])
            for i in range(NUM_PAST)
        ]

        for i in range(NUM_PAST, len(video_frames)):
            f_curr = extract_resnet_feature(video_frames[i])

            with torch.no_grad():
                f_tarb = tarb(
                    f_curr.unsqueeze(0),
                    [f.unsqueeze(0) for f in feature_buffer]
                ).squeeze(0)

            # Save feature row
            row = [
                f"{label}_vid{v}_frame{i}",
                label_int,
                *f_tarb.cpu().numpy().tolist()
            ]
            data.append(row)

            # Sliding window update
            feature_buffer.pop(0)
            feature_buffer.append(f_curr)

# ---------------------- SAVE FEATURES ----------------------
if not data:
    print("❌ No data collected. Check dataset structure or frame count.")
else:
    feature_cols = [f"feat_{i}" for i in range(len(data[0]) - 2)]
    df = pd.DataFrame(data, columns=["filename", "label"] + feature_cols)

    output_csv = "celeb_df_150_frames_tarb_features.csv"
    df.to_csv(output_csv, index=False)

    print(f"✅ Saved temporal-enhanced features to {output_csv}")


