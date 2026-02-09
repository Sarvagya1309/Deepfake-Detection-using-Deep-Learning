"""
Training and evaluation of an MLP classifier with Elastic Weight Consolidation (EWC).

This script trains a frame-level deepfake detector on MF-TARB features using
video-level splits to prevent data leakage.
"""

import random
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataset import FeatureDataset, extract_video_id
from models.mlp import MLPModel
from models.ewc import EWC

# ===================== Reproducibility =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== Device =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== Mixup Utilities =====================
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===================== Training & Evaluation =====================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler=None,
    device="cpu",
    ewc=None,
    ewc_lambda=0.4,
    epochs=10,
    mixup_alpha=0.2,
    gaussian_noise_std=0.01,
    grad_clip=1.0,
    early_stopping_patience=5,
):
    model.to(device)
    best_val_acc = 0.0
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            if gaussian_noise_std > 0:
                X = X + torch.randn_like(X) * gaussian_noise_std

            if mixup_alpha > 0:
                Xm, ya, yb, lam = mixup_data(X, y, alpha=mixup_alpha)
                outputs = model(Xm)
                loss = mixup_criterion(criterion, outputs, ya, yb, lam)
            else:
                outputs = model(X)
                loss = criterion(outputs, y)

            if ewc is not None:
                loss = loss + ewc_lambda * ewc.penalty(model)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if scheduler is not None:
            try:
                scheduler.step(val_acc)
            except TypeError:
                scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if patience >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
    return correct / total


def predict_probs(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            outputs = model(X)
            probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    return np.array(probs)

# ===================== Evaluation Helpers =====================
def tune_frame_threshold(probs, labels):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def frame_level_report(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    print(classification_report(labels, preds, target_names=["Real", "Fake"]))
    print("Accuracy:", accuracy_score(labels, preds))
    print("AUC:", roc_auc_score(labels, probs))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ===================== Pipeline =====================
df = pd.read_csv(
    "/content/celeb-df-150-frames-tarb-features/celeb_df_150_frames_tarb_features.csv"
).dropna(subset=["label", "filename"])

df["video_id"] = df["filename"].apply(extract_video_id)

videos = df.groupby("video_id")["label"].first().reset_index()

train_vids, temp_vids = train_test_split(
    videos, test_size=0.3, stratify=videos["label"], random_state=SEED
)
val_vids, test_vids = train_test_split(
    temp_vids, test_size=0.5, stratify=temp_vids["label"], random_state=SEED
)

train_df = df[df["video_id"].isin(train_vids["video_id"])]
val_df = df[df["video_id"].isin(val_vids["video_id"])]
test_df = df[df["video_id"].isin(test_vids["video_id"])]

feature_cols = [c for c in train_df.columns if c not in ("label", "filename", "video_id")]

scaler = StandardScaler().fit(train_df[feature_cols].values.astype("float32"))

train_dataset = FeatureDataset(train_df, scaler)
val_dataset = FeatureDataset(val_df, scaler)
test_dataset = FeatureDataset(test_df, scaler)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model = MLPModel(input_dim=len(feature_cols)).to(device)
criterion = nn.CrossEntropyLoss()

# -------- Phase 1: Standard Training --------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
model = train_model(model, train_loader, val_loader, optimizer, criterion, device=device)

# -------- Phase 2: Continual Learning with EWC --------
ewc = EWC(model, train_dataset, device=device, fisher_sample_size=500)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

model = train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device=device,
    ewc=ewc,
    ewc_lambda=0.4,
    epochs=10,
)

# -------- Final Evaluation --------
val_probs = predict_probs(model, val_loader, device)
test_probs = predict_probs(model, test_loader, device)

best_t, _ = tune_frame_threshold(val_probs, val_df["label"].values)
print(f"Best validation threshold: {best_t:.3f}")

frame_level_report(test_df["label"].values, test_probs, threshold=best_t)


