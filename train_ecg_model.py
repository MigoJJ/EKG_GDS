"""
Training and Evaluation Pipeline for 12-Lead 11-Class ECG 1D-ResNet
Supports multi-label training on PTB-XL (Folds 1-8), Validation on Fold 9, and Export to ONNX.
Can be executed locally (CPU) or on Kaggle GPU.
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pipeline.ptbxl_loader import PTBXLLoader, DIAGNOSTIC_CLASSES
from models.export_onnx import ECGResNet1D


class PTBXLDataset(Dataset):
    """PyTorch Dataset wrapper for PTB-XL 12-lead ECG signals and 11-class multi-labels."""

    def __init__(self, df, data_dir, sampling_rate: int = 100):
        self.df = df.reset_index()
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.label_cols = [f"label_{c}" for c in DIAGNOSTIC_CLASSES]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename_lr"] if self.sampling_rate == 100 else row["filename_hr"]
        full_path = self.data_dir / filename
        
        # Load WFDB
        import wfdb
        signal, _ = wfdb.rdsamp(str(full_path))  # (1000, 12)
        sig = signal.T.astype(np.float32)       # (12, 1000)

        # Standardize per lead
        for ch in range(sig.shape[0]):
            std = np.std(sig[ch])
            if std > 1e-6:
                sig[ch] = (sig[ch] - np.mean(sig[ch])) / std
            else:
                sig[ch] = 0.0

        labels = row[self.label_cols].values.astype(np.float32)
        return torch.tensor(sig, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def train_model(epochs: int = 5, batch_size: int = 32, lr: float = 1e-3, device_str: str = "cpu"):
    device = torch.device(device_str)
    print(f"Training on device: {device} | Epochs: {epochs} | Batch size: {batch_size}")

    loader = PTBXLLoader()
    train_df = loader.get_split("train")
    val_df = loader.get_split("val")

    train_ds = PTBXLDataset(train_df, loader.data_dir, sampling_rate=100)
    val_ds = PTBXLDataset(val_df, loader.data_dir, sampling_rate=100)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ECGResNet1D(in_channels=12, num_classes=11).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    save_path = Path("/home/migowj/ekg-assistant/models/best_ecg_resnet18.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for sigs, labels in train_loader:
            sigs, labels = sigs.to(device), labels.to(device)
            optimizer.zero_grad()
            probs = model(sigs)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(sigs)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sigs, labels in val_loader:
                sigs, labels = sigs.to(device), labels.to(device)
                probs = model(sigs)
                loss = criterion(probs, labels)
                val_loss += loss.item() * len(sigs)

        val_loss /= len(val_ds)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(save_path))
            print(f"  -> Saved new best checkpoint (Val Loss: {val_loss:.4f})")

    print("[DONE] Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device_str=args.device)
