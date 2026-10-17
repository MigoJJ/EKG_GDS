"""
Phase 3: Improved 12-Lead ECG 1D-ResNet Training (CPU or GPU, local or Kaggle)

- Focal loss on logits (the model's final Sigmoid is bypassed during training only)
- Vectorized on-device batch augmentation: amplitude scaling, baseline drift, noise, lead dropout
- Checkpoint selection by validation macro AUPRC with early stopping
- Clinical cut-offs re-tuned on Fold 9 for target sensitivities, then evaluated once on Fold 10
- All artifacts go to a separate candidate directory; the production model is never overwritten

Usage:
    PYTHONPATH=. .venv/bin/python -u train_improved.py --epochs 50
    EKG_PTBXL_DIR=/kaggle/input/ptb-xl python -u train_improved.py --output-dir /kaggle/working/out
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

from pipeline.config import MODELS_DIR
from pipeline.ptbxl_loader import PTBXLLoader, DIAGNOSTIC_CLASSES
from models.export_onnx import ECGResNet1D


# Target sensitivities used for cut-off tuning (same policy as models/clinical_cutoffs.json)
TARGET_SENSITIVITY = {cls: 0.80 for cls in DIAGNOSTIC_CLASSES}
TARGET_SENSITIVITY.update({"MI": 0.90, "AFIB": 0.90})


def log(msg: str = "") -> None:
    print(msg, flush=True)


class FocalLoss(nn.Module):
    """Multi-label focal loss computed from logits: FL = alpha_t * (1 - p_t)^gamma * BCE."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        loss = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            loss = torch.where(targets == 1, self.alpha * loss, (1 - self.alpha) * loss)
        return loss.mean()


def augment_batch(x: torch.Tensor, fs: int = 100) -> torch.Tensor:
    """Vectorized augmentation of a standardized batch (B, 12, T) on its current device."""
    b, leads, t_len = x.shape
    dev = x.device

    # 1. Amplitude scaling ±10% (50% of samples)
    scale = torch.where(torch.rand(b, 1, 1, device=dev) < 0.5,
                        torch.empty(b, 1, 1, device=dev).uniform_(0.9, 1.1),
                        torch.ones(b, 1, 1, device=dev))
    x = x * scale

    # 2. Baseline drift: slow sinusoid 0.05-0.5 Hz (50% of samples)
    t = torch.arange(t_len, device=dev, dtype=x.dtype) / fs
    freq = torch.empty(b, 1, 1, device=dev).uniform_(0.05, 0.5)
    phase = torch.empty(b, 1, 1, device=dev).uniform_(0, 2 * np.pi)
    amp = torch.where(torch.rand(b, 1, 1, device=dev) < 0.5,
                      torch.empty(b, 1, 1, device=dev).uniform_(0.05, 0.3),
                      torch.zeros(b, 1, 1, device=dev))
    x = x + amp * torch.sin(2 * np.pi * freq * t.view(1, 1, -1) + phase)

    # 3. Gaussian noise, sigma 0.01-0.05 (70% of samples)
    sigma = torch.where(torch.rand(b, 1, 1, device=dev) < 0.7,
                        torch.empty(b, 1, 1, device=dev).uniform_(0.01, 0.05),
                        torch.zeros(b, 1, 1, device=dev))
    x = x + sigma * torch.randn_like(x)

    # 4. Lead dropout: each lead zeroed with p=0.05 (simulates a disconnected electrode)
    keep = (torch.rand(b, leads, 1, device=dev) >= 0.05).to(x.dtype)
    return x * keep


def standardize(sig: np.ndarray) -> np.ndarray:
    """Per-lead z-score of a (12, T) signal; identical to ECGClassifier.preprocess_signal."""
    sig = sig.astype(np.float32)
    for ch in range(sig.shape[0]):
        std = np.std(sig[ch])
        sig[ch] = (sig[ch] - np.mean(sig[ch])) / std if std > 1e-6 else 0.0
    return sig


def preload_split(loader: PTBXLLoader, split: str):
    df = loader.get_split(split)
    log(f"[{split}] Loading {len(df)} records...")
    t0 = time.time()
    signals = []
    for i, filename in enumerate(df["filename_lr"]):
        sig, _ = loader.load_signal(filename, sampling_rate=100)
        signals.append(standardize(sig.T))
        if (i + 1) % 2000 == 0:
            log(f"  {i + 1}/{len(df)} ({100 * (i + 1) / len(df):.0f}%) {time.time() - t0:.0f}s")
    label_cols = [f"label_{c}" for c in DIAGNOSTIC_CLASSES]
    X = torch.tensor(np.stack(signals), dtype=torch.float32)
    y = torch.tensor(df[label_cols].values.astype(np.float32))
    log(f"[{split}] Done in {time.time() - t0:.0f}s, shape {tuple(X.shape)}")
    return X, y


def logits_model(model: ECGResNet1D) -> ECGResNet1D:
    """Bypasses the final Sigmoid so training uses numerically stable logits."""
    model.sigmoid = nn.Identity()
    return model


def probability_model(model: ECGResNet1D) -> ECGResNet1D:
    """Copy of the model with the final Sigmoid restored, as deployed in ONNX."""
    deployed = copy.deepcopy(model).cpu()
    deployed.sigmoid = nn.Sigmoid()
    return deployed.eval()


@torch.no_grad()
def deployed_probs(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Probabilities from a deployed (Sigmoid-included) model on CPU."""
    model.eval()
    return np.concatenate([model(xb).numpy() for (xb,) in DataLoader(TensorDataset(X), batch_size=batch_size)])


def macro_auprc(y_true: np.ndarray, probs: np.ndarray) -> float:
    scores = [average_precision_score(y_true[:, i], probs[:, i])
              for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0]
    return float(np.mean(scores))


def tune_cutoffs(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Highest threshold per class that still reaches its target sensitivity."""
    cutoffs = {}
    for i, cls in enumerate(DIAGNOSTIC_CLASSES):
        pos = np.sort(probs[y_true[:, i] == 1, i])[::-1]
        if len(pos) == 0:
            cutoffs[cls] = 0.5
            continue
        k = int(np.ceil(TARGET_SENSITIVITY[cls] * len(pos))) - 1
        # Floor (not round) to 4 decimals so the stored cut-off never exceeds the boundary probability
        cutoffs[cls] = float(np.floor(pos[min(k, len(pos) - 1)] * 1e4) / 1e4)
    return cutoffs


def evaluate(y_true: np.ndarray, probs: np.ndarray, cutoffs: Dict[str, float]) -> Dict:
    per_class = {}
    for i, cls in enumerate(DIAGNOSTIC_CLASSES):
        yt, p = y_true[:, i], probs[:, i]
        pred = p >= cutoffs[cls]
        tp, fp = int(np.sum(pred & (yt == 1))), int(np.sum(pred & (yt == 0)))
        fn, tn = int(np.sum(~pred & (yt == 1))), int(np.sum(~pred & (yt == 0)))
        per_class[cls] = {
            "n_pos": int(yt.sum()),
            "auc": round(float(roc_auc_score(yt, p)), 4) if 0 < yt.sum() < len(yt) else None,
            "auprc": round(float(average_precision_score(yt, p)), 4) if yt.sum() > 0 else None,
            "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
            "ppv": round(tp / (tp + fp), 4) if tp + fp else None,
            "npv": round(tn / (tn + fn), 4) if tn + fn else None,
        }

    def macro(key):
        vals = [m[key] for m in per_class.values() if m[key] is not None]
        return round(float(np.mean(vals)), 4)

    return {
        "macro": {k: macro(k) for k in ["auc", "auprc", "sensitivity", "specificity", "ppv", "npv"]},
        "per_class": per_class,
    }


def train(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    epochs: int = 50, batch_size: int = 64, lr: float = 1.5e-3,
    patience: int = 5, gamma: float = 2.0, augment: bool = True,
    device: Optional[torch.device] = None, checkpoint_path: Optional[Path] = None,
):
    """Trains ECGResNet1D and returns (best model with Sigmoid restored, history)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = logits_model(ECGResNet1D(in_channels=12, num_classes=len(DIAGNOSTIC_CLASSES))).to(device)
    criterion = FocalLoss(gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                              pin_memory=device.type == "cuda")
    y_val_np = y_val.numpy()

    best_auprc, best_state, stale, history = -1.0, None, 0, []
    t_start = time.time()
    log(f"Training on {device} | epochs={epochs} batch={batch_size} lr={lr} focal_gamma={gamma} patience={patience}")
    log(f"{'Epoch':>9} {'Progress':>8} {'TrainLoss':>10} {'ValAUPRC':>9} {'LR':>9} {'Epoch(s)':>8} {'ETA(min)':>8}")

    n_batches = len(train_loader)
    last_pct = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if augment:
                xb = augment_batch(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)

            # Overall progress line at every 1% across all epochs
            pct = int(100 * ((epoch - 1) * n_batches + batch_idx) / (epochs * n_batches))
            if pct > last_pct:
                last_pct = pct
                elapsed = time.time() - t_start
                eta = elapsed / pct * (100 - pct) / 60
                log(f"  [{pct:3d}%] epoch {epoch} batch {batch_idx}/{n_batches} "
                    f"loss {loss.item():.4f} | elapsed {elapsed / 60:.1f} min | ETA {eta:.1f} min")
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Validation uses deployed model (with Sigmoid)
        with torch.no_grad():
            model_deployed = probability_model(model)
            val_auprc = macro_auprc(y_val_np, deployed_probs(model_deployed, X_val))
        epoch_sec = time.time() - t0
        eta_min = (time.time() - t_start) / epoch * (epochs - epoch) / 60
        history.append({"epoch": epoch, "train_loss": total / len(X_train), "val_macro_auprc": val_auprc,
                        "lr": cur_lr, "seconds": round(epoch_sec, 1)})

        marker = ""
        if val_auprc > best_auprc:
            best_auprc, best_state, stale = val_auprc, copy.deepcopy(model.state_dict()), 0
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
            marker = " ★ best"
        else:
            stale += 1
        log(f"{epoch:>4}/{epochs:<4} {100 * epoch / epochs:>7.0f}% {total / len(X_train):>10.4f} "
            f"{val_auprc:>9.4f} {cur_lr:>9.6f} {epoch_sec:>8.1f} {eta_min:>8.1f}{marker}")

        if stale >= patience:
            log(f"Early stopping: no AUPRC improvement for {patience} epochs")
            break

    log(f"Best val macro AUPRC {best_auprc:.4f} | total {(time.time() - t_start) / 60:.1f} min")
    model.load_state_dict(best_state)
    return probability_model(model), history


def export_onnx(model: nn.Module, path: Path) -> None:
    torch.onnx.export(
        model, torch.randn(1, 12, 1000, dtype=torch.float32), str(path),
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["ecg_signal"], output_names=["probabilities"],
        dynamic_axes={"ecg_signal": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (ignored on GPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: models/candidates/<timestamp>")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (MODELS_DIR / "candidates" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("=" * 80)
    log(f"  PHASE 3 TRAINING  run={run_id}  device={device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else f" ({args.threads} threads)"))
    log(f"  Output: {out_dir}")
    log("=" * 80)

    loader = PTBXLLoader()
    X_train, y_train = preload_split(loader, "train")
    X_val, y_val = preload_split(loader, "val")
    X_test, y_test = preload_split(loader, "test")

    model, history = train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        gamma=args.gamma, augment=not args.no_augment, device=device,
        checkpoint_path=out_dir / "best_ecg_resnet18.pt",
    )

    # Cut-offs on Fold 9, single final evaluation on Fold 10
    val_probs = deployed_probs(model, X_val)
    test_probs = deployed_probs(model, X_test)
    cutoffs = tune_cutoffs(y_val.numpy(), val_probs)
    metrics = evaluate(y_test.numpy(), test_probs, cutoffs)

    torch.save(model.state_dict(), out_dir / "best_ecg_resnet18.pt")
    onnx_path = out_dir / "ecg_resnet18_11class.onnx"
    export_onnx(model, onnx_path)

    cutoffs_doc = {
        "metadata": {
            "version": f"phase3-{run_id}",
            "updated_at": datetime.now().strftime("%Y-%m-%d"),
            "tuning_dataset": f"PTB-XL Fold 9 (Validation, {len(y_val)} records)",
            "evaluation_dataset": f"PTB-XL Fold 10 (Test, {len(y_test)} records)",
            "strategy": "Highest threshold reaching target sensitivity (MI/AFIB 90%, others 80%)",
        },
        "cutoffs": cutoffs,
        "performance_fold10": metrics["macro"],
    }
    (out_dir / "clinical_cutoffs.json").write_text(json.dumps(cutoffs_doc, indent=2, ensure_ascii=False))
    (out_dir / "metrics_fold10.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "run_info.json").write_text(json.dumps({
        "run_id": run_id, "device": str(device), "args": {k: str(v) for k, v in vars(args).items()},
        "torch": torch.__version__, "epochs_run": len(history),
        "best_val_macro_auprc": max(h["val_macro_auprc"] for h in history),
        "sha256": {p.name: sha256(p) for p in [onnx_path, out_dir / "best_ecg_resnet18.pt"]},
    }, indent=2))

    log("")
    log("Fold 10 (test) macro: " + ", ".join(f"{k}={v}" for k, v in metrics["macro"].items()))
    log(f"{'Class':<6} {'AUC':>6} {'AUPRC':>6} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'Cutoff':>7}")
    for cls, m in metrics["per_class"].items():
        log(f"{cls:<6} {m['auc'] or 0:>6.3f} {m['auprc'] or 0:>6.3f} {m['sensitivity'] or 0:>6.3f} "
            f"{m['specificity'] or 0:>6.3f} {m['ppv'] or 0:>6.3f} {cutoffs[cls]:>7.4f}")
    log(f"\nArtifacts saved to {out_dir} (production model unchanged)")


if __name__ == "__main__":
    main()
