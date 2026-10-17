"""
High-Speed In-Memory Training Pipeline for 12-Lead ECG 1D-ResNet
Preloads all 17,418 training records into RAM (~800MB) to eliminate disk I/O,
trains on CPU using 8 threads, and exports the converged weights to ONNX.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import onnx

from pipeline.ptbxl_loader import PTBXLLoader, DIAGNOSTIC_CLASSES
from models.export_onnx import ECGResNet1D


def preload_dataset(df, loader, desc: str = "train"):
    """Preloads raw WFDB files into a single in-memory PyTorch float tensor."""
    print(f"[{desc}] Preloading {len(df)} records into RAM...")
    t0 = time.time()
    
    signals = []
    label_cols = [f"label_{c}" for c in DIAGNOSTIC_CLASSES]
    labels = df[label_cols].values.astype(np.float32)

    for idx, filename in enumerate(df["filename_lr"]):
        sig, _ = loader.load_signal(filename, sampling_rate=100)  # (1000, 12)
        sig = sig.T.astype(np.float32)                           # (12, 1000)
        
        # Standardize per lead
        for ch in range(sig.shape[0]):
            std = np.std(sig[ch])
            if std > 1e-6:
                sig[ch] = (sig[ch] - np.mean(sig[ch])) / std
            else:
                sig[ch] = 0.0
                
        signals.append(sig)
        if (idx + 1) % 5000 == 0:
            print(f"  -> Processed {idx + 1}/{len(df)} records ({time.time() - t0:.1f}s)")

    X_tensor = torch.tensor(np.array(signals), dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)
    ram_mb = (X_tensor.element_size() * X_tensor.nelement()) / (1024 * 1024)
    print(f"[{desc}] Preload complete in {time.time() - t0:.1f}s | Shape: {X_tensor.shape} | RAM: {ram_mb:.1f} MB\n")
    return X_tensor, y_tensor


def run_in_memory_training(epochs: int = 12, batch_size: int = 64, lr: float = 1.5e-3):
    # Set PyTorch thread count to utilize i7-6700 8 threads
    torch.set_num_threads(8)
    print("=" * 75)
    print(f"  12-LEAD ECG 1D-RESNET HIGH-SPEED IN-MEMORY TRAINING")
    print(f"  CPU Threads: {torch.get_num_threads()} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 75)

    loader = PTBXLLoader()
    train_df = loader.get_split("train")
    val_df = loader.get_split("val")

    # 1. Preload datasets into RAM
    t_start = time.time()
    X_train, y_train = preload_dataset(train_df, loader, "Train (Folds 1-8)")
    X_val, y_val = preload_dataset(val_df, loader, "Validation (Fold 9)")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = ECGResNet1D(in_channels=12, num_classes=11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Numerically stable multi-label BCE loss with epsilon clamping
    def stable_bce_loss(probs, targets):
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        loss = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
        return loss.mean()

    best_val_loss = float("inf")
    best_weights_path = Path("/home/migowj/ekg-assistant/models/best_ecg_resnet18.pt")
    onnx_output_path = Path("/home/migowj/ekg-assistant/models/ecg_resnet18_11class.onnx")

    # 3. Training Loop
    print("Starting training epochs...")
    print("-" * 75)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for sigs, targets in train_loader:
            optimizer.zero_grad()
            probs = model(sigs)
            loss = stable_bce_loss(probs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(sigs)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sigs, targets in val_loader:
                probs = model(sigs)
                loss = stable_bce_loss(probs, targets)
                val_loss += loss.item() * len(sigs)

        val_loss /= len(val_ds)
        cur_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {cur_lr:.5f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(best_weights_path))
            print(f"  ★ New best model checkpoint saved (Val Loss: {val_loss:.4f})")

    total_training_time = time.time() - t_start
    print("-" * 75)
    print(f"[Training Complete] Total elapsed: {total_training_time:.1f}s ({total_training_time/60.0:.1f} min)")

    # 4. Export Best Model to ONNX
    print(f"\nExporting best model to ONNX: {onnx_output_path}")
    model.load_state_dict(torch.load(str(best_weights_path)))
    model.eval()

    dummy_input = torch.randn(1, 12, 1000, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["ecg_signal"],
        output_names=["probabilities"],
        dynamic_axes={"ecg_signal": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False
    )

    onnx_model = onnx.load(str(onnx_output_path))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model successfully verified and updated on disk!")

    # 5. Immediate Clinical Verification on Key Records
    print("\n" + "=" * 75)
    print("  IMMEDIATE CLINICAL VERIFICATION ON TEST RECORDS")
    print("=" * 75)
    from pipeline.classifier import ECGClassifier
    classifier = ECGClassifier(model_path=onnx_output_path)

    for test_id, desc in [(9, "Record 9 (Normal Sinus Rhythm)"), (63, "Record 63 (Anteroseptal MI)")]:
        rec = loader.get_record(test_id, sampling_rate=100)
        res = classifier.predict(rec["signal"], return_top_k=3)
        print(f"\n{desc}:")
        print(f"  • Clinical Urgency: {res['clinical_urgency']}")
        print(f"  • Top Findings   : {res['top_findings']}")
        print(f"  • Positive Flags : {res['positive_findings']}")
        print(f"  • Probabilities  : NORM={res['probabilities']['NORM']*100:.1f}%, MI={res['probabilities']['MI']*100:.1f}%, LAFB={res['probabilities']['LAFB']*100:.1f}%")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    run_in_memory_training(epochs=12, batch_size=64, lr=1.5e-3)
