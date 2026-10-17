"""
1D-ResNet18 Architecture Definition and ONNX Model Exporter for 12-Lead ECG.
Accepts (batch_size, 12, 1000) and produces 11 multi-label sigmoid probabilities.
"""

from pathlib import Path
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

from pipeline.config import ONNX_MODEL_PATH


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ECGResNet1D(nn.Module):
    """Deep 1D Residual Network tailored for 12-lead multi-label ECG diagnosis."""

    def __init__(self, in_channels: int = 12, num_classes: int = 11):
        super().__init__()
        # Initial convolution stage
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 4 Residual stages
        self.stage1 = nn.Sequential(ResBlock1D(64, 64, stride=1), ResBlock1D(64, 64, stride=1))
        self.stage2 = nn.Sequential(ResBlock1D(64, 128, stride=2), ResBlock1D(128, 128, stride=1))
        self.stage3 = nn.Sequential(ResBlock1D(128, 256, stride=2), ResBlock1D(256, 256, stride=1))
        self.stage4 = nn.Sequential(ResBlock1D(256, 512, stride=2), ResBlock1D(512, 512, stride=1))

        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).squeeze(-1)
        logits = self.fc(x)
        probs = self.sigmoid(logits)
        return probs


def export_model_to_onnx(output_path: str = str(ONNX_MODEL_PATH)):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model = ECGResNet1D(in_channels=12, num_classes=11)
    model.eval()

    # Dummy input: batch_size=1, 12 leads, 1000 samples (10s @ 100Hz)
    dummy_input = torch.randn(1, 12, 1000, dtype=torch.float32)

    # Export to ONNX with dynamic batch size using classic TorchScript exporter
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["ecg_signal"],
        output_names=["probabilities"],
        dynamic_axes={
            "ecg_signal": {0: "batch_size"},
            "probabilities": {0: "batch_size"}
        },
        dynamo=False
    )

    # Validate ONNX model
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX model successfully exported and verified at: {out_file}")

    # Benchmark ONNX Runtime CPU inference
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(out_file), sess_opts, providers=["CPUExecutionProvider"])
    
    test_arr = np.random.randn(1, 12, 1000).astype(np.float32)
    import time
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = sess.run(["probabilities"], {"ecg_signal": test_arr})[0]
        times.append(time.perf_counter() - t0)

    avg_latency_ms = (sum(times) / len(times)) * 1000.0
    print(f"[OK] Benchmark latency on i7-6700 CPU: {avg_latency_ms:.2f} ms per 12-lead record!")
    print(f"     Output shape: {res.shape}, values in [0, 1]: {(res >= 0).all() and (res <= 1).all()}")


if __name__ == "__main__":
    export_model_to_onnx()
