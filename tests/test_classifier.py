"""
Unit Tests for 11-Class ECG Deep Learning ONNX Classifier
"""

import time
import pytest
import numpy as np
from pipeline.ptbxl_loader import PTBXLLoader, DIAGNOSTIC_CLASSES
from pipeline.classifier import ECGClassifier, DEFAULT_THRESHOLDS


@pytest.fixture(scope="module")
def classifier():
    return ECGClassifier()


@pytest.fixture(scope="module")
def loader():
    return PTBXLLoader()


def test_classifier_initialization(classifier):
    """Verifies ONNX session initialization and input/output node binding."""
    assert classifier.session is not None
    assert classifier.input_name == "ecg_signal"
    assert classifier.output_name == "probabilities"


def test_preprocess_signal_shape(classifier):
    """Verifies signal transposition and channel normalization."""
    # Input as (1000, 12)
    sig_1000x12 = np.random.randn(1000, 12).astype(np.float32)
    tensor = classifier.preprocess_signal(sig_1000x12)
    assert tensor.shape == (1, 12, 1000)

    # Input as (12, 1000)
    sig_12x1000 = np.random.randn(12, 1000).astype(np.float32)
    tensor2 = classifier.preprocess_signal(sig_12x1000)
    assert tensor2.shape == (1, 12, 1000)


def test_prediction_output_structure(classifier, loader):
    """Verifies that prediction returns valid probabilities and schema."""
    rec = loader.get_record(1, sampling_rate=100)
    res = classifier.predict(rec["signal"], return_top_k=3)

    assert "probabilities" in res
    assert "predictions" in res
    assert "positive_findings" in res
    assert "top_findings" in res
    assert "clinical_urgency" in res

    # 11 classes present in probabilities and predictions
    for cls_name in DIAGNOSTIC_CLASSES:
        assert cls_name in res["probabilities"]
        assert 0.0 <= res["probabilities"][cls_name] <= 1.0
        assert cls_name in res["predictions"]
        assert isinstance(res["predictions"][cls_name], (bool, np.bool_))

    assert len(res["top_findings"]) == 3
    assert res["clinical_urgency"] in ["EMERGENCY", "URGENT", "ROUTINE", "ROUTINE (NORMAL)"]


def test_inference_latency_cpu(classifier):
    """Verifies that CPU inference latency is below 50 ms (real-time requirement)."""
    dummy_ecg = np.random.randn(1000, 12).astype(np.float32)
    
    # Warm-up run
    _ = classifier.predict(dummy_ecg)

    # Measure average latency over 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = classifier.predict(dummy_ecg)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    assert avg_ms < 50.0, f"Average inference time {avg_ms:.2f} ms exceeds 50 ms threshold"
