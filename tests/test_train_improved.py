"""
Phase 3 Training Pipeline Tests
Guards against the double-sigmoid defect and verifies cut-off tuning and ONNX compatibility.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.export_onnx import ECGResNet1D
from pipeline.classifier import ECGClassifier
from pipeline.ptbxl_loader import DIAGNOSTIC_CLASSES
from train_improved import (
    FocalLoss, augment_batch, evaluate, export_onnx, logits_model,
    probability_model, train, tune_cutoffs, TARGET_SENSITIVITY,
)


def test_focal_loss_with_gamma_zero_equals_bce():
    logits = torch.randn(8, 11)
    targets = torch.randint(0, 2, (8, 11)).float()
    expected = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, targets), expected, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    targets = torch.ones(1, 11)
    easy, hard = torch.full((1, 11), 4.0), torch.full((1, 11), -1.0)
    fl, bce = FocalLoss(gamma=2.0), nn.functional.binary_cross_entropy_with_logits
    assert fl(easy, targets) / bce(easy, targets) < fl(hard, targets) / bce(hard, targets)


def test_training_model_outputs_logits_and_deployed_model_outputs_probabilities():
    """The training model must skip the Sigmoid; the deployed copy must restore it."""
    torch.manual_seed(0)
    train_model = logits_model(ECGResNet1D(12, 11)).eval()
    deployed = probability_model(train_model)
    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        logits, probs = train_model(x), deployed(x)
    assert isinstance(train_model.sigmoid, nn.Identity)
    assert torch.allclose(torch.sigmoid(logits), probs, atol=1e-6)


def test_augment_batch_preserves_shape_and_is_finite():
    x = torch.randn(16, 12, 1000)
    out = augment_batch(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()
    assert not torch.equal(out, x)


def test_tune_cutoffs_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (500, 11))
    probs = np.clip(0.3 * y + rng.random((500, 11)) * 0.7, 0, 1)
    cutoffs = tune_cutoffs(y, probs)
    for i, cls in enumerate(DIAGNOSTIC_CLASSES):
        sens = np.mean(probs[y[:, i] == 1, i] >= cutoffs[cls])
        assert sens >= TARGET_SENSITIVITY[cls]
    metrics = evaluate(y, probs, cutoffs)
    assert set(metrics["per_class"]) == set(DIAGNOSTIC_CLASSES)
    assert 0 <= metrics["macro"]["auc"] <= 1


def test_train_smoke_and_onnx_loads_in_classifier(tmp_path):
    """One tiny epoch end-to-end: train -> ONNX -> production ECGClassifier inference."""
    torch.manual_seed(0)
    X = torch.randn(48, 12, 1000)
    y = torch.randint(0, 2, (48, 11)).float()
    model, history = train(X[:32], y[:32], X[32:], y[32:], epochs=1, batch_size=16,
                           device=torch.device("cpu"))
    assert len(history) == 1 and isinstance(model.sigmoid, nn.Sigmoid)

    onnx_path = tmp_path / "candidate.onnx"
    export_onnx(model, onnx_path)
    clf = ECGClassifier(model_path=onnx_path)
    result = clf.predict(np.random.randn(1000, 12).astype(np.float32))
    assert set(result["probabilities"]) == set(DIAGNOSTIC_CLASSES)
    assert all(0.0 <= p <= 1.0 for p in result["probabilities"].values())
