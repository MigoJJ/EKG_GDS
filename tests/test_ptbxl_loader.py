"""
Unit Tests for PTB-XL Dataset Loader & 11-Class Diagnostic Mapper
"""

import pytest
import numpy as np
from pipeline.ptbxl_loader import PTBXLLoader, DIAGNOSTIC_CLASSES


@pytest.fixture(scope="module")
def loader():
    """Initializes the loader once for the test module."""
    return PTBXLLoader()


def test_metadata_loading(loader):
    """Verifies PTB-XL metadata structure and row count."""
    df = loader.get_split("all")
    assert len(df) == 21799, f"Expected 21,799 records, found {len(df)}"
    assert "strat_fold" in df.columns
    assert "scp_codes" in df.columns
    
    for cls_name in DIAGNOSTIC_CLASSES:
        col = f"label_{cls_name}"
        assert col in df.columns, f"Missing label column: {col}"
        assert df[col].isin([0, 1]).all(), f"Column {col} has non-binary values"


def test_fold_splits(loader):
    """Verifies train (folds 1-8), val (fold 9), test (fold 10) split distributions."""
    train_df = loader.get_split("train")
    val_df = loader.get_split("val")
    test_df = loader.get_split("test")

    assert len(train_df) == 17418, f"Train split count unexpected: {len(train_df)}"
    assert len(val_df) == 2183, f"Val split count unexpected: {len(val_df)}"
    assert len(test_df) == 2198, f"Test split count unexpected: {len(test_df)}"
    assert len(train_df) + len(val_df) + len(test_df) == 21799


def test_clinical_rule_no_norm_and_mi(loader):
    """Verifies clinical mutual exclusion: a record cannot have both NORM=1 and MI=1."""
    df = loader.get_split("all")
    conflicts = df[(df["label_NORM"] == 1) & (df["label_MI"] == 1)]
    assert len(conflicts) == 0, f"Found {len(conflicts)} records with both NORM and MI positive"


def test_signal_loading_100hz(loader):
    """Verifies loading a 100 Hz record (1000 samples, 12 leads)."""
    record = loader.get_record(ecg_id=1, sampling_rate=100)
    
    assert record["ecg_id"] == 1
    assert "signal" in record
    signal = record["signal"]
    
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (1000, 12), f"Expected (1000, 12), got {signal.shape}"
    assert not np.isnan(signal).any(), "Signal contains NaN values"
    assert record["meta"]["fs"] == 100
    assert len(record["meta"]["sig_name"]) == 12


def test_signal_loading_500hz(loader):
    """Verifies loading a 500 Hz record (5000 samples, 12 leads)."""
    record = loader.get_record(ecg_id=1, sampling_rate=500)
    
    signal = record["signal"]
    assert signal.shape == (5000, 12), f"Expected (5000, 12), got {signal.shape}"
    assert record["meta"]["fs"] == 500


def test_positive_cases_present(loader):
    """Verifies that key clinical conditions (MI, AFIB, LBBB, RBBB, LVH) exist in the dataset."""
    df = loader.get_split("all")
    assert df["label_MI"].sum() > 2000, "Too few MI positive records"
    assert df["label_AFIB"].sum() > 500, "Too few AFIB positive records"
    assert df["label_LBBB"].sum() > 400, "Too few LBBB positive records"
    assert df["label_RBBB"].sum() > 800, "Too few RBBB positive records"
    assert df["label_LVH"].sum() > 1000, "Too few LVH positive records"
