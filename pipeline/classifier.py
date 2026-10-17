"""
ECG Deep Learning 11-Class Diagnostic Classifier Module
Runs high-performance ONNX Runtime inference on 12-lead ECG signals with clinical thresholding.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
from pipeline.config import CUTOFFS_PATH, ONNX_MODEL_PATH
from pipeline.ptbxl_loader import DIAGNOSTIC_CLASSES


# Clinical Decision Thresholds (Multi-label threshold tuning for clinical sensitivity)
DEFAULT_THRESHOLDS = {
    "NORM": 0.5000,  # Baseline normal (Fold 9 Sens 87.6% >= 80% 유지)
    "MI":   0.1057,  # MI (PTB-XL 기준) 목표 90% 달성 튜닝
    "AFIB": 0.1500,  # 심방세동 (Fold 9 Sens 90.1% >= 90% 유지)
    "PVC":  0.3000,  # 심실조기수축 (Fold 9 Sens 93.0% >= 80% 유지)
    "PAC":  0.0113,  # 심방조기수축 목표 80% 달성 튜닝
    "1AVB": 0.0641,  # 1도방실차단 목표 80% 달성 튜닝
    "RBBB": 0.4136,  # 우각차단 목표 80% 달성 튜닝
    "LBBB": 0.5000,  # 좌각차단 (Fold 9 Sens 83.6% >= 80% 유지)
    "LAFB": 0.4556,  # 좌전섬유속차단 목표 80% 달성 튜닝
    "LVH":  0.0669,  # 좌심실비대 목표 80% 달성 튜닝
    "STTC": 0.3020,  # ST-T변화 목표 80% 달성 튜닝
}


def get_clinical_cutoffs() -> Dict[str, float]:
    """Loads cut-offs from models/clinical_cutoffs.json if available, else defaults."""
    import json
    cfg_file = CUTOFFS_PATH
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("cutoffs", DEFAULT_THRESHOLDS.copy())
        except Exception:
            pass
    return DEFAULT_THRESHOLDS.copy()


class ECGClassifier:
    """Inference engine for 12-lead 11-class ECG diagnostic classification."""

    def __init__(
        self,
        model_path: Union[str, Path] = ONNX_MODEL_PATH,
        thresholds: Optional[Dict[str, float]] = None,
        num_threads: int = 4,
    ):
        self.model_path = Path(model_path)
        self.thresholds = thresholds or get_clinical_cutoffs()

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {self.model_path}")

        # Configure ONNX Runtime session for optimal CPU utilization
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Prepares signal tensor for the network:
        - Transposes (samples, 12) -> (12, samples) if necessary.
        - Standardizes each lead to zero-mean, unit-variance.
        - Adds batch dimension: (1, 12, samples).
        """
        sig = raw_signal.astype(np.float32)

        # Transpose if given as (samples, leads) e.g. (1000, 12)
        if sig.ndim == 2:
            if sig.shape[1] == 12:
                sig = sig.T  # (12, 1000)
            elif sig.shape[0] != 12:
                raise ValueError(f"Expected 12-lead ECG, received shape {raw_signal.shape}")
            sig = sig[np.newaxis, :, :]  # (1, 12, 1000)
        elif sig.ndim == 3 and sig.shape[1] != 12 and sig.shape[2] == 12:
            sig = np.transpose(sig, (0, 2, 1))

        # Channel-wise standard normalization
        for b in range(sig.shape[0]):
            for ch in range(sig.shape[1]):
                std = np.std(sig[b, ch])
                if std > 1e-6:
                    sig[b, ch] = (sig[b, ch] - np.mean(sig[b, ch])) / std
                else:
                    sig[b, ch] = 0.0

        return sig

    def predict(
        self,
        raw_signal: np.ndarray,
        heart_axis_deg: Optional[float] = None,
        stemi_suspected: bool = False,
        stemi_region: Optional[str] = None,
        return_top_k: int = 3
    ) -> Dict[str, Union[Dict[str, float], Dict[str, bool], List[Tuple[str, float]], List[str], str]]:
        """
        Executes model inference and applies clinical decision thresholds.
        Args:
            raw_signal: (samples, 12) or (12, samples) array in mV.
            heart_axis_deg: optional frontal QRS axis (used for LAFB Hybrid OR rule).
            return_top_k: number of top findings to return.
        """
        tensor = self.preprocess_signal(raw_signal)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probs_array = outputs[0]  # (11,)

        probs_dict = {
            cls_name: round(float(probs_array[i]), 4)
            for i, cls_name in enumerate(DIAGNOSTIC_CLASSES)
        }

        # Apply clinical thresholds
        preds_dict = {}
        positive_findings = []
        for cls_name, prob in probs_dict.items():
            threshold = self.thresholds.get(cls_name, 0.50)
            is_pos = prob >= threshold
            preds_dict[cls_name] = is_pos
            if is_pos:
                positive_findings.append(cls_name)

        # 3) LAFB Hybrid OR Rule: DL prediction OR Frontal QRS Axis <= -30° (Left Axis Deviation)
        if heart_axis_deg is not None and heart_axis_deg <= -30.0:
            preds_dict["LAFB"] = True
            if "LAFB" not in positive_findings:
                positive_findings.append("LAFB")

        # Enforce clinical rule: Definite MI suppresses NORM
        if preds_dict.get("MI", False) and preds_dict.get("NORM", False):
            preds_dict["NORM"] = False
            if "NORM" in positive_findings:
                positive_findings.remove("NORM")

        # 1) & 2) Separate primary findings vs reference findings (PAC and LVH marked as 참고 소견 due to low PPV)
        REFERENCE_CLASSES = {"PAC", "LVH"}
        primary_findings = [c for c in positive_findings if c not in REFERENCE_CLASSES]
        reference_findings = [c for c in positive_findings if c in REFERENCE_CLASSES]

        # Top-k findings
        sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
        top_k = sorted_probs[:return_top_k]

        # 4) Clinical Risk Stratification: separate STEMI from other MI
        # EMERGENCY: STEMI criteria met on ECG (ST elevation with contiguity)
        if stemi_suspected:
            urgency = "EMERGENCY"
        # URGENT: DL predicts MI but no STEMI on ECG (likely NSTEMI or old MI)
        # Also URGENT: AFIB or LBBB (high-risk conduction abnormality)
        elif preds_dict.get("MI", False) or preds_dict.get("AFIB", False) or preds_dict.get("LBBB", False):
            urgency = "URGENT"
        # ROUTINE: normal ECG or minor findings (includes NORM)
        else:
            urgency = "ROUTINE"

        return {
            "probabilities": probs_dict,
            "predictions": preds_dict,
            "positive_findings": positive_findings,
            "primary_findings": primary_findings,
            "reference_findings": reference_findings,
            "top_findings": top_k,
            "clinical_urgency": urgency,
        }
