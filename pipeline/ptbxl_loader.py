"""
PTB-XL ECG Dataset Loader & 11-Class Diagnostic Mapping Pipeline
Supports 12-lead signal loading (100Hz / 500Hz) and multi-label ground truth extraction.
"""

from __future__ import annotations
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import wfdb
from pipeline.config import PTBXL_DIR


# 11-Class Diagnostic Taxonomy for Clinical ECG
DIAGNOSTIC_CLASSES = [
    "NORM",  # Normal ECG
    "MI",    # Myocardial Infarction (AMI, IMI, ASMI, etc.)
    "AFIB",  # Atrial Fibrillation
    "PVC",   # Premature Ventricular Contractions
    "PAC",   # Premature Atrial Contractions
    "1AVB",  # 1st Degree AV Block
    "RBBB",  # Right Bundle Branch Block (CRBBB, IRBBB)
    "LBBB",  # Left Bundle Branch Block (CLBBB, ILBBB)
    "LAFB",  # Left Anterior Fascicular Block
    "LVH",   # Left Ventricular Hypertrophy (LVH, VCLVH)
    "STTC",  # Non-specific ST-T changes / ischemia
]

# Mapping SCP codes from PTB-XL to the 11-class diagnostic classes
SCP_TO_CLASS_MAP = {
    # Normal
    "NORM": "NORM",
    
    # Myocardial Infarction
    "IMI": "MI", "ASMI": "MI", "ILMI": "MI", "ALMI": "MI",
    "AMI": "MI", "LMIMI": "MI", "LMI": "MI", "IPLMI": "MI", "IPMI": "MI", "PMI": "MI",
    
    # Arrhythmias & Ectopy
    "AFIB": "AFIB",
    "PVC": "PVC",
    "PAC": "PAC",
    
    # Conduction Delays
    "1AVB": "1AVB",
    "CRBBB": "RBBB", "IRBBB": "RBBB",
    "CLBBB": "LBBB", "ILBBB": "LBBB",
    "LAFB": "LAFB", "LPFB": "LAFB",
    
    # Hypertrophy
    "LVH": "LVH", "VCLVH": "LVH",
    
    # ST-T Abnormalities
    "STTC": "STTC", "NDT": "STTC", "NST_": "STTC", "STD_": "STTC",
    "STE_": "STTC", "INVT": "STTC", "DIG": "STTC", "LNGQT": "STTC", "ISC_": "STTC",
}

STANDARD_12_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class PTBXLLoader:
    """Loader and preprocessor for the PTB-XL 12-lead ECG database."""

    def __init__(self, data_dir: Union[str, Path] = PTBXL_DIR):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / "ptbxl_database.csv"
        self.scp_path = self.data_dir / "scp_statements.csv"
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"PTB-XL database CSV not found at: {self.csv_path}")
            
        self.df: Optional[pd.DataFrame] = None
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Loads and parses database annotations, assigning 11-class multi-labels."""
        df = pd.read_csv(self.csv_path, index_col="ecg_id")
        # Parse scp_codes column from string representation of dictionary
        df["scp_dict"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
        
        # Build 11-class binary label matrix
        label_matrix = np.zeros((len(df), len(DIAGNOSTIC_CLASSES)), dtype=np.int32)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            scp_dict = row["scp_dict"]
            classes_present = set()
            for scp_code, likelihood in scp_dict.items():
                if scp_code in SCP_TO_CLASS_MAP:
                    # Rhythm & ectopy codes (AFIB, PVC, PAC) are recorded with 0.0 by Schiller devices;
                    # their presence in scp_dict indicates positive annotation.
                    # Diagnostic statements use likelihood > 0.
                    if scp_code in {"AFIB", "PVC", "PAC"} or likelihood > 0:
                        classes_present.add(SCP_TO_CLASS_MAP[scp_code])
                    
            # Clinical Rule: If definite Pathology (MI) is present, remove NORM
            if "MI" in classes_present and "NORM" in classes_present:
                classes_present.remove("NORM")
                
            for cls_idx, cls_name in enumerate(DIAGNOSTIC_CLASSES):
                if cls_name in classes_present:
                    label_matrix[idx, cls_idx] = 1

        for cls_idx, cls_name in enumerate(DIAGNOSTIC_CLASSES):
            df[f"label_{cls_name}"] = label_matrix[:, cls_idx]
            
        self.df = df

    def get_split(self, split: str = "train") -> pd.DataFrame:
        """
        Retrieves train, val, or test split according to official PTB-XL folds:
        - train: folds 1-8
        - val: fold 9
        - test: fold 10 (Strict evaluation set)
        """
        if self.df is None:
            self._load_metadata()
            
        if split == "train":
            return self.df[self.df["strat_fold"].between(1, 8)].copy()
        elif split == "val":
            return self.df[self.df["strat_fold"] == 9].copy()
        elif split == "test":
            return self.df[self.df["strat_fold"] == 10].copy()
        elif split == "all":
            return self.df.copy()
        else:
            raise ValueError(f"Unknown split: {split}. Choose from 'train', 'val', 'test', 'all'.")

    def load_signal(
        self,
        record_path: str,
        sampling_rate: int = 100
    ) -> Tuple[np.ndarray, Dict[str, Union[int, List[str]]]]:
        """
        Loads 12-lead raw ECG waveform using WFDB.
        sampling_rate: 100 (1000 samples, 10s) or 500 (5000 samples, 10s).
        Returns:
            signal: (samples, 12) float numpy array in mV.
            meta: dictionary containing channel names, sampling rate, units.
        """
        full_path = self.data_dir / record_path
        signal, fields = wfdb.rdsamp(str(full_path))
        meta = {
            "fs": fields.get("fs", sampling_rate),
            "sig_name": fields.get("sig_name", STANDARD_12_LEADS),
            "units": fields.get("units", ["mV"] * 12),
            "n_sig": fields.get("n_sig", 12),
            "sig_len": fields.get("sig_len", signal.shape[0]),
        }
        return signal, meta

    def get_record(
        self,
        ecg_id: int,
        sampling_rate: int = 100
    ) -> Dict[str, Union[int, str, float, np.ndarray, Dict]]:
        """
        Fetches a comprehensive record including metadata, labels, and signal for a given ecg_id.
        """
        if self.df is None:
            self._load_metadata()
            
        if ecg_id not in self.df.index:
            raise KeyError(f"ecg_id {ecg_id} not found in database.")
            
        row = self.df.loc[ecg_id]
        rel_path = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
        signal, meta = self.load_signal(rel_path, sampling_rate=sampling_rate)
        
        labels = {cls_name: int(row[f"label_{cls_name}"]) for cls_name in DIAGNOSTIC_CLASSES}
        
        return {
            "ecg_id": ecg_id,
            "patient_id": row["patient_id"],
            "age": row["age"],
            "sex": "M" if row["sex"] == 0 else "F",  # 0: Male, 1: Female in PTB-XL
            "height": row["height"],
            "weight": row["weight"],
            "doctor_report": row.get("report", ""),
            "scp_codes": row["scp_dict"],
            "labels": labels,
            "strat_fold": row["strat_fold"],
            "signal": signal,
            "meta": meta,
        }
