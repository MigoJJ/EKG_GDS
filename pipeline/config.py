"""
Central Path & Service Configuration
All paths resolve relative to the project root by default and can be overridden with environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Model artifacts
MODELS_DIR = _env_path("EKG_MODELS_DIR", PROJECT_ROOT / "models")
ONNX_MODEL_PATH = _env_path("EKG_ONNX_MODEL", MODELS_DIR / "ecg_resnet18_11class.onnx")
TORCH_WEIGHTS_PATH = _env_path("EKG_TORCH_WEIGHTS", MODELS_DIR / "best_ecg_resnet18.pt")
CUTOFFS_PATH = _env_path("EKG_CUTOFFS", MODELS_DIR / "clinical_cutoffs.json")
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Datasets
PTBXL_DIR = _env_path("EKG_PTBXL_DIR", PROJECT_ROOT / "data" / "ptb-xl")

# Ollama LLM service
OLLAMA_BASE_URL = os.environ.get("EKG_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("EKG_OLLAMA_MODEL", "qwen2.5:3b")
