"""
Model Registry Integrity Tests
Verifies that deployed model artifacts match the SHA-256 hashes recorded in models/registry.json.
"""

import hashlib
import json
import pytest
from pipeline.config import MODELS_DIR, REGISTRY_PATH


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_active_model_is_registered(registry):
    assert registry["active_model"] in registry["models"]


@pytest.mark.parametrize("artifact", ["onnx", "torch_weights", "cutoffs"])
def test_artifact_hash_matches_registry(registry, artifact):
    """Fails if a model file was replaced without updating registry.json."""
    entry = registry["models"][registry["active_model"]]["artifacts"][artifact]
    path = MODELS_DIR / entry["path"]
    if not path.exists():
        pytest.skip(f"{path} not present (model weights are not tracked in git)")
    assert _sha256(path) == entry["sha256"], f"{entry['path']} does not match registry.json"
