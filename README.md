# 12-Lead EKG AI Reading Assistant & EMR Integration
# gemini --resume 638b7b27-b646-4dee-b94d-2adbc3aabdce  20261015
An end-to-end, CPU-optimized medical clinical decision support system combining:
1. **Signal Preprocessing & Feature Extraction** (SciPy / NeuroKit2): Zero-phase bandpass/notch filtering, R-Peak detection, HR, PR, QRS, QT/QTc (Bazett), QRS frontal axis, ST deviations, and Sokolow-Lyon LVH criteria.
2. **1D-ResNet18 Deep Learning Diagnostic Classifier** (ONNX Runtime): 11-class multi-label probabilities in **4.0 ms** on Intel Core i7-6700 CPU.
3. **Ollama Clinical LLM Interpretation** (Qwen 2.5): Narrative EMR report and structured JSON with KCD-10 disease codes, hallucination prevention guardrails, and mandatory legal disclaimers.
4. **FastAPI Microservice**: REST endpoints for real-time EMR client integration (Eclipse `com.emr.gds`).

---

## Quick Start

### 1. Launch the FastAPI Backend Service
```bash
cd /home/migowj/ekg-assistant
./run_server.sh
```
- Interactive API Documentation: **http://localhost:8000/docs** (Swagger UI)
- Health Check: **http://localhost:8000/api/v1/health**

### 2. Run Test Suite
```bash
cd /home/migowj/ekg-assistant
PYTHONPATH=. .venv/bin/pytest tests/ -v
```
*(33 passed, 3 xfailed — the xfailed tests in `tests/test_stemi_safety.py` document known STEMI-criteria defects scheduled for Phase 1 of [upgrade20261016.md](upgrade20261016.md))*

### Configuration
Paths and services are defined in `pipeline/config.py` and default to locations inside the project. Override with environment variables:

| Variable | Default |
| :--- | :--- |
| `EKG_MODELS_DIR` | `models/` |
| `EKG_ONNX_MODEL` | `models/ecg_resnet18_11class.onnx` |
| `EKG_TORCH_WEIGHTS` | `models/best_ecg_resnet18.pt` |
| `EKG_CUTOFFS` | `models/clinical_cutoffs.json` |
| `EKG_PTBXL_DIR` | `data/ptb-xl` |
| `EKG_OLLAMA_URL` | `http://localhost:11434` |
| `EKG_OLLAMA_MODEL` | `qwen2.5:3b` |

Model artifacts and their SHA-256 hashes are recorded in `models/registry.json`; `tests/test_model_registry.py` fails if a model file changes without a registry update. Model weights are not tracked in git.

Training dependencies (PyTorch, ONNX) are separate: `pip install -r requirements.txt -r requirements-train.txt`

### 3. Run Demos
```bash
# 1. Dataset record inspection demo
PYTHONPATH=. .venv/bin/python demo_inspect_record.py

# 2. Signal preprocessing & quantitative intervals demo
PYTHONPATH=. .venv/bin/python demo_feature_extraction.py

# 3. 1D-ResNet ONNX classification demo
PYTHONPATH=. .venv/bin/python demo_classification.py

# 4. Complete End-to-End Pipeline demo (Signal -> Features -> DL -> Ollama Report)
PYTHONPATH=. .venv/bin/python demo_full_pipeline.py
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health, ONNX model status, Ollama connection |
| `POST` | `/api/v1/ecg/features` | Fast extraction (<10ms) of HR, PR, QRS, QTc, Axis, ST deviations |
| `POST` | `/api/v1/ecg/classify` | Ultra-fast (<5ms) 1D-ResNet ONNX multi-label classification |
| `POST` | `/api/v1/ecg/analyze` | Complete pipeline: Signal → Features → Classifier → Ollama Report |
| `GET` | `/api/v1/ecg/sample/{id}` | Convenience endpoint to analyze a PTB-XL record by ID |

---

## 11-Class Diagnostic Taxonomy

- **NORM**: Normal ECG
- **MI**: Myocardial Infarction (AMI, IMI, ASMI, etc. - KCD: `I21.0`)
- **AFIB**: Atrial Fibrillation (KCD: `I48.0`)
- **PVC**: Premature Ventricular Contractions (KCD: `I49.3`)
- **PAC**: Premature Atrial Contractions (KCD: `I49.1`)
- **1AVB**: First-Degree AV Block (KCD: `I44.0`)
- **RBBB**: Right Bundle Branch Block (KCD: `I45.1`)
- **LBBB**: Left Bundle Branch Block (KCD: `I45.2`)
- **LAFB**: Left Anterior Fascicular Block (KCD: `I44.4`)
- **LVH**: Left Ventricular Hypertrophy (KCD: `I51.7`)
- **STTC**: Nonspecific ST-T Changes / Ischemia (KCD: `R94.3`)

---

## Java EMR Integration

See [EMR_JAVA_INTEGRATION.md](EMR_JAVA_INTEGRATION.md) for ready-to-use Java service classes (`EkgAiService.java`) and JavaFX asynchronous UI wiring for Eclipse `GDSEMREKG`.
