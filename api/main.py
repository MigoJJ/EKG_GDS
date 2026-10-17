"""
FastAPI Microservice for AI-Assisted 12-Lead EKG Reading and EMR Integration.
Provides REST endpoints for quantitative interval extraction, 1D-ResNet ONNX classification,
and Ollama LLM clinical interpretation.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, Dict, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import (
    ECGAnalysisRequest,
    ECGAnalysisResponse,
    HealthResponse,
    FeatureExtractionResponse,
    ClassificationResponse,
)
from pipeline.ptbxl_loader import PTBXLLoader
from pipeline.feature_extractor import ECGFeatureExtractor
from pipeline.classifier import ECGClassifier
from pipeline.ollama_client import OllamaEKGClient, MANDATORY_DISCLAIMER


# Global singleton pipeline components
loader: Optional[PTBXLLoader] = None
extractor: Optional[ECGFeatureExtractor] = None
classifier: Optional[ECGClassifier] = None
ollama: Optional[OllamaEKGClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and warms up models during application startup."""
    global loader, extractor, classifier, ollama
    print("[Startup] Initializing EKG Assistant components...")
    try:
        loader = PTBXLLoader()
    except Exception as e:
        print(f"[Warn] PTB-XL loader init failed (dataset may be offline): {e}")

    extractor = ECGFeatureExtractor(sampling_rate=100)
    
    try:
        classifier = ECGClassifier()
        # Warmup ONNX runtime
        dummy_signal = np.zeros((1000, 12), dtype=np.float32)
        _ = classifier.predict(dummy_signal)
    except Exception as e:
        print(f"[Error] Failed to initialize ONNX classifier: {e}")

    ollama = OllamaEKGClient()
    print("[Startup] All components initialized successfully.")
    yield
    print("[Shutdown] Cleaning up resources.")


app = FastAPI(
    title="12-Lead EKG AI Reading Assistant API",
    description="Microservice providing quantitative interval extraction, 1D-ResNet multi-label classification, and Ollama clinical narrative reports.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local web dashboards, Electron, and JavaFX WebView
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_INDEX = Path(__file__).parent.parent / "static" / "index.html"


@app.get("/", response_class=FileResponse, tags=["Web UI"])
@app.get("/ui", response_class=FileResponse, tags=["Web UI"])
async def serve_dashboard():
    """Serves the interactive EKG AI Reading Assistant web dashboard."""
    if not STATIC_INDEX.exists():
        raise HTTPException(status_code=404, detail="Web UI not found")
    return FileResponse(STATIC_INDEX)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns the operational status of all pipeline components."""
    onnx_ok = classifier is not None and classifier.session is not None
    ollama_ok = ollama is not None and ollama.check_health()
    dataset_ok = loader is not None and loader.df is not None

    overall_status = "healthy" if (onnx_ok and ollama_ok) else "degraded"
    return HealthResponse(
        status=overall_status,
        onnx_model_loaded=onnx_ok,
        ollama_connected=ollama_ok,
        ptbxl_dataset_available=dataset_ok,
        version="1.0.0",
    )


def _resolve_input_signal(request: ECGAnalysisRequest) -> tuple[np.ndarray, dict]:
    """Resolves signal and patient info from either raw array or PTB-XL database ID."""
    global loader
    patient_info = {
        "age": request.patient_age,
        "sex": request.patient_sex or "M",
        "symptoms": request.symptoms or "심전도 검사",
    }

    if request.signal is not None:
        sig = np.array(request.signal, dtype=np.float32)
        if sig.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signal must be a 2D array, received {sig.ndim} dimensions",
            )
        if sig.shape[1] == 12:
            pass  # Expected (samples, 12)
        elif sig.shape[0] == 12:
            sig = sig.T  # Transpose (12, samples) -> (samples, 12)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected 12 leads, received shape {sig.shape}",
            )
        return sig, patient_info

    elif request.ecg_id is not None:
        if loader is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PTB-XL dataset is not mounted or available",
            )
        try:
            rec = loader.get_record(request.ecg_id, sampling_rate=request.sampling_rate)
            patient_info["age"] = rec["age"] if request.patient_age is None else request.patient_age
            patient_info["sex"] = rec["sex"] if request.patient_sex is None else request.patient_sex
            patient_info["doctor_report"] = rec.get("doctor_report", "")
            return rec["signal"], patient_info
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PTB-XL Record ID {request.ecg_id} not found",
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'signal' (12-lead array) or 'ecg_id' (PTB-XL record ID)",
        )


@app.post("/api/v1/ecg/features", response_model=FeatureExtractionResponse, tags=["Analysis"])
async def extract_features_only(request: ECGAnalysisRequest):
    """Fast extraction (<10ms) of quantitative intervals (HR, PR, QRS, QTc, Axis, ST deviations)."""
    global extractor
    sig, patient_info = _resolve_input_signal(request)
    feats = extractor.extract_features(
        sig, patient_sex=patient_info["sex"], patient_age=patient_info.get("age")
    )
    return FeatureExtractionResponse(**feats)


@app.post("/api/v1/ecg/classify", response_model=ClassificationResponse, tags=["Analysis"])
async def classify_only(request: ECGAnalysisRequest):
    """Ultra-fast (<5ms) 1D-ResNet ONNX classification returning multi-label probabilities."""
    global classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    sig, _ = _resolve_input_signal(request)
    result = classifier.predict(sig)
    return ClassificationResponse(**result)


@app.post("/api/v1/ecg/analyze", response_model=ECGAnalysisResponse, tags=["Analysis"])
async def analyze_ecg(request: ECGAnalysisRequest):
    """
    End-to-end clinical ECG reading endpoint:
    Signal Preprocessing -> Feature Extraction -> Deep Learning ONNX -> Ollama Clinical Report.
    """
    global extractor, classifier, ollama
    if classifier is None or ollama is None:
        raise HTTPException(status_code=503, detail="AI engine not ready")

    t_total_start = time.perf_counter()
    sig, patient_info = _resolve_input_signal(request)

    # 1. Feature extraction
    t0 = time.perf_counter()
    feats = extractor.extract_features(
        sig, patient_sex=patient_info["sex"], patient_age=patient_info.get("age")
    )
    fe_latency = (time.perf_counter() - t0) * 1000.0

    # 2. Deep learning classification (with LAFB Hybrid OR Rule enabled)
    t0 = time.perf_counter()
    cls_res = classifier.predict(
        sig,
        heart_axis_deg=feats.get("heart_axis_deg"),
        stemi_suspected=feats.get("stemi_suspected", False),
        stemi_region=feats.get("stemi_region")
    )
    dl_latency = (time.perf_counter() - t0) * 1000.0

    # 3. Ollama clinical report generation
    narrative_report = None
    structured_json = None
    ollama_latency = 0.0

    if request.output_format in ["narrative", "both"]:
        t0 = time.perf_counter()
        narrative_report = ollama.generate_narrative_report(feats, cls_res, patient_info)
        ollama_latency += (time.perf_counter() - t0) * 1000.0

    if request.output_format in ["json", "both"]:
        t0 = time.perf_counter()
        structured_json = ollama.generate_structured_json(feats, cls_res, patient_info)
        ollama_latency += (time.perf_counter() - t0) * 1000.0

    total_latency = (time.perf_counter() - t_total_start) * 1000.0

    return ECGAnalysisResponse(
        status="success",
        patient_info=patient_info,
        quantitative_features=FeatureExtractionResponse(**feats),
        deep_learning_classification=ClassificationResponse(**cls_res),
        ollama_narrative_report=narrative_report,
        emr_structured_json=structured_json,
        latency_ms={
            "feature_extraction_ms": round(fe_latency, 2),
            "dl_inference_ms": round(dl_latency, 2),
            "ollama_generation_ms": round(ollama_latency, 2),
            "total_ms": round(total_latency, 2),
        },
        disclaimer=MANDATORY_DISCLAIMER,
    )


@app.get("/api/v1/ecg/sample/{ecg_id}", response_model=ECGAnalysisResponse, tags=["Analysis"])
async def analyze_sample_record(ecg_id: int, format: str = "narrative"):
    """Convenience endpoint to inspect and run analysis on any PTB-XL record by ID.
    format: 'narrative' (default, ~15s), 'json' (~10s), 'none' (instant, <10ms), or 'both'.
    """
    req = ECGAnalysisRequest(ecg_id=ecg_id, output_format=format)
    return await analyze_ecg(req)
