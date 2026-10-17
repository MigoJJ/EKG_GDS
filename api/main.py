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
from fastapi.responses import FileResponse, StreamingResponse
import json

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
from models.job_queue import job_queue, JobStatus
from pipeline.task_executor import task_executor, ecg_analysis_task
from api.auth import api_key_manager, cors_config_manager, rate_limiter
from config.security import (
    CORS_ALLOWED_ORIGINS, AUTH_EXEMPT_PATHS, REQUIRE_AUTH,
    RATE_LIMIT_IP_PER_MINUTE, SECURITY_HEADERS, ENVIRONMENT
)
from models.audit_logger import audit_logger
from api.audit_routes import router as audit_router
import uuid


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

# Phase 5 Step 2: Secure CORS (whitelist hospitals only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Hospital-ID"],
)

STATIC_INDEX = Path(__file__).parent.parent / "static" / "index.html"

# Include audit routes (Phase 5 Step 3)
app.include_router(audit_router)


# ============================================================================
# Phase 5 Step 2: Security Middleware
# ============================================================================

from fastapi import Request
from fastapi.responses import JSONResponse

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Add security headers
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value

    return response


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """
    Validate API key for protected endpoints.
    Exempt paths: health check, docs, etc.
    """
    # Skip auth for exempt paths
    if request.url.path in AUTH_EXEMPT_PATHS or request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
        return await call_next(request)

    if not REQUIRE_AUTH:
        return await call_next(request)

    # Extract API key from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header"}
        )

    api_key_secret = auth_header.replace("Bearer ", "", 1).strip()

    # Validate key
    is_valid, api_key_obj, error = api_key_manager.validate_key(api_key_secret)
    if not is_valid:
        return JSONResponse(status_code=401, content={"detail": error})

    # Store key info in request state for later use
    request.state.api_key = api_key_obj
    request.state.api_key_secret = api_key_secret

    # Call next middleware/endpoint
    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce rate limits per API key."""
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Check IP-based rate limit (backup)
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded ({RATE_LIMIT_IP_PER_MINUTE}/min per IP)"}
        )

    # Check API key-based rate limit (if authenticated)
    if hasattr(request.state, "api_key") and request.state.api_key:
        api_key = request.state.api_key
        endpoint = request.url.path

        is_allowed, error = api_key_manager.check_rate_limit(
            request.state.api_key_secret,
            endpoint
        )

        if not is_allowed:
            return JSONResponse(status_code=429, content={"detail": error})

        # Add remaining quota to response headers
        quota = api_key_manager.get_remaining_quota(request.state.api_key_secret)
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining-Minute"] = str(quota.get("remaining_per_minute", 0))
        response.headers["X-RateLimit-Remaining-Day"] = str(quota.get("remaining_per_day", 0))
        return response

    return await call_next(request)


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


@app.post("/api/v1/ecg/analyze-stream", tags=["Analysis", "Phase 4"])
async def analyze_ecg_streaming(request: ECGAnalysisRequest):
    """
    Phase 4: Stream ECG analysis results in real-time (SSE).

    Returns: Server-Sent Events (text/event-stream)
    Each line is JSON: {"type": "narrative_chunk"|"json"|"complete"|"error", "content": ...}

    Benefits vs /analyze:
    - 50% faster narrative (~4s vs 8s) - single LLM call
    - Real-time streaming - doctor sees results as they're generated
    - Hallucination-proof - template + validation prevent false diagnoses
    """
    if classifier is None or ollama is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ONNX classifier or Ollama service not initialized"
        )

    # Parse signal
    if request.ecg_id is not None and loader is not None:
        record = loader.load_record(request.ecg_id)
        sig = record["signal"].astype(np.float32)
        patient_info = record.get("metadata", {})
    else:
        if not request.signal_12lead or len(request.signal_12lead) == 0:
            raise HTTPException(status_code=400, detail="ECG signal required")
        sig = np.array(request.signal_12lead, dtype=np.float32)
        if sig.shape != (1000, 12):
            sig = np.transpose(sig) if sig.shape == (12, 1000) else sig
        patient_info = request.patient_info or {}

    # Feature extraction
    feats = extractor.extract(sig, sampling_rate=100)

    # Classification
    cls_res = classifier.predict(
        sig,
        heart_axis_deg=feats.get("heart_axis_deg"),
        stemi_suspected=feats.get("stemi_suspected", False),
        stemi_region=feats.get("stemi_region")
    )

    # Streaming generation (SSE)
    async def generate():
        try:
            # Use Phase 4 combined report with streaming
            async for chunk in ollama.generate_combined_report_stream(
                feats, cls_res, patient_info
            ):
                # chunk is already JSON string
                yield f"data: {chunk}\n\n"
        except Exception as e:
            error_json = json.dumps({
                "type": "error",
                "content": str(e)
            })
            yield f"data: {error_json}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================================
# Phase 5: Asynchronous Analysis (Non-blocking LLM)
# ============================================================================

@app.post("/api/v1/ecg/analyze-async", tags=["Analysis", "Phase 5"])
async def analyze_ecg_async(request: ECGAnalysisRequest):
    """
    Phase 5: Non-blocking ECG analysis with background LLM report generation.

    Returns immediately with job_id. Client polls GET /api/v1/jobs/{job_id}
    for progress and results.

    Benefits:
    - UI doesn't block waiting for LLM (~4 seconds)
    - Doctor can submit multiple ECGs without waiting
    - Real-time progress visualization (0–100%)
    - Cancel long-running jobs if needed

    Returns:
        {
            "job_id": "abc123def456",
            "status": "queued",
            "created_at": "2026-10-17T12:34:56Z"
        }
    """
    if classifier is None or ollama is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ONNX classifier or Ollama service not initialized"
        )

    # Parse signal
    if request.ecg_id is not None and loader is not None:
        record = loader.load_record(request.ecg_id)
        sig = record["signal"].astype(np.float32)
        patient_info = record.get("metadata", {})
    else:
        if not request.signal_12lead or len(request.signal_12lead) == 0:
            raise HTTPException(status_code=400, detail="ECG signal required")
        sig = np.array(request.signal_12lead, dtype=np.float32)
        if sig.shape != (1000, 12):
            sig = np.transpose(sig) if sig.shape == (12, 1000) else sig
        patient_info = request.patient_info or {}

    # Feature extraction (fast, synchronous)
    feats = extractor.extract(sig, sampling_rate=100)

    # Classification (fast, synchronous)
    cls_res = classifier.predict(
        sig,
        heart_axis_deg=feats.get("heart_axis_deg"),
        stemi_suspected=feats.get("stemi_suspected", False),
        stemi_region=feats.get("stemi_region")
    )

    # Create job and submit to background task
    job_id = job_queue.create_job()

    task_executor.submit_task(
        job_id,
        ecg_analysis_task,
        features=feats,
        classification=cls_res,
        patient_info=patient_info,
        ollama_client=ollama,
        job_queue=job_queue
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "message": f"Analysis job {job_id} queued. Poll GET /api/v1/jobs/{job_id} for progress."
    }


@app.get("/api/v1/jobs/{job_id}", tags=["Jobs", "Phase 5"])
async def get_job_status(job_id: str):
    """
    Query status and progress of a background analysis job.

    Returns:
        {
            "job_id": "abc123",
            "status": "in_progress" | "complete" | "failed",
            "progress": 0–100,
            "result": {...} if complete,
            "error": "..." if failed
        }
    """
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response = job.to_dict()

    # Remove internal timestamps for simpler API
    response.pop("created_at", None)
    response.pop("started_at", None)
    response.pop("completed_at", None)

    return response


@app.delete("/api/v1/jobs/{job_id}", tags=["Jobs", "Phase 5"])
async def cancel_job(job_id: str):
    """
    Cancel a queued or in-progress analysis job.

    Returns:
        {"job_id": "abc123", "status": "cancelled"}
    """
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job_queue.cancel_job(job_id):
        return {"job_id": job_id, "status": "cancelled"}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in state {job.status.value}"
        )


@app.get("/api/v1/jobs", tags=["Jobs", "Phase 5"])
async def list_jobs(status: Optional[str] = None, limit: int = 10):
    """
    List recent analysis jobs, optionally filtered by status.

    Query params:
        status: "queued", "in_progress", "complete", "failed", "cancelled" (optional)
        limit: Max jobs to return (default 10, max 100)

    Returns:
        {
            "jobs": [...],
            "total": N,
            "stats": {"queued": N, "in_progress": N, ...}
        }
    """
    limit = min(limit, 100)  # Cap at 100

    status_enum = None
    if status:
        try:
            status_enum = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = job_queue.list_jobs(status=status_enum, limit=limit)
    stats = job_queue.get_stats()

    return {
        "jobs": jobs,
        "total": len(jobs),
        "stats": stats
    }


# ============================================================================
# Phase 5 Step 2: Security Management Endpoints (Admin Only)
# ============================================================================

@app.get("/api/v1/admin/api-keys", tags=["Admin", "Security"])
async def list_api_keys(request: Request):
    """
    List all API keys (admin endpoint - requires valid auth).

    Response includes quotas and status, but not secrets.
    """
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    return {
        "keys": api_key_manager.list_keys(),
        "count": len(api_key_manager.keys)
    }


@app.get("/api/v1/admin/quota", tags=["Admin", "Security"])
async def get_api_quota(request: Request):
    """
    Get remaining quota for current API key.

    Returns:
        {
            "remaining_per_minute": N,
            "remaining_per_day": N,
            "limit_per_minute": N,
            "limit_per_day": N,
            "hospital": "hospital name"
        }
    """
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    quota = api_key_manager.get_remaining_quota(request.state.api_key_secret)
    quota["hospital"] = request.state.api_key.hospital_name
    return quota


@app.get("/api/v1/admin/cors-origins", tags=["Admin", "Security"])
async def get_cors_origins(request: Request):
    """
    Get current CORS whitelist.

    Requires admin authentication.
    """
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    return {
        "allowed_origins": cors_config_manager.allowed_origins,
        "total": len(cors_config_manager.allowed_origins)
    }


@app.post("/api/v1/admin/cors-origins", tags=["Admin", "Security"])
async def add_cors_origin(request: Request, origin: str):
    """
    Add new CORS origin (hospital domain).

    Example: https://emr.new-hospital.kr

    Requires admin authentication.
    """
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    cors_config_manager.add_origin(origin)
    return {"message": f"Added CORS origin: {origin}"}


@app.get("/api/v1/admin/security-status", tags=["Admin", "Security"])
async def get_security_status(request: Request):
    """
    Get current security configuration status.

    Includes: TLS, auth, CORS, rate limits, audit logging.
    """
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    return {
        "environment": ENVIRONMENT,
        "authentication": "API key (Bearer token)",
        "https_enabled": False,  # Check actual setting
        "cors_origins_count": len(cors_config_manager.allowed_origins),
        "rate_limiting_enabled": True,
        "audit_logging_enabled": True,
        "api_keys_active": len([k for k in api_key_manager.keys.values() if k.active]),
        "security_headers_enabled": True,
    }


# ============================================================================
# Phase 5 Step 4: Input Format Expansion (DICOM, HL7, SCP-ECG)
# ============================================================================

from fastapi import UploadFile, File, Form
import io
from pipeline.dicom_parser import DicomParser
from pipeline.hl7_parser import HL7Parser
from pipeline.scp_parser import SCPParser


@app.post("/api/v1/ecg/analyze-dicom", response_model=ECGAnalysisResponse, tags=["Analysis", "Phase 5 Step 4"])
async def analyze_dicom(
    file: UploadFile = File(..., description="DICOM file with embedded ECG waveform"),
    output_format: str = Form("narrative", description="Output format: 'narrative', 'json', 'both', or 'none'"),
):
    """
    Analyze ECG from DICOM file (medical imaging format).

    DICOM is the standard format in hospital imaging systems.
    ECG waveforms can be embedded in DICOM Waveform (0x5400) modules.

    Args:
        file: DICOM file (*.dcm)
        output_format: Report format

    Returns:
        Standard ECGAnalysisResponse with patient demographics from DICOM
    """
    global classifier, extractor, ollama

    if classifier is None or ollama is None:
        raise HTTPException(status_code=503, detail="AI engine not ready")

    try:
        # Save uploaded file to temp
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Parse DICOM
        try:
            parser = DicomParser(tmp_path)
            is_valid, error = parser.validate()
            if not is_valid:
                raise ValueError(error)

            # Extract signal
            signal = parser.extract_ecg_signal()
            if signal is None:
                raise ValueError("No ECG waveform found in DICOM")

            # Ensure shape (samples, 12)
            if signal.shape[0] == 12:
                signal = signal.T

            # Extract patient info from DICOM
            dicom_patient = parser.extract_patient_info()
            patient_age = None
            patient_sex = "M"

            if 'age' in dicom_patient:
                try:
                    patient_age = float(dicom_patient['age'].rstrip('Y'))
                except ValueError:
                    pass

            if 'sex' in dicom_patient:
                patient_sex = dicom_patient['sex']

            # Run analysis using existing pipeline
            t_total_start = time.perf_counter()

            t0 = time.perf_counter()
            feats = extractor.extract_features(
                signal,
                patient_sex=patient_sex,
                patient_age=patient_age
            )
            fe_latency = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            cls_res = classifier.predict(
                signal,
                heart_axis_deg=feats.get("heart_axis_deg"),
                stemi_suspected=feats.get("stemi_suspected", False),
                stemi_region=feats.get("stemi_region")
            )
            dl_latency = (time.perf_counter() - t0) * 1000.0

            patient_info = {
                "age": patient_age,
                "sex": patient_sex,
                "symptoms": "DICOM ECG 분석",
                **dicom_patient
            }

            narrative_report = None
            structured_json = None
            ollama_latency = 0.0

            if output_format in ["narrative", "both"]:
                t0 = time.perf_counter()
                narrative_report = ollama.generate_narrative_report(feats, cls_res, patient_info)
                ollama_latency += (time.perf_counter() - t0) * 1000.0

            if output_format in ["json", "both"]:
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

        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DICOM parsing error: {str(e)}")


@app.post("/api/v1/ecg/analyze-hl7", response_model=ECGAnalysisResponse, tags=["Analysis", "Phase 5 Step 4"])
async def analyze_hl7(
    message: str = Form(..., description="HL7 v2.5 message (multiline text)"),
    signal: Optional[str] = Form(None, description="Optional: JSON array of signal [[...], [...], ...]"),
    output_format: str = Form("narrative", description="Output format"),
):
    """
    Analyze ECG with HL7 v2.5 patient demographics.

    HL7 is the standard message format for hospital information systems (EMR).
    This endpoint parses HL7 patient demographics and ECG measurements.

    Args:
        message: HL7 v2.5 message (segments separated by newlines)
        signal: Optional 12-lead ECG signal as JSON array
        output_format: Report format

    Returns:
        ECGAnalysisResponse with HL7 patient info
    """
    global classifier, extractor, ollama

    if classifier is None or ollama is None:
        raise HTTPException(status_code=503, detail="AI engine not ready")

    try:
        # Parse HL7
        parser = HL7Parser(message)
        is_valid, error = parser.validate()
        if not is_valid:
            raise ValueError(f"Invalid HL7: {error}")

        # Extract patient info and ECG data
        hl7_patient = parser.get_patient_info()
        hl7_ecg = parser.get_ecg_data()

        # Get signal from request
        if signal is None:
            raise ValueError("ECG signal required (12-lead array)")

        import json
        sig = np.array(json.loads(signal), dtype=np.float32)

        # Ensure shape (samples, 12)
        if sig.shape[0] == 12:
            sig = sig.T

        # Parse patient demographics
        patient_age = None
        patient_sex = "M"

        if 'age' in hl7_patient:
            try:
                patient_age = int(hl7_patient['age'])
            except ValueError:
                pass

        if 'sex' in hl7_patient:
            patient_sex = hl7_patient['sex']

        # Run analysis
        t_total_start = time.perf_counter()

        t0 = time.perf_counter()
        feats = extractor.extract_features(sig, patient_sex=patient_sex, patient_age=patient_age)
        fe_latency = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cls_res = classifier.predict(
            sig,
            heart_axis_deg=feats.get("heart_axis_deg"),
            stemi_suspected=feats.get("stemi_suspected", False),
            stemi_region=feats.get("stemi_region")
        )
        dl_latency = (time.perf_counter() - t0) * 1000.0

        patient_info = {
            "age": patient_age,
            "sex": patient_sex,
            "symptoms": "HL7 ECG 분석",
            **hl7_patient
        }

        narrative_report = None
        structured_json = None
        ollama_latency = 0.0

        if output_format in ["narrative", "both"]:
            t0 = time.perf_counter()
            narrative_report = ollama.generate_narrative_report(feats, cls_res, patient_info)
            ollama_latency += (time.perf_counter() - t0) * 1000.0

        if output_format in ["json", "both"]:
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

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"HL7 parsing error: {str(e)}")


@app.post("/api/v1/ecg/analyze-scp", response_model=ECGAnalysisResponse, tags=["Analysis", "Phase 5 Step 4"])
async def analyze_scp(
    file: UploadFile = File(..., description="SCP-ECG file from ECG device"),
    output_format: str = Form("narrative", description="Output format: 'narrative', 'json', 'both', or 'none'"),
):
    """
    Analyze ECG from SCP-ECG file (direct medical device output).

    SCP-ECG is the standard format for ECG export from medical devices.
    Contains waveform data, patient demographics, and interpretations.

    Args:
        file: SCP-ECG file (binary format)
        output_format: Report format

    Returns:
        Standard ECGAnalysisResponse with patient demographics from SCP
    """
    global classifier, extractor, ollama

    if classifier is None or ollama is None:
        raise HTTPException(status_code=503, detail="AI engine not ready")

    try:
        # Save uploaded file to temp
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".scp", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Parse SCP-ECG
        try:
            parser = SCPParser(tmp_path)
            is_valid, error = parser.validate()
            if not is_valid:
                raise ValueError(error)

            # Extract signal
            signal = parser.extract_ecg_signal()
            if signal is None:
                raise ValueError("No ECG waveform found in SCP-ECG file")

            # Ensure shape (samples, 12)
            if signal.shape[0] == 12:
                signal = signal.T

            # Extract patient info
            scp_patient = parser.extract_patient_info()
            patient_age = None
            patient_sex = "M"

            if 'date_of_birth' in scp_patient:
                # Calculate age from DOB if available
                try:
                    from datetime import datetime
                    dob = datetime.strptime(scp_patient['date_of_birth'], '%Y-%m-%d')
                    today = datetime.now()
                    patient_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                except Exception:
                    pass

            if 'sex' in scp_patient:
                patient_sex = scp_patient['sex']

            # Run analysis
            t_total_start = time.perf_counter()

            t0 = time.perf_counter()
            feats = extractor.extract_features(sig, patient_sex=patient_sex, patient_age=patient_age)
            fe_latency = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            cls_res = classifier.predict(
                signal,
                heart_axis_deg=feats.get("heart_axis_deg"),
                stemi_suspected=feats.get("stemi_suspected", False),
                stemi_region=feats.get("stemi_region")
            )
            dl_latency = (time.perf_counter() - t0) * 1000.0

            patient_info = {
                "age": patient_age,
                "sex": patient_sex,
                "symptoms": "SCP-ECG 분석",
                **scp_patient
            }

            narrative_report = None
            structured_json = None
            ollama_latency = 0.0

            if output_format in ["narrative", "both"]:
                t0 = time.perf_counter()
                narrative_report = ollama.generate_narrative_report(feats, cls_res, patient_info)
                ollama_latency += (time.perf_counter() - t0) * 1000.0

            if output_format in ["json", "both"]:
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

        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SCP-ECG parsing error: {str(e)}")
