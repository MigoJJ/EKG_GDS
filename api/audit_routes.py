"""
Phase 5 Step 3: Audit API Routes

REST endpoints for querying and updating audit logs.

Endpoints:
  GET /api/v1/audit/requests - List audit entries
  GET /api/v1/audit/requests/{request_id} - Get one entry
  POST /api/v1/audit/feedback - Submit doctor feedback
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, status
from models.audit_logger import audit_logger

router = APIRouter(prefix="/api/v1/audit", tags=["Audit", "Phase 5"])


@router.get("/requests")
async def list_audit_requests(
    request: Request,
    hospital_id: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = 100,
):
    """
    List audit entries (requires API key authentication).

    Query params:
        hospital_id: Filter by hospital (optional)
        date: Filter by date YYYY-MM-DD (optional)
        limit: Max entries (default 100, max 1000)

    Returns:
        {
            "entries": [
                {
                    "request_id": "abc123",
                    "hospital_id": "seoul_med",
                    "timestamp": "2026-10-17",
                    "result": {...},
                    "doctor_feedback": {...}
                },
                ...
            ],
            "total": N
        }
    """
    # Require authentication
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Limit max entries
    limit = min(limit, 1000)

    # Query audit log
    entries = audit_logger.query_entries(
        hospital_id=hospital_id,
        date=date,
        limit=limit,
    )

    return {
        "entries": entries,
        "total": len(entries),
        "hospital_id": hospital_id,
        "date": date,
    }


@router.get("/requests/{request_id}")
async def get_audit_entry(request: Request, request_id: str):
    """
    Get a single audit entry by request_id.

    Returns:
        {
            "request_id": "abc123",
            "hospital_id": "seoul_med",
            "timestamp": "2026-10-17",
            "signal_hash": "a1b2c3d4e5f6...",
            "model_version": "phase3_kaggle_20261017",
            "cutoff_version": "fold9_20261017",
            "result": {
                "primary_diagnosis": ["MI", "RBBB"],
                "kcd_code": "I21.0",
                "urgency": "EMERGENCY",
                "confidence": 0.92,
                "latency_ms": {...}
            },
            "doctor_feedback": {
                "agree": true,
                "corrected_diagnosis": null,
                "notes": "신속한 PCI 권고, 내원함"
            }
        }
    """
    # Require authentication
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    entry = audit_logger.get_entry(request_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Entry {request_id} not found")

    return entry


@router.post("/feedback")
async def submit_doctor_feedback(
    request: Request,
    request_id: str,
    agree: bool,
    corrected_diagnosis: Optional[List[str]] = None,
    corrected_kcd_code: Optional[str] = None,
    notes: Optional[str] = None,
):
    """
    Submit doctor's final judgment for an analysis.

    Called after clinician review of AI result.

    Args:
        request_id: Request ID to add feedback to
        agree: True if doctor agrees with AI diagnosis
        corrected_diagnosis: Doctor's diagnosis if different
        corrected_kcd_code: Doctor's KCD code if different
        notes: Free-text notes from doctor

    Returns:
        {
            "request_id": "abc123",
            "status": "feedback_recorded",
            "message": "Doctor feedback saved"
        }
    """
    # Require authentication
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Log feedback
    audit_logger.log_feedback(
        request_id=request_id,
        agree=agree,
        corrected_diagnosis=corrected_diagnosis,
        corrected_kcd_code=corrected_kcd_code,
        notes=notes,
    )

    return {
        "request_id": request_id,
        "status": "feedback_recorded",
        "message": "Doctor feedback saved",
    }


@router.get("/stats")
async def get_audit_stats(request: Request, date: Optional[str] = None):
    """
    Get audit statistics for a date.

    Returns:
        {
            "date": "2026-10-17",
            "total_analyses": N,
            "doctor_agreement_rate": 0.92,
            "avg_confidence": 0.88,
            "hospitals": {"seoul_med": N, "busan_hospital": N},
            "urgency_distribution": {"EMERGENCY": N, "URGENT": N, "ROUTINE": N}
        }
    """
    # Require authentication
    if not hasattr(request.state, "api_key"):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Query entries
    entries = audit_logger.query_entries(date=date, limit=10000)

    if not entries:
        return {
            "date": date,
            "total_analyses": 0,
            "message": "No entries found"
        }

    # Calculate stats
    total = len(entries)
    with_feedback = [e for e in entries if e.get("doctor_feedback")]
    agree_count = sum(1 for e in with_feedback if e["doctor_feedback"].get("agree"))

    confidences = [e["result"]["confidence"] for e in entries if e.get("result")]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Hospital distribution
    hospitals = {}
    for e in entries:
        h = e.get("hospital_id", "unknown")
        hospitals[h] = hospitals.get(h, 0) + 1

    # Urgency distribution
    urgencies = {}
    for e in entries:
        if e.get("result"):
            u = e["result"].get("urgency", "unknown")
            urgencies[u] = urgencies.get(u, 0) + 1

    return {
        "date": date,
        "total_analyses": total,
        "analyses_with_feedback": len(with_feedback),
        "doctor_agreement_rate": agree_count / len(with_feedback) if with_feedback else None,
        "avg_confidence": round(avg_confidence, 3),
        "hospitals": hospitals,
        "urgency_distribution": urgencies,
    }
