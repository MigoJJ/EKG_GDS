"""
Phase 5: Security Configuration

Centralized security settings for hospital EMR integration:
- HTTPS/TLS configuration
- Authentication settings
- CORS policies
- Rate limiting
- Audit logging
- Data protection
"""

import os
from typing import List


# ============================================================================
# HTTPS / TLS Configuration
# ============================================================================

# Enable HTTPS in production
USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"

# TLS certificate paths (production)
TLS_CERTFILE = os.getenv("TLS_CERTFILE", "/etc/ssl/certs/ecg_server.crt")
TLS_KEYFILE = os.getenv("TLS_KEYFILE", "/etc/ssl/private/ecg_server.key")

# SSL/TLS version
SSL_MIN_VERSION = "TLSv1_3"  # Require TLS 1.3 or higher


# ============================================================================
# Server Binding (Network Security)
# ============================================================================

# Bind address: 127.0.0.1 (local) or 10.x.x.x (internal network)
# DO NOT use 0.0.0.0 in production (exposes to all IPs)
BIND_HOST = os.getenv("EKG_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.getenv("EKG_BIND_PORT", "8000"))

# Allowed internal network ranges (if using internal IP)
INTERNAL_NETWORKS = [
    "127.0.0.1",  # Localhost
    "10.0.0.0/8",  # Private network 10.x.x.x
    "172.16.0.0/12",  # Private network 172.16-31.x.x
    "192.168.0.0/16",  # Private network 192.168.x.x
]


# ============================================================================
# Authentication (API Keys)
# ============================================================================

# API key header name
API_KEY_HEADER = "Authorization"

# API key scheme (use "Bearer" for RFC 6750 compliance)
API_KEY_SCHEME = "Bearer"

# Require API key for all endpoints (whitelist exceptions below)
REQUIRE_AUTH = True

# Endpoints that don't require authentication
AUTH_EXEMPT_PATHS = [
    "/",
    "/ui",
    "/health",
    "/api/v1/health",
    "/docs",
    "/redoc",
    "/openapi.json",
]

# Default API key rotation (days)
API_KEY_ROTATION_DAYS = 90


# ============================================================================
# CORS (Cross-Origin Resource Sharing)
# ============================================================================

# Production: Strict CORS (whitelist hospitals only)
CORS_ALLOWED_ORIGINS = [
    # Development
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://localhost:5000",

    # Hospital EMR systems (examples)
    "https://emr.seoul-medical.kr",
    "https://emr.seoul-medical.kr:8443",
    "https://emr.busan-hospital.kr",
]

# Allow cookies/credentials in CORS requests
CORS_ALLOW_CREDENTIALS = True

# CORS methods
CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

# CORS headers
CORS_ALLOW_HEADERS = [
    "Authorization",
    "Content-Type",
    "X-Hospital-ID",  # Custom: hospital identifier
    "X-Request-ID",  # Custom: request tracking
]

# CORS max age (seconds)
CORS_MAX_AGE = 3600


# ============================================================================
# Rate Limiting
# ============================================================================

# Per API key
RATE_LIMIT_PER_MINUTE = 60  # Default
RATE_LIMIT_PER_DAY = 1000  # Default

# Development key (higher limits)
RATE_LIMIT_DEV_PER_MINUTE = 120
RATE_LIMIT_DEV_PER_DAY = 5000

# Per IP address (backup, global)
RATE_LIMIT_IP_PER_MINUTE = 100

# Rate limit check: include this to enable
ENABLE_RATE_LIMITING = True


# ============================================================================
# Audit Logging & Data Protection
# ============================================================================

# Log all API requests
LOG_REQUESTS = True

# Log request/response bodies (be careful with patient data!)
LOG_BODIES = False

# Encrypt sensitive data at rest
ENCRYPT_LOGS = True

# Encryption key (use environment variable in production)
ENCRYPTION_KEY = os.getenv("EKG_ENCRYPTION_KEY", "change-me-in-production")

# Audit log directory
AUDIT_LOG_DIR = os.getenv("EKG_AUDIT_LOG_DIR", "logs/audit")

# Maximum log file size (MB)
MAX_LOG_SIZE = 100

# Retention period (days) - delete logs older than this
LOG_RETENTION_DAYS = 90


# ============================================================================
# HIPAA / GDPR / Medical Data Protection
# ============================================================================

# Never store raw ECG signals in logs
PROTECT_ECG_SIGNAL = True

# Hash input signals (for anonymization)
USE_SIGNAL_HASH = True

# Mask hospital IDs in logs (use generic identifiers)
MASK_HOSPITAL_ID = True

# PII (Personally Identifiable Information) fields to never log
PII_FIELDS = [
    "patient_name",
    "patient_id",
    "medical_record_number",
    "ssn",
    "date_of_birth",
]

# Timestamp granularity in logs (day = less privacy leakage)
LOG_TIMESTAMP_GRANULARITY = "day"  # "day", "hour", "minute", "second"


# ============================================================================
# Error Handling
# ============================================================================

# Return detailed error messages only in development
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Hide internal server errors from clients (production)
HIDE_SERVER_ERRORS = not DEBUG_MODE

# Log internal server errors to separate file
LOG_ERRORS_TO_FILE = True
ERROR_LOG_FILE = "logs/errors.log"


# ============================================================================
# Database & External Services
# ============================================================================

# Connection timeouts (seconds)
OLLAMA_TIMEOUT_SECONDS = 30
DATABASE_TIMEOUT_SECONDS = 10

# Retry policy for external services
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


# ============================================================================
# Security Headers
# ============================================================================

# HTTP security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",  # Prevent MIME type sniffing
    "X-Frame-Options": "DENY",  # Prevent clickjacking
    "X-XSS-Protection": "1; mode=block",  # XSS protection
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",  # HTTPS only
    "Content-Security-Policy": "default-src 'self'",  # Restrict content sources
}


# ============================================================================
# Feature Flags
# ============================================================================

# Enable async analysis (Phase 5)
ENABLE_ASYNC_API = True

# Enable audit logging
ENABLE_AUDIT_LOGGING = True

# Enable rate limiting
ENABLE_RATE_LIMITING = True

# Require HTTPS (production only)
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "false").lower() == "true"


# ============================================================================
# Environment Detection
# ============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production

IS_DEVELOPMENT = ENVIRONMENT == "development"
IS_STAGING = ENVIRONMENT == "staging"
IS_PRODUCTION = ENVIRONMENT == "production"

# Adjust defaults based on environment
if IS_PRODUCTION:
    REQUIRE_AUTH = True
    REQUIRE_HTTPS = True
    HIDE_SERVER_ERRORS = True
    LOG_BODIES = False
    DEBUG_MODE = False

if IS_DEVELOPMENT:
    DEBUG_MODE = True
    HIDE_SERVER_ERRORS = False
    LOG_BODIES = True
