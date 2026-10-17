"""
Phase 5 Step 2: Security & Authentication

Implements API key authentication, CORS restrictions, and rate limiting
for hospital EMR integration.

Features:
  - API key validation (HTTPBearer)
  - Rate limiting per key (requests/minute)
  - CORS whitelist (specific hospitals only)
  - Request logging for audit trail
  - Hospital identification & authorization
"""

from __future__ import annotations
import time
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class APIKey:
    """Represents an API key with permissions and rate limits."""

    def __init__(
        self,
        key_id: str,
        hospital_name: str,
        key_secret: str,
        rate_limit_per_minute: int = 60,
        rate_limit_per_day: int = 1000,
        active: bool = True,
        allowed_endpoints: Optional[List[str]] = None,
    ):
        """
        Initialize API key.

        Args:
            key_id: Short identifier for the key
            hospital_name: Hospital/organization name
            key_secret: Secret key (validate against this)
            rate_limit_per_minute: Requests allowed per minute
            rate_limit_per_day: Requests allowed per day
            active: Whether key is currently active
            allowed_endpoints: List of allowed endpoints (None = all)
        """
        self.key_id = key_id
        self.hospital_name = hospital_name
        self.key_secret = key_secret
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_per_day = rate_limit_per_day
        self.active = active
        self.allowed_endpoints = allowed_endpoints or []
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, any]:
        """Serialize (without secret)."""
        return {
            "key_id": self.key_id,
            "hospital_name": self.hospital_name,
            "active": self.active,
            "rate_limits": {
                "per_minute": self.rate_limit_per_minute,
                "per_day": self.rate_limit_per_day,
            },
            "allowed_endpoints": self.allowed_endpoints if self.allowed_endpoints else ["all"],
        }


class APIKeyManager:
    """Manages API keys and authentication."""

    def __init__(self):
        """Initialize key manager with predefined keys."""
        self.keys: Dict[str, APIKey] = {}
        self.request_count = defaultdict(lambda: defaultdict(list))  # key_id → endpoint → [timestamps]

        # Initialize with example keys (in production, load from database/vault)
        self._init_default_keys()

    def _init_default_keys(self):
        """Load default API keys for known hospitals."""
        # Example: Development/testing
        self.add_key(
            APIKey(
                key_id="dev_001",
                hospital_name="Seoul Medical Center (Dev)",
                key_secret="sk_dev_test_123456789",
                rate_limit_per_minute=120,
                rate_limit_per_day=5000,
                active=True,
            )
        )

        # Example: Production
        self.add_key(
            APIKey(
                key_id="prod_001",
                hospital_name="Seoul Medical Center",
                key_secret="sk_prod_real_abcdefghij",
                rate_limit_per_minute=60,
                rate_limit_per_day=1000,
                active=True,
            )
        )

        # Example: Inactive (testing)
        self.add_key(
            APIKey(
                key_id="test_001",
                hospital_name="Test Hospital",
                key_secret="sk_test_inactive",
                rate_limit_per_minute=10,
                rate_limit_per_day=100,
                active=False,
            )
        )

    def add_key(self, api_key: APIKey):
        """Add or update an API key."""
        self.keys[api_key.key_secret] = api_key
        logger.info(f"Added API key for {api_key.hospital_name}")

    def validate_key(self, key_secret: str) -> Tuple[bool, Optional[APIKey], Optional[str]]:
        """
        Validate API key.

        Args:
            key_secret: Key secret to validate

        Returns:
            (is_valid, api_key_object, error_message)
        """
        if not key_secret:
            return False, None, "API key required"

        api_key = self.keys.get(key_secret)
        if not api_key:
            return False, None, "Invalid API key"

        if not api_key.active:
            return False, None, "API key is inactive"

        return True, api_key, None

    def check_rate_limit(
        self,
        key_secret: str,
        endpoint: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Args:
            key_secret: API key
            endpoint: Endpoint being called (e.g., "/api/v1/ecg/analyze")

        Returns:
            (is_allowed, error_message)
        """
        is_valid, api_key, error = self.validate_key(key_secret)
        if not is_valid:
            return False, error

        now = time.time()
        key_id = api_key.key_id
        one_minute_ago = now - 60
        one_day_ago = now - 86400

        # Get request timestamps
        timestamps = self.request_count[key_id][endpoint]

        # Remove old timestamps
        timestamps[:] = [t for t in timestamps if t > one_day_ago]

        # Check minute limit
        recent_count = sum(1 for t in timestamps if t > one_minute_ago)
        if recent_count >= api_key.rate_limit_per_minute:
            return False, f"Rate limit exceeded ({api_key.rate_limit_per_minute}/min)"

        # Check daily limit
        if len(timestamps) >= api_key.rate_limit_per_day:
            return False, f"Daily limit exceeded ({api_key.rate_limit_per_day}/day)"

        # Record this request
        timestamps.append(now)
        return True, None

    def get_remaining_quota(self, key_secret: str) -> Dict[str, int]:
        """Get remaining quota for a key."""
        is_valid, api_key, _ = self.validate_key(key_secret)
        if not is_valid:
            return {}

        now = time.time()
        key_id = api_key.key_id
        one_minute_ago = now - 60
        one_day_ago = now - 86400

        # Aggregate across all endpoints
        all_timestamps = []
        for endpoint_timestamps in self.request_count[key_id].values():
            all_timestamps.extend(endpoint_timestamps)

        recent_count = sum(1 for t in all_timestamps if t > one_minute_ago)
        daily_count = sum(1 for t in all_timestamps if t > one_day_ago)

        return {
            "remaining_per_minute": max(0, api_key.rate_limit_per_minute - recent_count),
            "remaining_per_day": max(0, api_key.rate_limit_per_day - daily_count),
            "limit_per_minute": api_key.rate_limit_per_minute,
            "limit_per_day": api_key.rate_limit_per_day,
        }

    def list_keys(self) -> list[Dict]:
        """List all API keys (without secrets)."""
        return [key.to_dict() for key in self.keys.values()]


class CORSConfigManager:
    """Manages CORS (Cross-Origin Resource Sharing) whitelist."""

    def __init__(self, allowed_origins: Optional[list[str]] = None):
        """
        Initialize CORS config.

        Args:
            allowed_origins: List of allowed origins (domains)
        """
        # Default: only specific hospital domains
        self.allowed_origins = allowed_origins or [
            "http://127.0.0.1:3000",  # Local dev
            "http://localhost:3000",  # Local dev
            "https://emr.seoul-medical.kr",  # Seoul Medical Center
            "https://emr.busan-hospital.kr",  # Example: Busan Hospital
        ]
        self.allow_credentials = True
        self.allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = ["Authorization", "Content-Type", "X-Hospital-ID"]

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is in whitelist."""
        if not origin:
            return False
        return origin in self.allowed_origins

    def add_origin(self, origin: str):
        """Add new allowed origin."""
        if origin not in self.allowed_origins:
            self.allowed_origins.append(origin)
            logger.info(f"Added CORS origin: {origin}")

    def remove_origin(self, origin: str):
        """Remove allowed origin."""
        if origin in self.allowed_origins:
            self.allowed_origins.remove(origin)
            logger.info(f"Removed CORS origin: {origin}")

    def get_config_dict(self) -> Dict:
        """Get CORS config for FastAPI middleware."""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": self.allow_credentials,
            "allow_methods": self.allow_methods,
            "allow_headers": self.allow_headers,
        }


class RateLimiter:
    """Generic rate limiter by IP address (backup to API key limiting)."""

    def __init__(self):
        """Initialize rate limiter."""
        self.request_history = defaultdict(list)  # ip → [timestamps]
        self.limit_per_minute = 100  # Per IP

    def is_allowed(self, client_ip: str) -> bool:
        """Check if client IP is within rate limit."""
        now = time.time()
        one_minute_ago = now - 60

        # Clean old timestamps
        self.request_history[client_ip] = [
            t for t in self.request_history[client_ip] if t > one_minute_ago
        ]

        # Check limit
        if len(self.request_history[client_ip]) >= self.limit_per_minute:
            return False

        # Record request
        self.request_history[client_ip].append(now)
        return True


# Global instances
api_key_manager = APIKeyManager()
cors_config_manager = CORSConfigManager()
rate_limiter = RateLimiter()
