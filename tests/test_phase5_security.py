"""
Phase 5 Step 2: Security & Authentication Tests

Tests for API keys, CORS, rate limiting, and security headers.
"""

import pytest
from api.auth import APIKeyManager, CORSConfigManager, RateLimiter, APIKey


class TestAPIKeyManager:
    """Test API key authentication and management."""

    def test_validate_valid_key(self):
        """Valid API key passes validation."""
        manager = APIKeyManager()
        is_valid, api_key, error = manager.validate_key("sk_dev_test_123456789")
        assert is_valid
        assert api_key is not None
        assert error is None

    def test_validate_invalid_key(self):
        """Invalid API key fails validation."""
        manager = APIKeyManager()
        is_valid, api_key, error = manager.validate_key("sk_invalid_key")
        assert not is_valid
        assert api_key is None
        assert error is not None

    def test_validate_inactive_key(self):
        """Inactive API key fails validation."""
        manager = APIKeyManager()
        is_valid, api_key, error = manager.validate_key("sk_test_inactive")
        assert not is_valid
        assert "inactive" in error.lower()

    def test_add_custom_key(self):
        """Add custom API key."""
        manager = APIKeyManager()
        new_key = APIKey(
            key_id="custom_001",
            hospital_name="Custom Hospital",
            key_secret="sk_custom_test",
            rate_limit_per_minute=50,
        )
        manager.add_key(new_key)

        is_valid, api_key, _ = manager.validate_key("sk_custom_test")
        assert is_valid
        assert api_key.hospital_name == "Custom Hospital"

    def test_rate_limit_per_minute(self):
        """API key respects per-minute rate limit."""
        manager = APIKeyManager()
        key_secret = "sk_dev_test_123456789"
        endpoint = "/api/v1/ecg/analyze"

        # First request should be allowed
        is_allowed, error = manager.check_rate_limit(key_secret, endpoint)
        assert is_allowed

        # Simulate many requests (modify request list directly for testing)
        api_key = manager.keys[key_secret]
        key_id = api_key.key_id

        # Fill up minute quota
        import time
        now = time.time()
        for i in range(api_key.rate_limit_per_minute):
            manager.request_count[key_id][endpoint].append(now)

        # Next request should fail
        is_allowed, error = manager.check_rate_limit(key_secret, endpoint)
        assert not is_allowed
        assert "rate limit" in error.lower()

    def test_get_remaining_quota(self):
        """Get remaining quota for API key."""
        manager = APIKeyManager()
        key_secret = "sk_dev_test_123456789"

        quota = manager.get_remaining_quota(key_secret)
        assert "remaining_per_minute" in quota
        assert "remaining_per_day" in quota
        assert "limit_per_minute" in quota
        assert "limit_per_day" in quota
        assert quota["remaining_per_minute"] <= quota["limit_per_minute"]

    def test_list_keys_excludes_secrets(self):
        """List keys should not include secrets."""
        manager = APIKeyManager()
        keys_list = manager.list_keys()

        assert len(keys_list) > 0
        for key_info in keys_list:
            assert "key_id" in key_info
            assert "hospital_name" in key_info
            assert "key_secret" not in key_info  # Secret should not be exposed!
            assert "active" in key_info


class TestCORSConfigManager:
    """Test CORS configuration."""

    def test_check_allowed_origin(self):
        """Check if origin is in whitelist."""
        manager = CORSConfigManager()

        assert manager.is_origin_allowed("http://127.0.0.1:3000")
        assert manager.is_origin_allowed("https://emr.seoul-medical.kr")
        assert not manager.is_origin_allowed("https://malicious.com")

    def test_add_origin(self):
        """Add new allowed origin."""
        manager = CORSConfigManager()
        new_origin = "https://emr.newhospital.kr"

        assert not manager.is_origin_allowed(new_origin)
        manager.add_origin(new_origin)
        assert manager.is_origin_allowed(new_origin)

    def test_remove_origin(self):
        """Remove allowed origin."""
        manager = CORSConfigManager()
        origin = "http://127.0.0.1:3000"

        assert manager.is_origin_allowed(origin)
        manager.remove_origin(origin)
        assert not manager.is_origin_allowed(origin)

    def test_get_config_dict(self):
        """Get CORS config for FastAPI."""
        manager = CORSConfigManager()
        config = manager.get_config_dict()

        assert "allow_origins" in config
        assert "allow_credentials" in config
        assert "allow_methods" in config
        assert "allow_headers" in config
        assert isinstance(config["allow_origins"], list)
        assert len(config["allow_origins"]) > 0

    def test_reject_empty_origin(self):
        """Reject empty/None origin."""
        manager = CORSConfigManager()

        assert not manager.is_origin_allowed("")
        assert not manager.is_origin_allowed(None)


class TestRateLimiter:
    """Test IP-based rate limiting (backup)."""

    def test_allow_requests_within_limit(self):
        """Allow requests below rate limit."""
        limiter = RateLimiter()
        client_ip = "10.0.0.1"

        for _ in range(50):
            assert limiter.is_allowed(client_ip)

    def test_reject_requests_exceeding_limit(self):
        """Reject requests exceeding rate limit."""
        limiter = RateLimiter()
        client_ip = "10.0.0.2"

        # Fill up quota
        for _ in range(limiter.limit_per_minute):
            limiter.is_allowed(client_ip)

        # Next request should be rejected
        assert not limiter.is_allowed(client_ip)

    def test_different_ips_have_separate_limits(self):
        """Different IPs have independent rate limits."""
        limiter = RateLimiter()
        ip1 = "10.0.0.1"
        ip2 = "10.0.0.2"

        # Fill IP1 quota
        for _ in range(limiter.limit_per_minute):
            limiter.is_allowed(ip1)

        # IP1 should be rejected
        assert not limiter.is_allowed(ip1)

        # IP2 should still be allowed
        assert limiter.is_allowed(ip2)


class TestAPIKeyToDict:
    """Test API key serialization."""

    def test_key_to_dict_excludes_secret(self):
        """Key serialization should exclude secret."""
        key = APIKey(
            key_id="test_001",
            hospital_name="Test Hospital",
            key_secret="sk_test_secret_12345",
        )

        key_dict = key.to_dict()
        assert "key_id" in key_dict
        assert "hospital_name" in key_dict
        assert "active" in key_dict
        assert "rate_limits" in key_dict
        assert "key_secret" not in key_dict

    def test_key_metadata_preserved(self):
        """Key metadata is preserved in serialization."""
        key = APIKey(
            key_id="prod_001",
            hospital_name="Seoul Medical Center",
            key_secret="sk_prod_real",
            rate_limit_per_minute=100,
            rate_limit_per_day=2000,
            active=True,
        )

        key_dict = key.to_dict()
        assert key_dict["hospital_name"] == "Seoul Medical Center"
        assert key_dict["rate_limits"]["per_minute"] == 100
        assert key_dict["rate_limits"]["per_day"] == 2000
        assert key_dict["active"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
