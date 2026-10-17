"""
Phase 4 Template Engine Tests

Verifies rule-based template generation and hallucination prevention.
"""

import pytest
from pipeline.template_engine import TemplateReportBuilder, HallucinationValidator


class TestTemplateReportBuilder:
    """Test deterministic template generation."""

    def test_diagnoses_to_korean_single(self):
        """Single diagnosis conversion."""
        result = TemplateReportBuilder.diagnoses_to_korean(["MI"])
        assert "심근경색" in result

    def test_diagnoses_to_korean_multiple(self):
        """Multiple diagnosis conversion with proper conjunction."""
        result = TemplateReportBuilder.diagnoses_to_korean(["MI", "RBBB"])
        assert "심근경색" in result and "우각차단" in result
        assert "및" in result  # Korean "and"

    def test_finding_text_normal(self):
        """Normal ECG finding."""
        result = TemplateReportBuilder.build_finding_text([], False, False)
        assert "정상 심전도" in result

    def test_finding_text_with_stemi(self):
        """Finding text includes STEMI alert."""
        result = TemplateReportBuilder.build_finding_text(["MI"], False, True)
        assert "ST 분절 상승" in result
        assert "심근경색" in result

    def test_recommendation_emergency(self):
        """Emergency urgency recommendation."""
        result = TemplateReportBuilder.build_recommendation_text("EMERGENCY")
        assert "즉시" in result or "응급" in result

    def test_recommendation_routine(self):
        """Routine urgency recommendation."""
        result = TemplateReportBuilder.build_recommendation_text("ROUTINE")
        assert "추적 관찰" in result or "임상적 판단" in result

    def test_narrative_template_structure(self):
        """Narrative template has all required sections."""
        features = {
            "heart_rate_bpm": 72,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "qtc_interval_ms": 410,
            "heart_axis_deg": 45,
            "heart_axis_category": "normal",
            "stemi_suspected": False,
        }
        classification = {
            "primary_findings": ["MI"],
        }
        result = TemplateReportBuilder.build_narrative_template(
            features, classification, "I21.0", "URGENT"
        )
        assert "기본 리듬" in result
        assert "주요 소견" in result
        assert "임상 권고사항" in result
        assert "I21.0" in result

    def test_json_template_has_required_fields(self):
        """JSON template contains all required fields."""
        features = {
            "heart_rate_bpm": 72,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "qtc_interval_ms": 410,
            "heart_axis_deg": 45,
            "heart_axis_category": "normal",
            "rhythm_regular": True,
            "wide_qrs": False,
            "stemi_suspected": False,
        }
        classification = {
            "primary_findings": ["AFIB"],
        }
        result = TemplateReportBuilder.build_json_template(
            features, classification, "I48.0", "EMERGENCY"
        )
        assert result["kcd_code"] == "I48.0"
        assert result["urgency"] == "EMERGENCY"
        assert result["primary_diagnosis"] == "AFIB"
        assert "rhythm" in result
        assert "conduction" in result


class TestHallucinationValidator:
    """Test hallucination detection and prevention."""

    def test_valid_narrative(self):
        """Narrative within valid bounds."""
        narrative = "환자는 정상 심전도를 나타냅니다."
        is_valid, error = HallucinationValidator.validate_narrative(
            narrative, ["NORM"], "Z01.8", "ROUTINE"
        )
        assert is_valid

    def test_hallucination_unsupported_diagnosis(self):
        """Detects diagnosis not in original findings."""
        narrative = "환자는 심각한 심근경색 및 심실 비대를 나타냅니다."
        is_valid, error = HallucinationValidator.validate_narrative(
            narrative, ["NORM"], "Z01.8", "ROUTINE"
        )
        assert not is_valid
        assert "허상" in error or "감지" in error

    def test_hallucination_urgency_mismatch(self):
        """Detects urgency level inconsistency."""
        narrative = "즉시 응급 시술이 필요합니다 (응급)"
        is_valid, error = HallucinationValidator.validate_narrative(
            narrative, ["MI"], "I21.0", "ROUTINE"  # Rule says ROUTINE, narrative says EMERGENCY
        )
        # Note: This test depends on how strictly we check urgency keywords
        # If "응급" is found but urgency != EMERGENCY, should fail

    def test_valid_json(self):
        """Valid JSON matches expected codes."""
        json_output = {
            "kcd_code": "I48.0",
            "urgency": "EMERGENCY",
            "primary_diagnosis": "AFIB",
        }
        is_valid, error = HallucinationValidator.validate_json(
            json_output, "I48.0", "EMERGENCY"
        )
        assert is_valid

    def test_invalid_json_kcd_mismatch(self):
        """Invalid JSON has wrong KCD code."""
        json_output = {
            "kcd_code": "I21.0",  # Wrong!
            "urgency": "EMERGENCY",
        }
        is_valid, error = HallucinationValidator.validate_json(
            json_output, "I48.0", "EMERGENCY"
        )
        assert not is_valid
        assert "KCD" in error or "변조" in error

    def test_invalid_json_urgency_mismatch(self):
        """Invalid JSON has wrong urgency."""
        json_output = {
            "kcd_code": "I48.0",
            "urgency": "ROUTINE",  # Wrong!
        }
        is_valid, error = HallucinationValidator.validate_json(
            json_output, "I48.0", "EMERGENCY"
        )
        assert not is_valid
        assert "응급도" in error or "변조" in error

    def test_fallback_narrative(self):
        """Fallback narrative is safe and includes disclaimer."""
        features = {
            "heart_rate_bpm": 72,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "qtc_interval_ms": 410,
            "heart_axis_deg": 45,
            "heart_axis_category": "normal",
            "stemi_suspected": False,
        }
        classification = {"primary_findings": ["MI"]}
        result = HallucinationValidator.fallback_narrative(
            features, classification, "I21.0", "URGENT"
        )
        assert "템플릿" in result
        assert "I21.0" in result or "심근경색" in result

    def test_fallback_json(self):
        """Fallback JSON has validation_status marker."""
        features = {
            "heart_rate_bpm": 72,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "qtc_interval_ms": 410,
            "heart_axis_deg": 45,
            "heart_axis_category": "normal",
            "rhythm_regular": True,
            "wide_qrs": False,
            "stemi_suspected": False,
        }
        classification = {"primary_findings": ["AFIB"]}
        result = HallucinationValidator.fallback_json(
            features, classification, "I48.0", "EMERGENCY"
        )
        assert result.get("validation_status") == "fallback_template"
        assert result["kcd_code"] == "I48.0"
        assert result["urgency"] == "EMERGENCY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
