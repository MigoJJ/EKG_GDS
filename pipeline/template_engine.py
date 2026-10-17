"""
Template Report Builder for Phase 4 LLM Improvement

Generates rule-based template text for ECG findings, recommendations, and JSON structure.
This separates deterministic content (findings, codes, urgency) from LLM's text-polishing role,
eliminating hallucination risks.

Workflow: Rule Engine → Template → LLM (text refinement only) → Validation → Response
"""

from typing import Dict, List, Any, Optional, Tuple
from pipeline.stemi_kcd_engine import get_primary_kcd_code


DIAGNOSIS_KOREAN = {
    "NORM": "정상 심전도",
    "MI": "심근경색",
    "AFIB": "심방세동",
    "PVC": "심실 조기탈분극",
    "PAC": "심방 조기탈분극",
    "1AVB": "1도 방실차단",
    "RBBB": "우각차단",
    "LBBB": "좌각차단",
    "LAFB": "좌전섬유속차단",
    "LVH": "좌심실비대",
    "STTC": "비특이적 ST-T 이상",
}

URGENCY_KOREAN = {
    "EMERGENCY": "응급",
    "URGENT": "긴급",
    "ROUTINE": "일상",
}

URGENCY_RECOMMENDATION = {
    "EMERGENCY": "즉시 심혈관 중재술(PCI) 준비. 의료 협력팀과 긴급 소통 필수.",
    "URGENT": "긴급 심혈관 내과 상담 권고. 선별 검사(혈액 생화학, troponin) 의뢰.",
    "ROUTINE": "담당 의사의 임상적 판단에 따라 추적 관찰 및 추가 검사 결정.",
}


class TemplateReportBuilder:
    """Builds deterministic template text for ECG findings, preventing LLM hallucination."""

    @staticmethod
    def diagnoses_to_korean(findings: List[str]) -> str:
        """Convert diagnosis codes (e.g., ['MI', 'RBBB']) to Korean text."""
        korean_names = [DIAGNOSIS_KOREAN.get(f, f) for f in findings]
        if len(korean_names) == 1:
            return korean_names[0]
        elif len(korean_names) == 2:
            return f"{korean_names[0]} 및 {korean_names[1]}"
        else:
            return ", ".join(korean_names[:-1]) + f", {korean_names[-1]}"

    @staticmethod
    def build_finding_text(primary_findings: List[str], wide_qrs: bool, stemi_suspected: bool) -> str:
        """
        Generate finding statement from rule engine outputs.

        Example:
            ['MI', 'RBBB'] + stemi=True → "ST 분절 상승 소견의 심근경색 및 우각차단"
        """
        parts = []

        if stemi_suspected:
            parts.append("ST 분절 상승 소견의")

        if primary_findings:
            parts.append(TemplateReportBuilder.diagnoses_to_korean(primary_findings))
        else:
            parts.append("정상 심전도")

        finding_text = " ".join(parts)
        return f"환자의 심전도는 {finding_text}를 나타냅니다."

    @staticmethod
    def build_interval_text(features: Dict[str, Any]) -> str:
        """Build text describing intervals (HR, PR, QRS, QTc, Axis)."""
        hr = features.get("heart_rate_bpm", "—")
        pr_ms = features.get("pr_interval_ms", "—")
        qrs_ms = features.get("qrs_duration_ms", "—")
        qtc_ms = features.get("qtc_interval_ms", "—")
        axis_deg = features.get("heart_axis_deg", "—")
        axis_cat = features.get("heart_axis_category", "—")

        return (
            f"기본 리듬은 심박수 {hr}bpm, "
            f"PR 간격 {pr_ms}ms, "
            f"QRS 폭 {qrs_ms}ms, "
            f"QTc {qtc_ms}ms, "
            f"심전축 {axis_deg}도({axis_cat})입니다."
        )

    @staticmethod
    def build_recommendation_text(urgency: str) -> str:
        """Convert urgency level to actionable recommendation."""
        return URGENCY_RECOMMENDATION.get(
            urgency,
            "담당 의사의 임상적 판단에 따라 추적 관찰하시기 바랍니다."
        )

    @staticmethod
    def build_narrative_template(
        features: Dict[str, Any],
        classification: Dict[str, Any],
        kcd_code: str,
        urgency: str,
    ) -> str:
        """
        Assemble a complete narrative template (before LLM refinement).

        Structure:
          1. Basic rhythm & intervals
          2. Key findings
          3. Recommendation
          4. Mandatory disclaimer
        """
        primary_findings = classification.get("primary_findings", [])
        stemi_suspected = features.get("stemi_suspected", False)
        wide_qrs = features.get("wide_qrs", False)

        sections = [
            "## 심전도 판독 보고서\n",
            "### 1. 기본 리듬 및 간격\n",
            TemplateReportBuilder.build_interval_text(features),
            "\n\n### 2. 주요 소견\n",
            TemplateReportBuilder.build_finding_text(primary_findings, wide_qrs, stemi_suspected),
            "\n\n### 3. 진단 코드 및 응급도\n",
            f"KCD-10 코드: {kcd_code}  \n응급도: {URGENCY_KOREAN.get(urgency, urgency)}\n",
            "\n### 4. 임상 권고사항\n",
            TemplateReportBuilder.build_recommendation_text(urgency),
        ]

        return "".join(sections)

    @staticmethod
    def build_json_template(
        features: Dict[str, Any],
        classification: Dict[str, Any],
        kcd_code: str,
        urgency: str,
    ) -> Dict[str, Any]:
        """
        Build template JSON structure (before LLM enrichment).
        LLM can refine descriptions but must not change codes or urgency.
        """
        primary_findings = classification.get("primary_findings", [])
        finding_text = TemplateReportBuilder.build_finding_text(
            primary_findings,
            features.get("wide_qrs", False),
            features.get("stemi_suspected", False),
        )

        return {
            "rhythm": f"HR {features.get('heart_rate_bpm', '—')} bpm, "
                      f"{'규칙' if features.get('rhythm_regular', True) else '불규칙'}",
            "conduction": (
                "각차단" if features.get("wide_qrs", False) else "정상 전도"
            ),
            "morphology": finding_text,
            "primary_diagnosis": ", ".join(primary_findings) if primary_findings else "NORM",
            "kcd_code": kcd_code,
            "urgency": urgency,
            "clinical_recommendation": TemplateReportBuilder.build_recommendation_text(urgency),
            "confidence_note": "본 판독은 AI 보조 진단이며, 최종 진단은 담당 의사의 임상적 판단에 따릅니다.",
        }


class HallucinationValidator:
    """Detects and prevents LLM hallucinations by comparing output to original rules."""

    @staticmethod
    def validate_narrative(
        narrative: str,
        primary_findings: List[str],
        kcd_code: str,
        urgency: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if narrative contains unsupported diagnosis or urgency claims.

        Returns:
            (is_valid, error_message)
        """
        # List of diagnoses mentioned in original findings
        valid_diagnoses = set(primary_findings)

        # Check for mention of diagnoses NOT in original findings
        for diagnosis_key, diagnosis_name in DIAGNOSIS_KOREAN.items():
            if diagnosis_key not in valid_diagnoses and diagnosis_name in narrative:
                # ⚠️ LLM mentioned a diagnosis that wasn't in the findings
                return False, f"허상 진단 감지: {diagnosis_name} (원본에 없음)"

        # Check for urgency consistency
        urgency_mentions = [
            (URGENCY_KOREAN.get(u), u)
            for u in ["EMERGENCY", "URGENT", "ROUTINE"]
            if u in narrative
        ]
        if urgency_mentions and urgency_mentions[0][1] != urgency:
            return False, f"응급도 불일치: {urgency_mentions[0][1]} (규칙: {urgency})"

        # Check for specific false claims
        false_claims = [
            ("고도의 심근경색", "hallucination keyword"),
            ("생명 위험", "unsubstantiated claim"),
        ]
        for claim, reason in false_claims:
            if claim in narrative:
                return False, f"허위 주장 감지: {claim}"

        return True, None

    @staticmethod
    def validate_json(
        parsed_json: Dict[str, Any],
        expected_kcd_code: str,
        expected_urgency: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if JSON structure matches expected codes and urgency.
        """
        # KCD code must match rule engine's decision
        if parsed_json.get("kcd_code") != expected_kcd_code:
            return False, f"KCD 코드 변조: {parsed_json.get('kcd_code')} != {expected_kcd_code}"

        # Urgency must match rule engine's decision
        if parsed_json.get("urgency") != expected_urgency:
            return False, f"응급도 변조: {parsed_json.get('urgency')} != {expected_urgency}"

        return True, None

    @staticmethod
    def fallback_narrative(
        features: Dict[str, Any],
        classification: Dict[str, Any],
        kcd_code: str,
        urgency: str,
    ) -> str:
        """
        Return safe template text when LLM output fails validation.
        This guarantees a clinically correct response even if LLM hallucinates.
        """
        template = TemplateReportBuilder.build_narrative_template(
            features, classification, kcd_code, urgency
        )
        fallback_note = (
            "\n\n---\n"
            "⚠️ **[주의] AI 판독 검증 오류로 템플릿 응답**  \n"
            "LLM 생성 텍스트에 일관성 오류가 감지되어 규칙 기반 템플릿으로 대체되었습니다. "
            "담당 의사의 최종 검증이 필수입니다."
        )
        return template + fallback_note

    @staticmethod
    def fallback_json(
        features: Dict[str, Any],
        classification: Dict[str, Any],
        kcd_code: str,
        urgency: str,
    ) -> Dict[str, Any]:
        """Return safe JSON template when LLM output fails validation."""
        json_template = TemplateReportBuilder.build_json_template(
            features, classification, kcd_code, urgency
        )
        json_template["validation_status"] = "fallback_template"
        json_template["validation_note"] = "LLM hallucination detected; using template."
        return json_template
