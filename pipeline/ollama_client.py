"""
Ollama EKG Clinical Interpretation Client Module
Builds structured clinical prompts from ECG features and deep learning probabilities,
enforces medical hallucination guardrails, and calls local Ollama models (Qwen 2.5).
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx


MANDATORY_DISCLAIMER = (
    "※ 본 판독문은 인공지능 심전도(ECG) 판독 보조 소프트웨어의 분석 결과이며, "
    "최종 진단 및 치료 방침의 결정은 반드시 담당 의사의 임상적 판단에 따라야 합니다."
)

KCD_MAP = {
    "NORM": ("Z01.8", "정상 심전도 (Normal ECG)"),
    "MI":   ("I21.0", "MI (PTB-XL 기준)"),
    "AFIB": ("I48.0", "심방세동 (Atrial Fibrillation)"),
    "PVC":  ("I49.3", "심실 조기탈분극 (Premature Ventricular Contractions)"),
    "PAC":  ("I49.1", "심방 조기탈분극 (Premature Atrial Contractions)"),
    "1AVB": ("I44.0", "1도 방실차단 (First-Degree AV Block)"),
    "RBBB": ("I45.1", "우각차단 (Right Bundle Branch Block)"),
    "LBBB": ("I45.2", "좌각차단 (Left Bundle Branch Block)"),
    "LAFB": ("I44.4", "좌전섬유속차단 (Left Anterior Fascicular Block)"),
    "LVH":  ("I51.7", "좌심실비대 (Left Ventricular Hypertrophy)"),
    "STTC": ("R94.3", "비특이적 ST-T 분절 이상 (Nonspecific ST-T Changes)"),
}


SYSTEM_PROMPT_NARRATIVE = """당신은 심장내과 전문의이자 EMR(전자의무기록) 심전도(12-Lead ECG) 판독 보조 AI입니다.
아래 제공된 [정량 생체 측정치]와 [딥러닝 모델 예측 확률]을 바탕으로 객관적이고 체계적인 한국어 임상 판독 보고서를 작성하세요.

[필수 작성 지침]
1. 허위 소견 금지 (Hallucination Guardrail): 제공된 수치와 유도 소견에 명시되지 않은 허위 증상이나 유도 이상을 임의로 날조하지 마세요.
2. 아래 4개 항목을 명확한 마크다운 헤더로 작성하세요:
   ### 1. [기본 리듬 및 심전도 간격]
   ### 2. [형태학적 및 축 소견 (ST-T, QRS, 전압)]
   ### 3. [주요 의심 진단 및 KCD 질병코드]
   ### 4. [임상 권고사항 및 응급도]
3. STEMI 의심 소견(ST 분절 상승)이 있을 경우 응급도를 최우선(EMERGENCY)으로 명시하고 즉각적인 심혈관 중재술(PCI) 준비를 권고하세요.
4. 보고서 끝에 반드시 공식 법적 고지 문구를 삽입하세요.
"""

SYSTEM_PROMPT_JSON = """You are an ECG clinical AI diagnostic engine.
Analyze the provided 12-lead ECG features and deep learning probabilities.
You must output ONLY valid JSON matching this schema:
{
  "rhythm": "string",
  "conduction": "string",
  "morphology": "string",
  "primary_diagnosis": "string",
  "kcd_code": "string",
  "urgency": "EMERGENCY" | "URGENT" | "ROUTINE",
  "clinical_recommendation": "string",
  "emr_summary_ko": "string"
}
Do not include any explanation or markdown tags outside the JSON block.
"""


class OllamaEKGClient:
    """Interfaces with local Ollama service to generate professional ECG interpretations."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:3b",
        timeout: float = 90.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def check_health(self) -> bool:
        """Verifies connection to the local Ollama server."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def build_clinical_prompt(
        self,
        features: Dict[str, Any],
        classification: Dict[str, Any],
        patient_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Assembles a grounded clinical prompt combining patient data, quantitative intervals, and model probabilities."""
        patient_info = patient_info or {}
        age_str = f"{patient_info.get('age', '미상')}세" if patient_info.get("age") else "연령 미상"
        sex_str = "남성" if patient_info.get("sex") == "M" else ("여성" if patient_info.get("sex") == "F" else "성별 미상")
        symptom = patient_info.get("symptoms", "일상 진료 심전도 검사")

        top_diagnoses_str = ", ".join([
            f"{cls_name} ({prob * 100:.1f}%)"
            for cls_name, prob in classification.get("top_findings", [])
        ])

        st_elev = features.get("st_elevation_leads", [])
        st_depr = features.get("st_depression_leads", [])
        st_summary = []
        if st_elev:
            st_summary.append(f"ST 상승 유도: {', '.join(st_elev)}")
        if st_depr:
            st_summary.append(f"ST 하강 유도: {', '.join(st_depr)}")
        st_str = " / ".join(st_summary) if st_summary else "특이 ST 분절 편위 없음"

        prompt = f"""[환자 기본 정보]
- 인적 사항: {age_str} {sex_str}
- 임상 상태/주요 증상: {symptom}

[12-Lead ECG 정량 측정치]
- 심박수 (Heart Rate): {features.get('heart_rate_bpm')} bpm ({'규칙적 리듬' if features.get('rhythm_regular') else '불규칙적 리듬'})
- PR 간격: {features.get('pr_interval_ms')} ms ({'방실차단 의심(>=200ms)' if features.get('prolonged_pr') else '정상'})
- QRS 폭: {features.get('qrs_duration_ms')} ms ({'각차단/와이드 QRS(>=120ms)' if features.get('wide_qrs') else '정상 폭'})
- QT / QTc (Bazett 보정): {features.get('qt_interval_ms')} ms / {features.get('qtc_interval_ms')} ms ({'QT 연장' if features.get('prolonged_qtc') else '정상 범위'})
- 전면 심전축 (Heart Axis): {features.get('heart_axis_deg')}도 [{features.get('heart_axis_category')}]
- Sokolow-Lyon 좌심실 비대 지수: {features.get('sokolow_lyon_mv')} mV ({'LVH 기준 충족(>=3.5mV)' if features.get('lvh_voltage_criteria') else '정상 전압'})
- ST 분절 소견: {st_str} (STEMI 의심 플래그: {'경보(ALERT)' if features.get('stemi_suspected') else '없음'})

[딥러닝 1D-ResNet 전문 모델 판정]
- 신뢰도 상위 진단군: {top_diagnoses_str}
- 모델 사전 위험도 평가: {classification.get('clinical_urgency', 'ROUTINE')}

위 수치와 확률 데이터를 종합하여 전문의 검토용 심전도 판독 보고서를 작성하세요.
"""
        return prompt

    def generate_narrative_report(
        self,
        features: Dict[str, Any],
        classification: Dict[str, Any],
        patient_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Calls Ollama to generate a formal Korean narrative EMR interpretation report."""
        user_prompt = self.build_clinical_prompt(features, classification, patient_info)

        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT_NARRATIVE,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Low temperature to suppress hallucinations
                "top_p": 0.9,
                "num_predict": 280,  # Keeps output concise and finishes in ~15-20s on CPU
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                raw_response = data.get("response", "").strip()

                # Ensure mandatory disclaimer is present
                if MANDATORY_DISCLAIMER not in raw_response:
                    raw_response += f"\n\n---\n{MANDATORY_DISCLAIMER}"

                return raw_response
        except Exception as e:
            return (
                f"### [오류] Ollama 판독 보고서 생성 실패\n"
                f"내용: {str(e)}\n\n"
                f"---\n{MANDATORY_DISCLAIMER}"
            )

    def generate_structured_json(
        self,
        features: Dict[str, Any],
        classification: Dict[str, Any],
        patient_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calls Ollama with JSON mode to produce an EMR-ready structured payload."""
        user_prompt = self.build_clinical_prompt(features, classification, patient_info)

        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT_JSON,
            "prompt": user_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                raw_json_str = data.get("response", "{}").strip()
                parsed = json.loads(raw_json_str)
                parsed["disclaimer"] = MANDATORY_DISCLAIMER
                return parsed
        except Exception as e:
            return {
                "error": str(e),
                "disclaimer": MANDATORY_DISCLAIMER
            }
