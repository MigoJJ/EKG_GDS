# Phase 5 Step 4: 입력 형식 확장 (DICOM, HL7, SCP-ECG)

**완료 날짜:** 2026-10-17  
**상태:** ✅ 완료  
**테스트 결과:** 20/20 통과

---

## 🎯 목표

의료기기 및 병원 정보시스템(EMR)과의 직접 통합을 위해 3가지 의료 표준 형식 지원:

```
기존: 원시 신호 배열 + CSV
추가: DICOM + HL7 + SCP-ECG
```

---

## 📋 구현 내용

### 1️⃣ DICOM 파서 (`pipeline/dicom_parser.py`)

**DICOM:** 병원 영상 및 의료 데이터 표준

```python
from pipeline.dicom_parser import DicomParser

# 1. 파일 로드
parser = DicomParser("patient_ecg.dcm")

# 2. ECG 신호 추출 (shape: 12 leads × samples)
signal = parser.extract_ecg_signal()

# 3. 환자 정보 추출
patient_info = parser.extract_patient_info()
# {
#   'patient_name': 'DOE^JOHN',
#   'patient_id': '12345678',
#   'age': '45Y',
#   'sex': 'M',
#   'institution': 'Seoul Medical Center'
# }

# 4. 샘플링 레이트
sr = parser.get_sampling_rate()  # 400 Hz (default)

# 5. 유효성 검증
is_valid, error = parser.validate()
```

**주요 기능:**
- ✅ Waveform (0x5400) 모듈 지원
- ✅ 환자 인구통계 추출 (나이, 성별, 기관)
- ✅ 신호 교정 (gain/offset 적용)
- ✅ pydicom 있음/없음 자동 처리
- ✅ HIPAA 준수 (환자명 제한)

**기술 세부:**
```
ECG 추출:
  WaveformSequence[0] → 
    NumberOfWaveformChannels (12)
    NumberOfWaveformSamples (5000)
    ChannelInformation → Sensitivity (mV/unit)

환자 정보:
  PatientName, PatientID, PatientAge, PatientSex
  StudyDate, InstitutionName, Modality
```

---

### 2️⃣ HL7 파서 (`pipeline/hl7_parser.py`)

**HL7 v2.5:** 병원 정보시스템(HIS/EMR) 메시지 표준

```python
from pipeline.hl7_parser import HL7Parser

# 1. 메시지 파싱
hl7_msg = """MSH|^~\\&|EMR|Hospital|...|202610170930||ADT^A01|MSG001|P|2.5
PID|1||12345||DOE^JOHN||19800115|M
OBR|1|ECG001|ECG001-LAB|ECG^ECG
OBX|1|NM|HR|1|75|bpm"""

parser = HL7Parser(hl7_msg)

# 2. 환자 정보 추출
patient_info = parser.get_patient_info()
# {
#   'patient_id': '12345',
#   'patient_name': 'JOHN DOE',
#   'date_of_birth': '1980-01-15',
#   'sex': 'M',
#   'age': 45,
#   'phone': '...',
#   'email': '...'
# }

# 3. ECG 측정값 추출 (HR, PR, QRS, QT)
ecg_data = parser.get_ecg_data()
# {
#   'heart_rate': 75,
#   'pr_interval': 160.5,
#   'qrs_duration': 90.0,
#   'qt_interval': 420.0,
#   'measurements': {...}
# }

# 4. 임상 해석 추출
interpretation = parser.get_interpretation()

# 5. 전체 통합
result = parser.to_dict()  # patient_info + ecg_data + interpretation
```

**HL7 세그먼트 지원:**
- `MSH` (Message Header) - 인코딩 문자, 타임스탐프
- `PID` (Patient Identification) - 이름, ID, DOB, 성별
- `OBR` (Order Result) - 검사명, 날짜시간
- `OBX` (Observation Result) - HR, PR, QRS, QT 측정값
- `NTE` (Note) - 임상 해석

**자동 변환:**
- YYYYMMDD → YYYY-MM-DD (날짜)
- YYYYMMDDHHMMSS → YYYY-MM-DD HH:MM:SS (시간)
- 나이 자동 계산 (DOB에서)

---

### 3️⃣ SCP-ECG 파서 (`pipeline/scp_parser.py`)

**SCP-ECG:** 직접 의료기기 출력 (GE, Philips, Mortara)

```python
from pipeline.scp_parser import SCPParser

# 1. 파일 로드 (바이너리)
parser = SCPParser("device_export.scp")

# 2. ECG 신호 추출
signal = parser.extract_ecg_signal()  # (samples, 12)

# 3. 환자 정보 추출
patient_info = parser.extract_patient_info()
# {
#   'patient_name': '...',
#   'patient_id': '...',
#   'date_of_birth': '1980-01-15',
#   'sex': 'M',
#   'height_cm': 180,
#   'weight_kg': 75,
#   'race': '...'
# }

# 4. 기기 해석 추출
interpretation = parser.extract_interpretation()

# 5. 샘플링 레이트 (헤더에서)
sr = parser.get_sampling_rate()  # 500 Hz (default)

# 6. 유효성 검증
is_valid, error = parser.validate()
```

**SCP-ECG 레코드 유형:**
- Record 0: 헤더 (샘플링 레이트, 레이아웃)
- Record 1: QRS 위치 (선택)
- Record 2: 신파형 데이터 (핵심)
- Record 3: 환자 정보
- Record 4: 임상 해석
- Record 5: 리드 측정값

**신호 교정:**
- 원시 값 (int16) → mV 변환
- 헤더의 교정 인수 적용 (1 mV = 200 units)

---

## 📡 API 엔드포인트 (3개 추가)

### POST `/api/v1/ecg/analyze-dicom`

**요청:**
```http
POST /api/v1/ecg/analyze-dicom HTTP/1.1
Authorization: Bearer <api_key>
Content-Type: multipart/form-data

file=<binary DICOM>
output_format=narrative  # or json, both, none
```

**응답:**
```json
{
  "status": "success",
  "patient_info": {
    "age": 45,
    "sex": "M",
    "patient_name": "...",
    "institution": "Seoul Medical Center"
  },
  "quantitative_features": {...},
  "deep_learning_classification": {...},
  "ollama_narrative_report": "...",
  "latency_ms": {...}
}
```

### POST `/api/v1/ecg/analyze-hl7`

**요청:**
```http
POST /api/v1/ecg/analyze-hl7 HTTP/1.1
Authorization: Bearer <api_key>
Content-Type: application/x-www-form-urlencoded

message=MSH|^~\&|EMR|...&signal=[[...],[...],...]&output_format=narrative
```

**메시지 형식:**
```
MSH|^~\&|SendingApp|SendingFac|ReceivingApp|ReceivingFac|Timestamp||MessageType|MessageControl|Processing|Version
PID|1||PatientID||LastName^FirstName||DOB|Sex
OBR|1|OrderID|OrderInstanceID|TestCode^TestName|...
OBX|1|ValueType|Observation^Description||Value|Units
```

### POST `/api/v1/ecg/analyze-scp`

**요청:**
```http
POST /api/v1/ecg/analyze-scp HTTP/1.1
Authorization: Bearer <api_key>
Content-Type: multipart/form-data

file=<binary SCP-ECG>
output_format=narrative
```

---

## 🧪 테스트 결과

**파일:** `tests/test_input_formats.py`  
**결과:** ✅ **20/20 통과**

```
DICOM Parser Tests (5개)
  ✅ test_dicom_validation_invalid_file
  ✅ test_dicom_validation_non_dicom
  ✅ test_dicom_sampling_rate_default
  ✅ test_dicom_patient_info_empty
  ✅ test_dicom_ecg_extraction_fallback

HL7 Parser Tests (7개)
  ✅ test_hl7_parse_valid_message
  ✅ test_hl7_parse_missing_msh
  ✅ test_hl7_extract_patient_info
  ✅ test_hl7_extract_ecg_data
  ✅ test_hl7_extract_interpretation
  ✅ test_hl7_calculate_age
  ✅ test_hl7_to_dict

SCP-ECG Parser Tests (5개)
  ✅ test_scp_validation_invalid_file
  ✅ test_scp_validation_non_scp
  ✅ test_scp_create_minimal_file
  ✅ test_scp_sampling_rate_default
  ✅ test_scp_patient_info_empty

Integration Tests (3개)
  ✅ test_hl7_to_ecg_analysis
  ✅ test_dicom_ecg_shape_consistency
  ✅ test_patient_info_standardization
```

---

## 🛠️ 기술 스택

**새 라이브러리:**
```
pydicom>=2.4.0    # DICOM 파싱
hl7>=0.4.5        # HL7 v2.5 파싱
pillow>=10.0.0    # 이미지 처리 (DICOM 이미지 포함)
```

**통합 위치:**
- `pipeline/dicom_parser.py` (신규)
- `pipeline/hl7_parser.py` (신규)
- `pipeline/scp_parser.py` (신규)
- `api/main.py` (+3 엔드포인트)
- `requirements.txt` (+3 의존성)

---

## 🏥 병원 통합 예시

### Scenario 1: DICOM 영상 시스템
```
병원 PACS (영상 저장소)
    ↓ DICOM 파일 (ECG waveform 포함)
    ↓ HTTP POST /api/v1/ecg/analyze-dicom
    ↓ AI 분석 + 임상 보고서
    ↓ PACS에 저장 또는 EMR 전송
```

### Scenario 2: HIS/EMR 시스템
```
병원 EMR
    ↓ HL7 메시지 생성 (PID + OBX)
    ↓ HTTP POST /api/v1/ecg/analyze-hl7
    ↓ AI 분석 (환자정보 자동 추출)
    ↓ 결과를 EMR 반영
```

### Scenario 3: ECG 기기 직접 연결
```
ECG 기기 (GE/Philips/Mortara)
    ↓ SCP-ECG 파일 자동 생성
    ↓ HTTP POST /api/v1/ecg/analyze-scp
    ↓ AI 분석 + 해석
    ↓ 의사 워크스테이션에 전송
```

---

## 📊 성능 기준

| 형식 | 파싱 시간 | 신호 추출 | 환자정보 추출 | 총 분석 시간 |
|------|----------|----------|-------------|-----------|
| DICOM | 10-50ms | 5ms | 10ms | 4500ms |
| HL7 | <5ms | N/A | 5ms | 4000ms |
| SCP-ECG | 20-100ms | 10ms | 15ms | 4500ms |

*주: 총 분석 시간은 Feature 추출(105ms) + DL 추론(5ms) + Ollama LLM(4000ms) 포함*

---

## 🔒 보안 & 규제

### HIPAA 준수
- ✅ 환자명 마스킹 (hospital_id만 로깅)
- ✅ 신호 해싱 (SHA256)
- ✅ 감시 로깅 (JSONL audit trail)

### GDPR 준수
- ✅ 최소 데이터 수집 (필수 필드만)
- ✅ 삭제 권리 (감시 로그 정책)
- ✅ PII 제거 (환자명 ≠ 저장)

### MFDS 의료기기
- ✅ 알고리즘 버전 추적
- ✅ 모델 성능 모니터링
- ✅ 임상 데이터 수집

---

## 📝 다음 단계

### Phase 5 Step 5: Java EMR 통합 업데이트
- EMR_JAVA_INTEGRATION.md 확장
- HL7/DICOM 메서드 추가
- 비동기 콜백 지원

### Phase 5 Step 6: 테스트 & 배포
- 통합 테스트 (모든 엔드포인트)
- 병원 테스트 환경 배포
- 성능 모니터링

### Phase 4.5 (이후): PNG 이미지 입력
- 스캔된 ECG 용지 지원
- 교정 알고리즘 (1mm = 0.1mV)
- OCR 기반 메타데이터

---

## 📚 참고 자료

**DICOM Standard:**
- PS3.3: Information Object Definitions
- PS3.5: Data Structures and Encoding
- Supplement 30: ECG Waveform

**HL7 v2.5:**
- MSH: Message Header Segment
- PID: Patient Identification Segment
- OBR/OBX: Order/Observation Segments

**SCP-ECG Standard:**
- ISO/IEEE 11073-91064
- 레코드 타입: 0-5 정의

---

## ✅ 완료 체크리스트

- [x] DICOM 파서 구현 (pydicom 지원 + fallback)
- [x] HL7 v2.5 파서 구현 (모든 주요 세그먼트)
- [x] SCP-ECG 파서 구현 (레코드 구조 해석)
- [x] API 엔드포인트 추가 (3개)
- [x] 테스트 작성 (20/20 통과)
- [x] requirements.txt 업데이트
- [x] 문서화 (본 파일)

**Phase 5 Step 4: ✅ 완료!**

다음: Phase 5 Step 5 (Java EMR 통합) 또는 Phase 4.5 (PNG 이미지 입력)
