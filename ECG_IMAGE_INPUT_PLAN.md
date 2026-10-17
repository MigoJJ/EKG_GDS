# ECG PNG 이미지 입력 계획

**목표:** 의료기기 PNG/JPEG 파일 → ECG 신호 추출 → 기존 프로그램 활용  
**우선순위:** 중간 (Phase 5 이후)  
**예상 소요:** 8–12일  
**복잡도:** 높음 (기기별 교정, 정확성 요구)

---

## 현재 입력 지원

```
✅ Numpy array        shape=(1000, 12) [샘플, 유도]
✅ PTB-XL database    레코드 ID로 로드
✅ CSV/JSON 신호     구조화된 데이터

❌ PNG/JPEG 이미지   ← 새로 추가할 부분
❌ DICOM ECG         
❌ HL7 aECG (XML)    
```

---

## 기술적 도전

### 문제 1: 이미지 해석
```
의료기기 PNG 파일:
  ├─ 정보: 그리드, 곡선, 텍스트 (환자명, HR, 기기명 등)
  ├─ 포맷: GE, Philips, Mortara 등 제각각
  └─ 신호: 픽셀 좌표 → 전압(mV), 시간(s)으로 변환 필요
```

### 문제 2: 교정 (Calibration)
```
표준 ECG 용지:
  종축 (전압):  1mm = 0.1 mV
  횡축 (시간):  1mm = 40ms (또는 25mm/s 기준)
  
현실:
  - 이미지 해상도 다름 (DPI에 따라)
  - 기기별 격자 간격 다름
  - 줌/축소 가능
  
→ 그리드 자동 감지 + 교정 필수
```

### 문제 3: 정확성 요구
```
일반 이미지: ±10% 오차 허용 가능
의료기기: ±5% 이내 (임상 진단 영향)

특히 ST 구간, QT 간격 측정:
  - 0.1mm 차이 = 4ms 시간 오차
  - 0.5mm 차이 = 0.05mV 전압 오차
```

---

## 구현 전략 (3단계)

### Phase 4.5-1: 기본 이미지 처리

**목표:** PNG → 픽셀 좌표 추출

**모듈:** `pipeline/ecg_image_parser.py`

```python
class ECGImageParser:
    """ECG 이미지에서 신호 추출."""
    
    def parse_ecg_image(image_path: str, device_type: str = "auto"):
        """
        PNG/JPEG 파일 읽기
        
        Args:
            image_path: 파일 경로
            device_type: "ge", "philips", "mortara", "auto"
        
        Returns:
            {
                "signal_12lead": np.ndarray (1000, 12),
                "sampling_rate": 100 또는 500,
                "device_info": "GE CardioServ v5.0" 등,
                "metadata": {...}
            }
        """
    
    def detect_grid(image) -> Tuple[float, float, float, float]:
        """
        그리드 감지 → 픽셀/mm, 위치 반환
        
        Returns: (pixels_per_mm_x, pixels_per_mm_y, grid_x, grid_y)
        """
    
    def extract_waveform_curves(image, grid_specs) -> Dict[str, np.ndarray]:
        """
        12개 유도 곡선 추출
        
        Returns: {"I": [...], "II": [...], ..., "V6": [...]}
        """
    
    def calibrate_voltage(pixels, pixels_per_mm) -> np.ndarray:
        """
        픽셀 좌표 → 전압(mV)
        
        1 mm = 0.1 mV 기준
        """
```

**기술:**
- OpenCV: 이미지 처리, 컨투어 검출
- scikit-image: 세련된 필터링
- numpy: 신호 변환

**단계:**
1. 이미지 로드
2. 그레이스케일 변환
3. 이진화 (Otsu's threshold)
4. 그리드 라인 감지 (Hough transform)
5. 곡선 추적 (contour detection)
6. 12개 채널 분류
7. 교정

---

### Phase 4.5-2: 기기별 지원

**목표:** GE, Philips, Mortara 표준 포맷 지원

**구조:**
```python
class DeviceCalibration(ABC):
    """의료기기별 교정 기준."""
    
    @abstractmethod
    def detect_grid_params(image) -> dict:
        """기기별 그리드 감지"""
    
    @abstractmethod
    def extract_metadata(image) -> dict:
        """환자명, 기기명, 시간 등"""

class GECardioServCalibration(DeviceCalibration):
    """GE CardioServ 시리즈."""
    pass

class PhilipsCalibration(DeviceCalibration):
    """Philips PageWriter 등."""
    pass

class MortaraCalibration(DeviceCalibration):
    """Mortara ELI 시리즈."""
    pass
```

**확인 사항:**
- GE 문서: 기본 격자 간격, DPI
- Philips 문서: 동일
- Mortara 문서: 동일

---

### Phase 4.5-3: API 엔드포인트

**새 엔드포인트:**

```http
POST /api/v1/ecg/analyze-from-image
Content-Type: multipart/form-data
```

**요청:**
```json
{
  "file": <binary PNG/JPEG>,
  "device_type": "ge|philips|mortara|auto",
  "output_format": "both",
  "patient_info": {
    "age": 65,
    "sex": "M"
  }
}
```

**응답:**
```json
{
  "status": "success",
  "image_info": {
    "device_detected": "GE CardioServ v5.0",
    "confidence": 0.95,
    "grid_params": {...}
  },
  "signal_extracted": {
    "shape": [1000, 12],
    "sampling_rate": 100,
    "quality_score": 0.92
  },
  "quantitative_features": { ... },
  "classification": { ... },
  "ollama_report": { ... }
}
```

**구현:**
```python
@app.post("/api/v1/ecg/analyze-from-image", tags=["Analysis"])
async def analyze_from_image(
    file: UploadFile,
    device_type: str = "auto",
    output_format: str = "both"
):
    # 임시 파일 저장
    image_path = save_temp_file(file)
    
    # 이미지 → 신호
    parser = ECGImageParser()
    result = parser.parse_ecg_image(image_path, device_type)
    
    # 기존 파이프라인으로
    sig = result["signal_12lead"]
    features = extractor.extract(sig)
    classification = classifier.predict(sig)
    report = ollama.generate_combined_report(features, classification)
    
    # 응답
    return {
        "status": "success",
        "image_info": result.get("metadata", {}),
        "quantitative_features": features,
        "classification": classification,
        "ollama_report": report
    }
```

---

## 테스트 계획

### 테스트 데이터 준비

```
tests/fixtures/ecg_images/
├── ge_normal_001.png           (정상)
├── ge_afib_001.png             (심방세동)
├── ge_mi_001.png               (심근경색)
├── philips_normal_001.png
├── philips_stemi_001.png
├── mortara_lvh_001.png
└── README.md                   (각 파일의 메타데이터)
```

**파일 생성 방법:**
1. 실제 의료기기에서 출력한 PNG 수집
2. 또는 공개 데이터셋 이미지 (PTB-XL)

### 테스트 케이스

```python
class TestECGImageParser:
    
    def test_grid_detection_ge():
        """GE 이미지 그리드 감지"""
    
    def test_waveform_extraction():
        """곡선 추출 정확도 (±5mm 이내)"""
    
    def test_calibration_voltage():
        """mV 변환 정확성"""
    
    def test_12_channel_classification():
        """12개 유도 자동 분류"""
    
    def test_device_auto_detection():
        """기기 자동 인식"""
    
    def test_metadata_extraction():
        """환자명, HR, 기기명 추출"""
    
    def test_noise_robustness():
        """손상된/노이즈 이미지 처리"""
    
    def test_api_endpoint():
        """REST 엔드포인트"""
    
    # 총 20+ 테스트
```

---

## 일정

| 주간 | 항목 | 일수 |
|------|------|------|
| 1 | 기술 수집, 샘플 파일 입수 | 2 |
| 2 | 기본 이미지 처리 프로토타입 | 3 |
| 3 | GE, Philips 기기 지원 | 3 |
| 4 | 교정 알고리즘 정밀화 | 2 |
| 5 | API 통합, 테스트 | 2 |
| **총** | | **12일** |

---

## 위험 및 대응

| 위험 | 확률 | 영향 | 대응 |
|------|------|------|------|
| 기기 포맷 차이 | 높음 | 파서 복잡화 | 초기 2개 기기만 지원 |
| 교정 정확도 부족 | 중간 | 진단 오류 | 의료팀 검증, 여유도 포함 |
| 이미지 손상 | 중간 | 추출 실패 | 자동 감지, 사용자 경고 |
| 성능 (느린 처리) | 낮음 | UX 불량 | 캐싱, 다중처리 |

---

## 의료 안전성 고려

### 규제 요구사항

```
FDA 510(k) 유사:
  ✓ 알고리즘 검증 (벤치마크 데이터)
  ✓ 정확도 증명 (의료기기 표준과 비교)
  ✓ 교정 추적 (어떤 기준 사용했는지 문서화)
  ✓ 오류 처리 (손상 이미지 감지)

한국 MFDS:
  ✓ 의료기기 분류 (Class II 예상)
  ✓ 임상 데이터 (적어도 50–100 케이스)
  ✓ 위험 관리 (False Positive/Negative)
```

### 교정 증명

```
문서화:
  □ 표준 ECG 용지 기준 (1mm = 0.1mV)
  □ 각 기기별 사양서 인용
  □ 벤치마크 데이터 (오차율 < 5%)
  □ 검증 프로토콜 (임상의 판정과 비교)
```

---

## 현재 권장 우선순위

```
🔴 Phase 5 (EMR 연동)        — 우선, 핵심
🟡 Phase 4.5 (PNG 입력)      — 차순, 추가
🟢 Phase 6 (규제 대비)       — 3순, 지속
```

**이유:**
- EMR 통합이 운영 필수
- PNG는 편의 기능
- 규제는 전반적으로 진행

---

## 시작하기 (준비)

### 즉시 확보할 것

1. **샘플 파일 3–5개**
   - GE CardioServ: 2개
   - Philips PageWriter: 2개
   - 기타: 1개

2. **기기 사양 문서**
   - GE: 매뉴얼에서 그리드 정보
   - Philips: 동일
   - 기술 지원팀 연락

3. **벤치마크 데이터**
   - 원본 신호 (CSV 등)
   - PNG 재출력 이미지 비교

### 프로토타입 (첫 3일)

```python
# 최소한의 증명 개념
import cv2
import numpy as np
from PIL import Image

def simple_ecg_parser(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 이진화
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 컨투어 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 12개 유도로 분류
    channels = classify_contours(contours)
    # 교정
    signal = calibrate(channels)
    return signal
```

---

## 결론

**ECG PNG 입력은 유용하지만 복잡한 기능**

```
✅ 장점:
  - 의료기기와 직접 호환
  - 병원 워크플로우 개선
  - 진료 기록 저장 가능

⚠️ 도전:
  - 기기별 차이 많음
  - 정확도 요구 높음
  - 규제 문서화 필요

📅 추천:
  - Phase 5 (EMR) 완료 후
  - Phase 4.5로 추가 구현
  - Phase 6과 병행 진행
```

---

**작성:** 2026-10-17  
**상태:** 계획 문서 (구현 전)  
**다음:** Phase 5 (EMR 연동) 시작

