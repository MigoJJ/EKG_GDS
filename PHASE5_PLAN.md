# Phase 5: EMR 연동·운영 계획

**상태:** 계획 수립 완료  
**기간:** 1–2주 (Step 1부터 Step 6)  
**우선순위:** 🔴 최우선  
**복잡도:** 높음 (비동기, 보안, 규제)

---

## 핵심 목표 5가지

### 1️⃣ 비동기 처리
```
동기 (Phase 4): POST /analyze → 4초 대기
비동기 (Phase 5): POST /analyze-async → job_id 즉시 반환
                  GET /jobs/{job_id} → 진행률 폴링
```
**효과:** 의사 대기 시간 0으로 (백그라운드 실행)

### 2️⃣ 보안 강화
```
❌ → ✅
CORS "*"        API 키 + 화이트리스트
인증 없음       HTTPBearer 토큰
모든 IP         127.0.0.1 또는 내부망
로깅 없음       모든 요청 감시 기록
속도제한 없음   100 req/min
HTTP 만         HTTPS (선택)
```

### 3️⃣ 감시 로깅 (규제 증거)
```
모든 요청 기록:
  ├─ 입력 (신호 해시, 모델 버전)
  ├─ 결과 (진단명, KCD, 응급도)
  └─ 의사 피드백 (동의/불동의, 수정)

규제 활용:
  ✓ 알고리즘 이력 추적
  ✓ 성능 모니터링
  ✓ 임상 검증 데이터
```

### 4️⃣ 입력 형식 확장 (선택)
```
현재: Numpy array, CSV
추가: DICOM, HL7, SCP-ECG
순서: DICOM → HL7 → SCP (필요시)
```

### 5️⃣ Java EMR 통합
```
비동기 호출 추가:
  - analyzeEcgAsync()
  - waitForResult()
  - subscribeToProgress()
```

---

## Step 분해 (6단계, 10–12일)

### Step 1: 비동기 아키텍처 (2–3일)
```
새 파일:
  ├─ models/job_queue.py (JobQueue 클래스)
  └─ pipeline/task_executor.py (백그라운드 실행)

수정:
  └─ api/main.py
     ├─ POST /api/v1/ecg/analyze-async (신규)
     ├─ GET /api/v1/jobs/{job_id} (신규)
     └─ DELETE /api/v1/jobs/{job_id} (신규)
```

### Step 2: 보안 및 인증 (2일)
```
새 파일:
  └─ api/auth.py (API 키, 속도 제한, CORS)

수정:
  ├─ api/main.py (미들웨어 추가)
  └─ config/security.py (설정)
```

### Step 3: 감시 로깅 (2–3일)
```
새 파일:
  ├─ models/audit_logger.py
  └─ api/audit_routes.py

신규 디렉토리:
  └─ logs/audit/ (JSONL 로그)
```

### Step 4: 입력 형식 확장 (3–5일, 선택)
```
새 파일:
  ├─ pipeline/dicom_parser.py (DICOM)
  ├─ pipeline/hl7_parser.py (HL7)
  └─ pipeline/scp_parser.py (SCP, 선택)

수정:
  └─ api/main.py (+3 엔드포인트)
```

### Step 5: Java 통합 업데이트 (2일)
```
수정:
  ├─ EMR_JAVA_INTEGRATION.md
  └─ EkgAiService.java (확장)
```

### Step 6: 테스트 & 배포 (2–3일)
```
새 파일:
  ├─ tests/test_async_api.py
  ├─ tests/test_security.py
  ├─ tests/test_audit_logging.py
  └─ tests/test_input_formats.py
```

---

## 기술 선택

### 비동기 작업 큐
**권장:** asyncio + in-memory queue
- 간단함 (복잡도 낮음)
- 충분함 (병원 규모)
- 확장 가능 (나중에 Redis로 업그레이드)

### 로그 저장소
**권장:** JSONL + PostgreSQL 옵션
- 즉시: JSONL (파일 기반, 간단)
- 나중: PostgreSQL (규제 필요시)

---

## 규제 & 안전성

### 의료 데이터 보호
```
✓ 신호 원본 저장 금지 (해시만)
✓ 환자 PII 마스킹 (hospital_id로 대체)
✓ 타임스탬프 범위화 (일별)
✓ 로그 암호화 (프로덕션)
✓ 접근 제어 (관리자만)
```

### 규제 준수
```
식약처 의료기기:
  ✓ 알고리즘 버전 이력
  ✓ 성능 모니터링 (감시 로그)
  ✓ 오류 추적 (위양성/위음성)
  ✓ 임상 데이터 (피드백)
```

---

## 성공 기준

```
✅ 비동기 분석 동작
✅ job_id 즉시 반환 (< 200ms)
✅ 폴링 메커니즘 (진행률 0–100%)
✅ API 키 검증 (모든 요청)
✅ CORS 화이트리스트
✅ 감시 로그 저장 (JSONL)
✅ 피드백 수집 API
✅ 모든 테스트 통과 (40+ 테스트)
✅ 병원 내부망 바인딩
✅ Java 통합 예제
```

---

## 일정 상세

| 주 | Step | 작업 | 일수 |
|----|------|------|------|
| 1 | 1 | 비동기 API + 폴링 | 3 |
| 1–2 | 2 | API 키, CORS, 속도제한 | 2 |
| 2 | 3 | 감시 로그 (JSONL) | 3 |
| 2 | 4 | DICOM 파싱 (선택) | 3 |
| 2 | 5 | Java 통합 문서 | 2 |
| 2 | 6 | 테스트 & 배포 | 3 |
| **총** | | | **10–12** |

---

## 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|-----|
| 비동기 복잡도 | 개발 지연 | asyncio 기초부터 (2–3일) |
| 메모리 누수 | 서버 다운 | 정기 정리 스크립트 |
| 로그 용량 증가 | 디스크 부족 | 일별 로테이션 |
| 규제 미준수 | 인허가 거부 | PIMS 표준 엄격히 |

---

## Phase 5 완료 후

```
운영 준비 완료:
  ✅ 병원 EMR과 통합 가능
  ✅ 의료 데이터 보호
  ✅ 규제 증거 수집 중
  ✅ 확장 기반 구축
  
다음:
  Phase 6: 규제 대비 & 임상 검증
```

---

**문서 상태:** 준비 완료  
**다음 작업:** Step 1 (비동기 API) 시작

