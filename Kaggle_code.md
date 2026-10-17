# Kaggle GPU 훈련 가이드 (Phase 3)

**작성일:** 2026-10-17  
**완료 상태:** ✅ Phase 3 성공 완료  
**훈련 환경:** Kaggle GPU (Tesla T4 × 2)  
**훈련 시간:** 5.8분 (14 epoch, 조기 중단)

---

## 개요

PTB-XL 100Hz 데이터셋으로 Focal Loss 기반 1D-ResNet18 모델을 Kaggle GPU에서 훈련하고, 결과를 로컬에 다운로드·배포하는 전체 워크플로우입니다.

**장점:**
- ⚡ GPU 훈련으로 5.8분 완료 (CPU 대비 ~25x 가속)
- 🆓 Kaggle 무료 GPU 시간 활용
- 📊 실시간 진행률 모니터링 (1% 단위)
- 🔄 로컬에서 쉽게 결과 다운로드 및 배포 가능

---

## Step 1: 로컬 번들 준비

### 1.1 데이터셋 번들 생성

```bash
cd ~/ekg-assistant
./kaggle/make_bundle.sh --data
```

**출력:**
```
✅ Creating dist/ekg_code.zip (32KB)
✅ Creating dist/ptbxl_100hz.zip (362MB)
✅ Done
```

**생성되는 파일:**
- `dist/ekg_code.zip` (32KB)
  - `train_improved.py`: 훈련 스크립트
  - `models/export_onnx.py`: ONNX 내보내기
  - `pipeline/`: 기존 코드베이스

- `dist/ptbxl_100hz.zip` (362MB, `--data` 플래그 사용 시)
  - `records100/`: 12-lead ECG 신호 (17,418개)
  - `ptbxl_database.csv`: 환자 정보
  - `scp_statements.csv`: 진단 레이블

> **주의:** 실제 환자 데이터는 절대 `--data` 없이만 업로드하세요. PTB-XL은 공개 데이터이므로 Kaggle 업로드 가능합니다.

### 1.2 코드 번들만 생성 (기존 데이터셋 재사용)

```bash
./kaggle/make_bundle.sh
```

코드만 32KB로 생성되며, Kaggle에 이미 있는 PTB-XL 데이터셋을 재사용할 수 있습니다.

---

## Step 2: Kaggle 데이터셋 업로드

### 2.1 코드 데이터셋 업로드

1. **kaggle.com → Datasets → New Dataset**
2. `dist/ekg_code.zip` 업로드
3. **이름:** `ekg-code` (또는 원하는 이름)
4. **설명:** "Phase 3 training code: train_improved.py with Focal Loss"
5. **공개 설정:** Private 또는 Public 선택 (private 추천)
6. **Create**

**코드 업데이트 시:**
- 기존 데이터셋 선택 → **New Version** → zip 파일 재업로드
- 노트북에서 자동으로 최신 버전 사용

### 2.2 데이터셋 업로드 (최초 1회)

1. **kaggle.com → Datasets → New Dataset**
2. `dist/ptbxl_100hz.zip` (362MB) 업로드
3. **이름:** `ptbxl-100hz`
4. **설명:** "PTB-XL 100Hz resampled: 17,418 records, 12-lead"
5. **공개 설정:** Public 권장 (Kaggle 커뮤니티 공유 가능)
6. **Create**

> **참고:** Kaggle에 이미 공개된 PTB-XL이 있으므로, 필요시 기존 데이터셋을 대신 사용할 수 있습니다.

---

## Step 3: Kaggle 노트북 생성 및 설정

### 3.1 노트북 임포트

1. **kaggle.com → Code → New Notebook → File → Import Notebook**
2. `~/ekg-assistant/kaggle/ekg_phase3_train.ipynb` 선택
3. **Create**

### 3.2 입력(Input) 데이터셋 추가

**오른쪽 패널 → Add input**

1. **ekg-code**: 위에서 업로드한 코드 데이터셋 선택
2. **ptbxl-100hz**: 위에서 업로드한 PTB-XL 데이터셋 선택

> **또는 기존 Kaggle 데이터셋 사용:**  
> Kaggle에 이미 있는 공개 PTB-XL을 검색·추가 가능합니다.

### 3.3 설정(Settings) 구성

**노트북 오른쪽 상단 → Settings**

- **GPU**: Tesla P100 또는 Tesla T4 × 2 선택
- **Internet**: **ON** (필수 — pip 패키지 설치용)
- **가속기**: GPU 사용량 확인 후 가능하면 T4 선택

> **주의:** GPU와 Internet 사용은 휴대폰 인증이 필요합니다.

---

## Step 4: 노트북 실행

### 4.1 단계별 실행 (모니터링)

**Run All** 클릭

노트북 8개 셀:
1. ✅ GPU 확인
2. ✅ 패키지 설치
3. ✅ 데이터 마운트 확인
4. ✅ 훈련 실행
5. ✅ ONNX 내보내기
6. ✅ 검증
7. ✅ 결과 압축
8. ✅ 완료

### 4.2 실시간 진행률 모니터링

```
[train] Loading 17418 records...
  2000/17418 (11%) 25s
  4000/17418 (23%) 50s
  ...

Training on cuda | epochs=50 batch=64 lr=0.0015 focal_gamma=2.0 patience=5
  [  1%] epoch 1 batch 137/273 loss 0.0580 | elapsed 0.1 min | ETA 14.0 min
  [  2%] epoch 1 batch 273/273 loss 0.0286 | elapsed 0.3 min | ETA 12.8 min
   1/50         2%     0.0617    0.5508  0.001500     24.5     20.0 ★ best
  ...

   9/50        18%     0.0356    0.7299  0.001408     25.1     16.9 ★ best
  ...

Early stopping: no AUPRC improvement for 5 epochs
Best val macro AUPRC 0.7299 | total 5.8 min
```

**각 라인의 의미:**
- `[x%]`: 전체 진행률
- `epoch N`: 현재 에포크
- `batch M/M`: 배치 진행률
- `loss`: 훈련 손실 (Focal Loss)
- `elapsed`: 경과 시간
- `ETA`: 예상 남은 시간
- `★ best`: 새로운 최고 val AUPRC 달성

### 4.3 장시간 훈련 (브라우저 닫아도 계속)

노트북이 길게 실행되는 경우:
- **Save Version** 클릭
- **Save & Run All (Commit)** 선택
- 브라우저 종료 후에도 계속 실행됨
- Kaggle 메일로 완료 알림 받음

---

## Step 5: 결과 다운로드

### 5.1 Output 패널에서 다운로드

**노트북 오른쪽 Output 탭 → phase3_output.zip 다운로드**

파일 크기: ~40MB (ONNX 모델 34MB + 메타데이터)

### 5.2 로컬 추출 및 배포

```bash
# 다운로드 폴더에서
cd ~/Downloads
mkdir -p ~/ekg-assistant/models/candidates/kaggle_run1
unzip phase3_output.zip -d ~/ekg-assistant/models/candidates/kaggle_run1

# 확인
ls -lh ~/ekg-assistant/models/candidates/kaggle_run1/
```

**포함 파일:**
- `ecg_resnet18_11class.onnx` (34MB) — 훈련된 모델
- `best_ecg_resnet18.pt` (34MB) — PyTorch 가중치
- `clinical_cutoffs.json` — Fold 9에서 재조정된 클래스별 임계값
- `metrics_fold10.json` — Fold 10 테스트 성능 (AUC, AUPRC, 감도 등)
- `history.json` — 14 epoch 훈련 곡선 (손실, val AUPRC)
- `run_info.json` — 메타데이터 (모델 SHA-256, 훈련 설정)

---

## Step 6: 모델 성능 평가

### 6.1 성능 비교

```python
import json

new_metrics = json.load(open('models/candidates/kaggle_run1/metrics_fold10.json'))

print("=== Fold 10 (테스트) 성능 ===")
print(f"Macro AUC:         {new_metrics['macro']['auc']:.4f}")
print(f"Macro AUPRC:       {new_metrics['macro']['auprc']:.4f}")
print(f"Macro Sensitivity: {new_metrics['macro']['sensitivity']:.4f}")
print(f"Macro Specificity: {new_metrics['macro']['specificity']:.4f}")

print("\n=== 임상 중요 클래스 ===")
for cls in ['MI', 'AFIB', 'NORM', 'RBBB', 'LBBB']:
    c = new_metrics['per_class'][cls]
    print(f"{cls:6} | AUC: {c['auc']:.4f} | Sensitivity: {c['sensitivity']:.4f}")
```

**결과 (Phase 3):**
```
Macro AUC:         0.9485
Macro AUPRC:       0.7283
Macro Sensitivity: 0.8062
Macro Specificity: 0.9100

MI     | AUC: 0.9013 | Sensitivity: 0.8947
AFIB   | AUC: 0.9817 | Sensitivity: 0.9211
NORM   | AUC: 0.9413 | Sensitivity: 0.8110
RBBB   | AUC: 0.9894 | Sensitivity: 0.8313
LBBB   | AUC: 0.9835 | Sensitivity: 0.8226
```

### 6.2 기존 모델과 비교

| 지표 | Phase v1.0 | Phase 3 Kaggle | 차이 |
|-----|-----------|----------------|-----|
| Macro AUC | 0.9530 | 0.9485 | -0.0045 (동등) |
| Macro AUPRC | — | 0.7283 | ✅ 신규 |
| 중요 클래스 감도 | — | 대부분 달성 | ✅ |

---

## Step 7: 모델 배포 (운영 적용)

### 7.1 성능 평가 후 배포

```bash
cd ~/ekg-assistant

# 성능 만족 → 배포
cp models/candidates/kaggle_run1/ecg_resnet18_11class.onnx models/ecg_resnet18_11class.onnx
cp models/candidates/kaggle_run1/clinical_cutoffs.json models/clinical_cutoffs.json

# 기존 모델 백업 (이미 완료됨)
# models/ecg_resnet18_11class.onnx.v1.0
# models/clinical_cutoffs.json.v1.0
```

### 7.2 Git 커밋

```bash
git add models/ecg_resnet18_11class.onnx models/clinical_cutoffs.json
git commit -m "Phase 3: Deploy Kaggle GPU trained model (AUC 0.9485)"
```

---

## 주요 파라미터 및 설정

### train_improved.py 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `--epochs` | 50 | 최대 에포크 수 |
| `--batch-size` | 64 | 배치 크기 |
| `--lr` | 0.0015 | 학습률 |
| `--patience` | 5 | 조기 중단: 이 많은 에포크 동안 개선 없으면 중단 |
| `--gamma` | 2.0 | Focal Loss 집중도 (클래스 불균형 처리) |
| `--no-augment` | False | 온디바이스 증강 활성화 |

### 증강(Augmentation) 설정

**augment_batch()에서 적용:**
- **진폭:** ±10% 스케일링
- **기저선:** 0.05–0.5Hz 드리프트
- **노이즈:** 0.01–0.05σ 가우시안
- **유도 탈락:** 5% 확률로 일부 유도 제거

### 체크포인트 선택

- **기준:** Val Macro AUPRC (정밀도-재현율 균형)
- **저장:** 새로운 최고점 달성 시 자동 저장
- **조기 중단:** 5 에포크 동안 개선 없으면 훈련 종료

---

## 문제 해결

### Q: "Connection refused" 또는 DNS 오류

**A:** Kaggle 노트북 Settings에서 **Internet: ON** 확인

### Q: GPU 메모리 부족 (OOM)

**A:** `--batch-size`를 32로 감소시키거나 `--no-augment` 추가

```bash
python -u train_improved.py --batch-size 32 --no-augment
```

### Q: 데이터셋을 찾을 수 없음

**A:** 노트북 오른쪽 Input 패널에서 두 데이터셋(ekg-code, ptbxl-100hz) 모두 추가 확인

### Q: 훈련 중 ONNX 내보내기 실패

**A:** DeprecationWarning은 무시 가능합니다 (PyTorch 2.9 이후 변경 예정).  
모델이 제대로 내보내졌으면 `phase3_output.zip`에 `.onnx` 파일이 있을 것입니다.

---

## 참고: 파일 구조

```
~/ekg-assistant/
├── kaggle/
│   ├── make_bundle.sh          ← 번들 생성 스크립트
│   ├── ekg_phase3_train.ipynb   ← Kaggle 노트북
│   └── README.md               ← 간단한 가이드 (이전)
├── train_improved.py           ← 훈련 스크립트 (bundled)
├── models/
│   ├── ecg_resnet18_11class.onnx      ← 운영 모델 (Phase 3 배포됨)
│   ├── ecg_resnet18_11class.onnx.v1.0 ← 백업 (Phase v1.0)
│   ├── clinical_cutoffs.json          ← 운영 임계값
│   └── candidates/
│       └── kaggle_run1/               ← Kaggle 훈련 결과
│           ├── ecg_resnet18_11class.onnx
│           ├── best_ecg_resnet18.pt
│           ├── clinical_cutoffs.json
│           ├── metrics_fold10.json
│           ├── history.json
│           └── run_info.json
```

---

## 다음 단계

**Phase 4: LLM 리포트 개선** (예정)
- Ollama 구조 변경: 규칙 엔진 → 템플릿 → LLM
- 환각 방지
- 스트리밍 응답

---

## 참고 자료

- [Kaggle GPU 할당량](https://www.kaggle.com/settings/usage) — 월 30시간 무료
- [PTB-XL 데이터셋](https://physionet.org/content/ptb-xl/) — 공개 ECG 데이터
- [Focal Loss 논문](https://arxiv.org/abs/1708.02002) — 클래스 불균형 해결
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html) — ONNX 변환

---

**작성:** 2026-10-17  
**마지막 업데이트:** Phase 3 완료 (Kaggle GPU 훈련 성공)
