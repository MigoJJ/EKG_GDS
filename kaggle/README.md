# Kaggle GPU 학습 가이드 (Phase 3)

PTB-XL은 공개 데이터셋이므로 Kaggle에 올려도 됩니다.
**의원 실제 환자 심전도는 절대 Kaggle에 올리지 마세요** — 로컬 CPU에서만 학습합니다.

## 1. 업로드 파일 만들기 (로컬)

```bash
cd ~/ekg-assistant
./kaggle/make_bundle.sh --data
```

- `dist/ekg_code.zip` — 학습 코드 (약 32KB)
- `dist/ptbxl_100hz.zip` — PTB-XL 100Hz 기록 + CSV 2개 (약 600MB)

Kaggle에 PTB-XL 공개 데이터셋이 이미 있으면 `--data` 없이 코드만 만들고 그 데이터셋을 추가해도 됩니다.
노트북은 `/kaggle/input` 아래에서 `ptbxl_database.csv`를 자동으로 찾습니다(옆에 `records100/` 폴더가 있어야 함).

## 2. Kaggle 데이터셋 만들기

kaggle.com → **Datasets → New Dataset**
1. `ekg_code.zip` 업로드 → 이름 예: `ekg-code` (코드를 고칠 때마다 **New Version**으로 갱신)
2. `ptbxl_100hz.zip` 업로드 → 이름 예: `ptbxl-100hz` (최초 1회)

## 3. 노트북 실행

1. **Code → New Notebook → File → Import Notebook** → `kaggle/ekg_phase3_train.ipynb`
2. 오른쪽 패널 **Add Input**으로 위 두 데이터셋 추가
3. **Settings**: Accelerator = **GPU T4 x2** 또는 **P100**, Internet = **On**
   (GPU·인터넷 사용에는 휴대폰 인증된 계정이 필요합니다)
4. **Run All** — 에폭마다 진행률(%), 남은 시간, Val AUPRC가 한 줄씩 출력됩니다

긴 학습은 **Save Version → Save & Run All (Commit)**으로 실행하면 브라우저를 닫아도 계속 돌아갑니다.

## 4. 결과 가져오기 (로컬)

노트북 **Output** 패널에서 `phase3_output.zip` 다운로드 후:

```bash
mkdir -p ~/ekg-assistant/models/candidates/kaggle_run1
unzip ~/Downloads/phase3_output.zip -d ~/ekg-assistant/models/candidates/kaggle_run1
```

포함 파일: `ecg_resnet18_11class.onnx`, `best_ecg_resnet18.pt`, `clinical_cutoffs.json`(Fold 9 재조정),
`metrics_fold10.json`, `history.json`, `run_info.json`(SHA-256 포함)

운영 모델(`models/ecg_resnet18_11class.onnx`)은 자동으로 바뀌지 않습니다.
기존 모델과 Fold 10 성능을 비교한 뒤 교체 여부를 결정합니다 (5단계).
