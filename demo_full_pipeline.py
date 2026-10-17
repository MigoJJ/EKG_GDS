"""
End-to-End Pipeline Demonstration:
Takes a raw 12-lead ECG from PTB-XL -> Preprocesses & Extracts Quantitative Intervals ->
Runs 1D-ResNet ONNX Classifier -> Generates Ollama Clinical Narrative & Structured EMR JSON.
"""

from pipeline.ptbxl_loader import PTBXLLoader
from pipeline.feature_extractor import ECGFeatureExtractor
from pipeline.classifier import ECGClassifier
from pipeline.ollama_client import OllamaEKGClient


def run_pipeline(ecg_id: int):
    print("=" * 80)
    print(f"  [AI 12-LEAD EKG READING ASSISTANT] - Processing Record ID: {ecg_id}")
    print("=" * 80)

    loader = PTBXLLoader()
    extractor = ECGFeatureExtractor(sampling_rate=100)
    classifier = ECGClassifier()
    ollama = OllamaEKGClient(model_name="qwen2.5:3b")

    # 1. Load Raw ECG
    rec = loader.get_record(ecg_id, sampling_rate=100)
    print(f"Patient Demographics: Age {rec['age']}, Sex {rec['sex']}")
    print(f"Doctor Ground-Truth : {rec['doctor_report']}")
    print(f"Annotated Labels    : {[k for k, v in rec['labels'].items() if v == 1]}")
    print("-" * 80)

    # 2. Extract Features
    print("[Step 1: Signal Preprocessing & Feature Extraction]")
    features = extractor.extract_features(rec["signal"], patient_sex=rec["sex"], patient_age=rec["age"])
    print(f"  • Heart Rate  : {features['heart_rate_bpm']} bpm (Rhythm: {'Regular' if features['rhythm_regular'] else 'IRREGULAR'})")
    print(f"  • PR / QRS    : {features['pr_interval_ms']} ms / {features['qrs_duration_ms']} ms")
    print(f"  • QT / QTc    : {features['qt_interval_ms']} ms / {features['qtc_interval_ms']} ms")
    print(f"  • QRS Axis    : {features['heart_axis_deg']}° [{features['heart_axis_category']}]")
    print(f"  • Sokolow LVH : {features['sokolow_lyon_mv']} mV (Criteria: {features['lvh_voltage_criteria']})")
    print(f"  • STEMI Alert : {features['stemi_suspected']} (Elevation in: {features['st_elevation_leads']})")
    print("-" * 80)

    # 3. Model Classification
    print("[Step 2: 1D-ResNet Deep Learning Classification (ONNX CPU)]")
    classification = classifier.predict(rec["signal"], return_top_k=3)
    print(f"  • Urgency Tier: {classification['clinical_urgency']}")
    print(f"  • Top Findings: {classification['top_findings']}")
    print("-" * 80)

    # 4. Ollama Clinical Narrative Report
    print("[Step 3: Ollama Clinical Interpretation Report (Qwen 2.5)]")
    patient_info = {"age": rec["age"], "sex": rec["sex"], "symptoms": "응급실 내원 흉통 환자"}
    report = ollama.generate_narrative_report(features, classification, patient_info)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    # Test on Record 63 (Anteroseptal Myocardial Infarction)
    run_pipeline(63)
