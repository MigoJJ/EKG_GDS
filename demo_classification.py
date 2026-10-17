"""
Phase 3 Demo Script: Runs 1D-ResNet ONNX Classifier on PTB-XL records
Demonstrates multi-label diagnostic probabilities, threshold decisions, and clinical urgency categorization.
"""

from pipeline.ptbxl_loader import PTBXLLoader
from pipeline.classifier import ECGClassifier


def run_classification_demo(ecg_id: int, loader: PTBXLLoader, classifier: ECGClassifier):
    rec = loader.get_record(ecg_id, sampling_rate=100)
    result = classifier.predict(rec["signal"], return_top_k=4)

    print("=" * 75)
    print(f"ECG Record ID: {rec['ecg_id']} | Patient: {rec['age']}yo {rec['sex']}")
    print(f"Doctor Report: {rec['doctor_report']}")
    ground_truth = [k for k, v in rec['labels'].items() if v == 1]
    print(f"Ground Truth : {ground_truth if ground_truth else ['(Unclassified)']}")
    print("-" * 75)
    print(f"[Model Classification Result] (Clinical Urgency: {result['clinical_urgency']})")
    print(f"  • Positive Findings : {result['positive_findings'] if result['positive_findings'] else 'None (Under Threshold)'}")
    print("  • Top Diagnoses:")
    for cls_name, prob in result["top_findings"]:
        thresh = classifier.thresholds.get(cls_name, 0.50)
        status = "★ POSITIVE" if prob >= thresh else "negative"
        print(f"      - {cls_name:5s}: {prob * 100:5.1f}%  (Threshold: {thresh * 100:.0f}%, {status})")
    print("=" * 75)


def main():
    loader = PTBXLLoader()
    classifier = ECGClassifier()

    print("\n>>> Testing Record 9 (Normal Baseline):")
    run_classification_demo(9, loader, classifier)

    print("\n>>> Testing Record 63 (Anteroseptal Myocardial Infarction):")
    run_classification_demo(63, loader, classifier)


if __name__ == "__main__":
    main()
