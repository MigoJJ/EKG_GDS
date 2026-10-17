"""
Phase 2 Demo Script: Runs Signal Preprocessing and Feature Extraction on PTB-XL records
Displays Heart Rate, PR, QRS, QTc, Axis, ST deviations, and LVH criteria.
"""

from pipeline.ptbxl_loader import PTBXLLoader
from pipeline.feature_extractor import ECGFeatureExtractor


def inspect_record_features(ecg_id: int, loader: PTBXLLoader, extractor: ECGFeatureExtractor):
    rec = loader.get_record(ecg_id, sampling_rate=100)
    feats = extractor.extract_features(rec["signal"], patient_sex=rec["sex"], patient_age=rec["age"])

    print("=" * 75)
    print(f"ECG Record ID: {rec['ecg_id']} | Patient: {rec['age']}yo {rec['sex']}")
    print(f"Doctor Report : {rec['doctor_report']}")
    print(f"Target Labels : {[k for k, v in rec['labels'].items() if v == 1]}")
    print("-" * 75)
    print("[1. Basic Intervals & Rhythm]")
    print(f"  • Heart Rate       : {feats['heart_rate_bpm']} bpm ({'Regular' if feats['rhythm_regular'] else 'IRREGULAR'})")
    print(f"  • PR Interval      : {feats['pr_interval_ms']} ms {'(Prolonged >=200ms - 1AVB)' if feats['prolonged_pr'] else '(Normal)'}")
    print(f"  • QRS Duration     : {feats['qrs_duration_ms']} ms {'(Wide >=120ms - Bundle Branch Block)' if feats['wide_qrs'] else '(Normal)'}")
    print(f"  • QT / QTc (Bazett): {feats['qt_interval_ms']} ms / {feats['qtc_interval_ms']} ms {'(Prolonged QTc)' if feats['prolonged_qtc'] else '(Normal)'}")
    print(f"  • Frontal QRS Axis : {feats['heart_axis_deg']}° [{feats['heart_axis_category']}]")
    print("\n[2. Morphological & Voltage Findings]")
    print(f"  • Sokolow-Lyon LVH : {feats['sokolow_lyon_mv']} mV ({'POSITIVE >=3.5mV' if feats['lvh_voltage_criteria'] else 'Negative'})")
    print(f"  • STEMI Suspected  : {'*** CRITICAL ALERT (STEMI) ***' if feats['stemi_suspected'] else 'No'}")
    if feats['st_elevation_leads']:
        print(f"  • ST Elevation in  : {feats['st_elevation_leads']}")
    if feats['st_depression_leads']:
        print(f"  • ST Depression in : {feats['st_depression_leads']}")
    print("=" * 75)


def main():
    loader = PTBXLLoader()
    extractor = ECGFeatureExtractor(sampling_rate=100)
    
    print("\n>>> Testing Record 9 (Normal Baseline):")
    inspect_record_features(9, loader, extractor)
    
    print("\n>>> Testing Record 63 (Anteroseptal Myocardial Infarction):")
    inspect_record_features(63, loader, extractor)


if __name__ == "__main__":
    main()
