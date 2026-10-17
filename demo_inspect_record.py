"""
Demo inspection script: Loads a record from the PTB-XL test set and displays
patient demographics, signal attributes, mapped diagnostic labels, and original doctor report.
"""

from pipeline.ptbxl_loader import PTBXLLoader, DIAGNOSTIC_CLASSES


def main():
    loader = PTBXLLoader()
    print("=" * 70)
    print("PTB-XL ECG Dataset Loader - Record Inspection Demo")
    print("=" * 70)

    # Let's inspect a few records from Test Fold 10
    test_df = loader.get_split("test")
    sample_ids = test_df.index[:3].tolist()
    
    for ecg_id in sample_ids:
        rec = loader.get_record(ecg_id, sampling_rate=100)
        print(f"\n[Record ID: {rec['ecg_id']}] (Fold {rec['strat_fold']})")
        print(f"  Patient ID  : {rec['patient_id']} | Age: {rec['age']} | Sex: {rec['sex']}")
        print(f"  Signal      : Shape {rec['signal'].shape} ({rec['meta']['fs']} Hz, 12 leads)")
        print(f"  Leads       : {rec['meta']['sig_name']}")
        print(f"  Signal Range: Min {rec['signal'].min():.3f} mV, Max {rec['signal'].max():.3f} mV")
        print(f"  Doctor Report: {rec['doctor_report']}")
        print(f"  Raw SCP Codes: {rec['scp_codes']}")
        
        pos_labels = [cls for cls, val in rec['labels'].items() if val == 1]
        print(f"  11-Class Diagnosis: {pos_labels if pos_labels else ['(Unclassified)']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
