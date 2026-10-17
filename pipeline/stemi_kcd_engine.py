"""
STEMI Detection and KCD Code Assignment Engine

Implements Fourth Universal Definition of MI ST-elevation criteria:
- Age/sex-specific thresholds: Men >=40 and <=40, Women
- V2-V3: >=0.2 mV (men >=40), >=0.25 mV (men <40), >=0.15 mV (women)
- All other leads: >=0.1 mV
- Elevation must occur in >=2 anatomically contiguous leads
- Produces KCD-10 diagnosis codes based on STEMI location and presentation

Reference: Fourth Universal Definition of MI (2018), ESC/ACCF guidelines.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np


# Lead-to-anatomical region mapping for contiguity checking
LEAD_GROUPS = {
    "anterior": ["V1", "V2", "V3", "V4"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "inferior": ["II", "III", "aVF"],
    "posterolateral": ["V5", "V6"],
}

# Reverse mapping: which groups does each lead belong to?
LEAD_TO_GROUPS = {}
for region, leads in LEAD_GROUPS.items():
    for lead in leads:
        if lead not in LEAD_TO_GROUPS:
            LEAD_TO_GROUPS[lead] = []
        LEAD_TO_GROUPS[lead].append(region)


def get_ste_threshold(
    lead: str, patient_sex: str = "M", patient_age: Optional[float] = None
) -> float:
    """
    Returns ST-elevation threshold (mV) per Fourth Universal Definition of MI.

    Args:
        lead: lead name (e.g., "V2", "II")
        patient_sex: 'M' or 'F'
        patient_age: age in years (used for men: >=40 vs <40)

    Returns:
        threshold in mV
    """
    if patient_sex == "F":
        # Women: V2-V3 >=0.15 mV, others >=0.1 mV
        return 0.15 if lead in ["V2", "V3"] else 0.10
    elif patient_sex == "M":
        if lead in ["V2", "V3"]:
            # Men >=40: >=0.2 mV; Men <40: >=0.25 mV
            if patient_age is not None and patient_age < 40:
                return 0.25
            else:
                return 0.20
        else:
            # All other leads: >=0.1 mV
            return 0.10
    else:
        # Fallback
        return 0.10


def detect_stemi_region(
    st_deviations: Dict[str, float],
    elevated_leads: List[str],
    patient_sex: str = "M",
    patient_age: Optional[float] = None,
) -> Tuple[bool, Optional[str], Dict[str, float]]:
    """
    Determines if STEMI criteria are met and identifies the region.

    Args:
        st_deviations: dict of {lead: ST deviation in mV}
        elevated_leads: list of leads with ST elevation above threshold
        patient_sex: 'M' or 'F'
        patient_age: optional age

    Returns:
        (stemi_suspected: bool, region: str or None, metrics: dict)
        where region in ['anterior', 'inferior', 'lateral', 'posterior'] or None
    """
    metrics = {
        "elevated_lead_count": len(elevated_leads),
        "contiguous_groups_involved": [],
        "has_reciprocal_depression": False,
    }

    if len(elevated_leads) < 2:
        return False, None, metrics

    # Check contiguity: all elevated leads must share at least one anatomical region
    regions_per_lead = [LEAD_TO_GROUPS.get(lead, []) for lead in elevated_leads]

    # Find common regions across all elevated leads
    if not regions_per_lead:
        return False, None, metrics

    common_regions = set(regions_per_lead[0])
    for regions in regions_per_lead[1:]:
        common_regions &= set(regions)

    if not common_regions:
        # No anatomical contiguity → not STEMI
        return False, None, metrics

    metrics["contiguous_groups_involved"] = list(common_regions)

    # Detect reciprocal depression
    # Simplified: opposite wall should have ST depression
    if "anterior" in common_regions and st_deviations.get("III", 0) < -0.05:
        metrics["has_reciprocal_depression"] = True
    elif "inferior" in common_regions and (st_deviations.get("V1", 0) < -0.05 or st_deviations.get("V2", 0) < -0.05):
        metrics["has_reciprocal_depression"] = True
    elif "lateral" in common_regions and st_deviations.get("III", 0) < -0.05:
        metrics["has_reciprocal_depression"] = True

    # Classify region
    # Prioritize by specificity
    region = None
    if len(common_regions) == 1:
        region = list(common_regions)[0]
    else:
        # Multiple regions involved (e.g., anterior + lateral → anterolateral)
        if "anterior" in common_regions and "lateral" in common_regions:
            region = "anterolateral"
        elif "inferior" in common_regions and "lateral" in common_regions:
            region = "inferolateral"
        else:
            region = list(common_regions)[0]  # Default to first

    return True, region, metrics


def assign_kcd_code_for_mi(
    stemi_suspected: bool,
    stemi_region: Optional[str],
    mi_probability: float,
    has_qrs_duration_wide: bool = False,
    has_pathologic_q_waves: bool = False,
) -> Tuple[str, str]:
    """
    Assigns KCD-10 code for MI based on STEMI presentation, region, and DL probability.

    Returns:
        (kcd_code: str, description: str)

    KCD-10 MI codes:
      - I21.0: Anterior wall MI
      - I21.1: Inferior wall MI
      - I21.9: Unspecified MI
      - I25.2: Old MI (history of MI, used when no ST elevation but DL+)
    """
    if not mi_probability >= 0.50:  # Threshold for MI presence
        return "Z01.8", "정상 심전도"

    # STEMI (acute presentation with ST elevation)
    if stemi_suspected:
        if stemi_region == "anterior" or stemi_region == "anterolateral":
            return "I21.0", "전벽 ST분절상승심근경색 (Anterior STEMI)"
        elif stemi_region == "inferior" or stemi_region == "inferolateral":
            return "I21.1", "하벽 ST분절상승심근경색 (Inferior STEMI)"
        else:
            return "I21.9", "부위 미상 ST분절상승심근경색 (Unspecified STEMI)"

    # NSTEMI or old MI (DL positive but no STEMI on ECG)
    # Heuristic: wide QRS or pathologic Q waves suggest chronic/old MI
    if has_qrs_duration_wide or has_pathologic_q_waves:
        return "I25.2", "과거 심근경색 (Old Myocardial Infarction)"
    else:
        # Recent NSTEMI without ST elevation is hard to code without troponin
        # Default to unspecified or old MI code as fallback
        return "I21.9", "비-ST분절상승심근경색 의심 (NSTEMI suspected)"


def get_primary_kcd_code(
    primary_findings: List[str],
    stemi_suspected: bool,
    stemi_region: Optional[str],
    mi_probability: float,
    wide_qrs: bool = False,
) -> Tuple[str, str]:
    """
    Selects the single most important KCD-10 code from findings.
    MI takes priority if present; otherwise pick the highest-risk finding.

    Returns:
        (kcd_code, description)
    """
    # Priority: MI > AFIB > LBBB > others
    if "MI" in primary_findings or mi_probability >= 0.50:
        return assign_kcd_code_for_mi(stemi_suspected, stemi_region, mi_probability, wide_qrs)

    kcd_static = {
        "AFIB": ("I48.0", "심방세동 (Atrial Fibrillation)"),
        "LBBB": ("I45.2", "좌각차단 (Left Bundle Branch Block)"),
        "RBBB": ("I45.1", "우각차단 (Right Bundle Branch Block)"),
        "1AVB": ("I44.0", "1도 방실차단 (First-Degree AV Block)"),
        "LAFB": ("I44.4", "좌전섬유속차단 (Left Anterior Fascicular Block)"),
        "PVC":  ("I49.3", "심실 조기탈분극 (Premature Ventricular Contractions)"),
        "PAC":  ("I49.1", "심방 조기탈분극 (Premature Atrial Contractions)"),
        "LVH":  ("I51.7", "좌심실비대 (Left Ventricular Hypertrophy)"),
        "STTC": ("R94.3", "비특이적 ST-T 분절 이상 (Nonspecific ST-T Changes)"),
        "NORM": ("Z01.8", "정상 심전도 (Normal ECG)"),
    }

    for finding in primary_findings:
        if finding in kcd_static:
            return kcd_static[finding]

    # Fallback
    return "Z01.8", "정상 심전도"
