"""
Phase 5 Step 4: HL7 Parser

Parses HL7 v2.5 messages (Health Level 7) for patient and ECG metadata.

HL7 is the standard message format for hospital information systems (HIS/EMR).
ECG data is transmitted with patient demographics, vital signs, and measurements.

Format:
    MSH|^~\\&|...             Message header
    PID|...                   Patient identification
    OBR|...                   Order result
    OBX|...                   Observation result

Usage:
    parser = HL7Parser(hl7_message_string)
    patient_info = parser.get_patient_info()
    ecg_data = parser.get_ecg_data()
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import hl7
    HAS_HL7 = True
except ImportError:
    HAS_HL7 = False
    logger.warning("hl7 library not installed. Install with: pip install hl7")


class HL7Parser:
    """Parse HL7 messages and extract patient/ECG information."""

    # HL7 encoding characters (default)
    FIELD_SEP = '|'
    COMPONENT_SEP = '^'
    REPETITION_SEP = '~'
    ESCAPE_CHAR = '\\'
    SUB_COMPONENT_SEP = '&'

    def __init__(self, hl7_message: str):
        """
        Initialize HL7 parser.

        Args:
            hl7_message: HL7 message string (can be multiline)
        """
        self.raw_message = hl7_message.strip()
        self.segments = {}
        self._parse_message()

    def _parse_message(self):
        """Parse HL7 message into segments."""
        lines = self.raw_message.split('\n')

        if not lines:
            raise ValueError("Empty HL7 message")

        # First line should be MSH (message header)
        msh_line = lines[0].strip()
        if not msh_line.startswith('MSH'):
            raise ValueError("Invalid HL7 message (missing MSH segment)")

        # Parse MSH to get encoding characters
        self._parse_encoding_chars(msh_line)

        # Parse all segments
        for line in lines:
            line = line.strip()
            if not line:
                continue

            segment_type = line.split(self.FIELD_SEP)[0]
            fields = line.split(self.FIELD_SEP)

            if segment_type not in self.segments:
                self.segments[segment_type] = []

            self.segments[segment_type].append(fields)

        logger.info(f"Parsed HL7 message: {len(self.segments)} segment types")

    def _parse_encoding_chars(self, msh_line: str):
        """Parse encoding characters from MSH segment."""
        if len(msh_line) > 3:
            # MSH|^~\&| → fields[1] = '^~\&'
            enc = msh_line[3:7]
            if len(enc) >= 4:
                self.COMPONENT_SEP = enc[0]
                self.REPETITION_SEP = enc[1]
                self.ESCAPE_CHAR = enc[2]
                self.SUB_COMPONENT_SEP = enc[3]

    def get_patient_info(self) -> Dict[str, Any]:
        """
        Extract patient demographics from PID segment.

        Returns:
            {
                'patient_id': '...',
                'patient_name': '...',
                'date_of_birth': '...',
                'sex': 'M' or 'F',
                'age': int or None,
                'address': '...',
                'phone': '...',
                'email': '...'
            }
        """
        info = {}

        if 'PID' not in self.segments:
            logger.warning("No PID segment in HL7 message")
            return info

        pid = self.segments['PID'][0]  # First PID segment

        try:
            # PID fields (0-indexed after 'PID' at position 0)
            # [0] = PID
            # [1] = Set ID
            # [2] = Patient number (internal)
            # [3] = Patient ID (external, MRN)
            # [4] = Alternate patient ID
            # [5] = Patient name
            # [6] = Mother's maiden name
            # [7] = Date of birth
            # [8] = Sex
            # [13] = Phone number
            # [14] = Business phone
            # [15] = Email

            if len(pid) > 3:
                patient_id = pid[3]
                if isinstance(patient_id, str):
                    info['patient_id'] = patient_id.split(self.COMPONENT_SEP)[0]

            if len(pid) > 5:
                name = pid[5]
                if isinstance(name, str):
                    parts = name.split(self.COMPONENT_SEP)
                    if len(parts) >= 2:
                        info['patient_name'] = f"{parts[1]} {parts[0]}"  # First^Last
                    else:
                        info['patient_name'] = name

            if len(pid) > 7:
                dob = pid[7]
                if dob:
                    # Format: YYYYMMDD
                    try:
                        info['date_of_birth'] = self._parse_hl7_date(dob)
                        info['age'] = self._calculate_age(dob)
                    except Exception:
                        info['date_of_birth'] = dob

            if len(pid) > 8:
                sex = pid[8]
                if sex in ['M', 'F', 'U', 'A', 'N']:
                    info['sex'] = sex

            if len(pid) > 13:
                phone = pid[13]
                if phone:
                    info['phone'] = phone

            if len(pid) > 15:
                email = pid[15]
                if email:
                    info['email'] = email

            return info

        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing PID segment: {e}")
            return info

    def get_ecg_data(self) -> Dict[str, Any]:
        """
        Extract ECG measurements from OBX segments.

        Returns:
            {
                'heart_rate': int,
                'pr_interval': float,
                'qrs_duration': float,
                'qt_interval': float,
                'interpretation': str,
                'measurements': {...}
            }
        """
        ecg_data = {}

        if 'OBX' not in self.segments:
            logger.warning("No OBX (observation) segments in HL7 message")
            return ecg_data

        measurements = {}

        # Parse each OBX segment
        for obx in self.segments['OBX']:
            try:
                # OBX fields:
                # [0] = OBX
                # [1] = Set ID
                # [2] = Value type (NM=numeric, ST=string, etc)
                # [3] = Observation identifier (code^description)
                # [4] = Observation sub-ID
                # [5] = Observation value
                # [6] = Units
                # [7] = Reference range
                # [8] = Abnormal flags
                # [9] = Probability

                if len(obx) < 6:
                    continue

                obs_id = obx[3] if len(obx) > 3 else ""
                value = obx[5] if len(obx) > 5 else ""
                units = obx[6] if len(obx) > 6 else ""

                # Parse observation identifier
                if self.COMPONENT_SEP in str(obs_id):
                    code, description = str(obs_id).split(self.COMPONENT_SEP)[:2]
                else:
                    code = str(obs_id)
                    description = ""

                # Map common ECG codes
                if code in ['HR', '5731-0']:  # Heart rate
                    try:
                        ecg_data['heart_rate'] = int(float(value))
                    except ValueError:
                        pass

                elif code in ['PR', '8633-5']:  # PR interval
                    try:
                        ecg_data['pr_interval'] = float(value)
                    except ValueError:
                        pass

                elif code in ['QRS', '8646-7']:  # QRS duration
                    try:
                        ecg_data['qrs_duration'] = float(value)
                    except ValueError:
                        pass

                elif code in ['QT', '8647-5']:  # QT interval
                    try:
                        ecg_data['qt_interval'] = float(value)
                    except ValueError:
                        pass

                # Store all measurements
                measurements[code] = {
                    'description': description,
                    'value': value,
                    'units': units,
                }

            except Exception as e:
                logger.warning(f"Error parsing OBX segment: {e}")

        if measurements:
            ecg_data['measurements'] = measurements

        return ecg_data

    def get_interpretation(self) -> Optional[str]:
        """
        Extract ECG interpretation from OBX or NTE segments.

        Returns:
            ECG interpretation text or None
        """
        # Look for NTE (note) segments with interpretation
        if 'NTE' in self.segments:
            for nte in self.segments['NTE']:
                if len(nte) > 3:
                    text = nte[3]
                    if isinstance(text, str) and text.strip():
                        return text

        # Or look in OBX for interpretation code
        if 'OBX' in self.segments:
            for obx in self.segments['OBX']:
                try:
                    if len(obx) > 3:
                        obs_id = str(obx[3])
                        if 'INTERPRETATION' in obs_id.upper() or 'IMPRESSION' in obs_id.upper():
                            if len(obx) > 5:
                                return obx[5]
                except Exception:
                    pass

        return None

    def get_study_datetime(self) -> Optional[str]:
        """Get ECG study date/time from OBR segment."""
        if 'OBR' not in self.segments:
            return None

        try:
            obr = self.segments['OBR'][0]
            # OBR[6] = observation datetime
            if len(obr) > 6 and obr[6]:
                return self._parse_hl7_datetime(obr[6])
            return None
        except Exception:
            return None

    def _parse_hl7_date(self, hl7_date: str) -> str:
        """Parse HL7 date format YYYYMMDD to YYYY-MM-DD."""
        try:
            if len(hl7_date) >= 8:
                year = hl7_date[0:4]
                month = hl7_date[4:6]
                day = hl7_date[6:8]
                return f"{year}-{month}-{day}"
            return hl7_date
        except Exception:
            return hl7_date

    def _parse_hl7_datetime(self, hl7_dt: str) -> str:
        """Parse HL7 datetime format YYYYMMDDHHMMSS."""
        try:
            if len(hl7_dt) >= 8:
                date_str = self._parse_hl7_date(hl7_dt[:8])
                if len(hl7_dt) >= 14:
                    time_str = f"{hl7_dt[8:10]}:{hl7_dt[10:12]}:{hl7_dt[12:14]}"
                    return f"{date_str} {time_str}"
                return date_str
            return hl7_dt
        except Exception:
            return hl7_dt

    def _calculate_age(self, hl7_dob: str) -> Optional[int]:
        """Calculate age from HL7 date of birth."""
        try:
            dob = datetime.strptime(hl7_dob[:8], '%Y%m%d')
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return max(0, age)
        except Exception:
            return None

    def validate(self) -> Tuple[bool, str]:
        """
        Validate HL7 message structure.

        Returns:
            (is_valid, error_message)
        """
        if not self.raw_message:
            return False, "Empty message"

        if 'MSH' not in self.segments:
            return False, "No MSH segment (invalid HL7)"

        if 'PID' not in self.segments:
            return False, "No PID segment (missing patient info)"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert parsed HL7 to dictionary."""
        return {
            'patient_info': self.get_patient_info(),
            'ecg_data': self.get_ecg_data(),
            'interpretation': self.get_interpretation(),
            'study_datetime': self.get_study_datetime(),
        }
