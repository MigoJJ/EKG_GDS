"""
Phase 4.5: ECG Image Parser

Converts scanned ECG paper (PNG/JPEG) to digital signal.
Supports standard ECG paper formats (GE, Philips, Mortara).

Workflow:
  1. Load image (PNG/JPEG)
  2. Detect ECG grid (Hough transform)
  3. Extract waveform curves (contour detection)
  4. Calibrate pixel→mV/sec (standard: 1mm=0.1mV, 40mm/s)
  5. Return 12-lead signal

Medical Standards:
  - Standard ECG paper: 10mm/mV vertical, 25mm/sec horizontal
  - For printouts: scale may vary (0.5x - 2.0x)
  - Accuracy target: ±5% voltage, ±2% timing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ECGImageParser:
    """Extract 12-lead ECG signal from scanned paper."""

    # Standard ECG grid dimensions (in pixels, at 300 DPI)
    # Adjust based on image resolution
    STANDARD_DPI = 300
    GRID_SPACING_MM = 1.0  # 1mm grid spacing on standard ECG paper
    VOLTAGE_PER_MM = 0.1  # Standard: 0.1 mV per mm
    SPEED_MM_SEC = 25.0  # Standard: 25 mm/sec
    
    # Lead order on standard 12-lead ECG paper
    LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self, image_path: str, dpi: int = 300):
        """
        Initialize ECG image parser.

        Args:
            image_path: Path to ECG image (PNG or JPEG)
            dpi: Image resolution (dots per inch)
        """
        self.image_path = Path(image_path)
        self.dpi = dpi
        self.image = None
        self.gray = None
        self.grid_spacing_px = None
        self.signal = None
        
        self._load_image()

    def _load_image(self):
        """Load and validate ECG image."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # Load image
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        logger.info(f"Loaded ECG image: {self.image.shape}")

    def extract_signal(self) -> Optional[np.ndarray]:
        """
        Extract 12-lead ECG signal from image.

        Returns:
            Signal as numpy array (shape: 12, num_samples) in mV
        """
        try:
            # 1. Detect grid
            self._detect_grid()
            if self.grid_spacing_px is None:
                raise ValueError("Failed to detect ECG grid")

            # 2. Extract waveforms
            leads_pixels = self._extract_waveforms()
            if not leads_pixels:
                raise ValueError("Failed to extract waveforms")

            # 3. Calibrate (pixel → mV)
            signal = self._calibrate_signal(leads_pixels)
            
            self.signal = signal
            logger.info(f"Extracted signal: {signal.shape}")
            return signal

        except Exception as e:
            logger.error(f"Signal extraction failed: {e}")
            return None

    def _detect_grid(self):
        """Detect ECG grid spacing using Hough line detection."""
        try:
            # Pre-process: threshold to enhance grid
            _, binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)
            
            # Invert (grid is dark, background is light)
            binary = cv2.bitwise_not(binary)
            
            # Detect edges
            edges = cv2.Canny(binary, 30, 100)
            
            # Detect lines (Hough)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            
            if lines is None:
                logger.warning("No grid lines detected")
                return

            # Extract horizontal line spacing
            horizontal_y = []
            for line in lines:
                rho, theta = line[0]
                # Check if line is roughly horizontal (theta ~ 0 or π)
                if np.abs(theta) < 0.1 or np.abs(theta - np.pi) < 0.1:
                    y = int(rho / np.cos(theta) if np.cos(theta) != 0 else rho)
                    horizontal_y.append(y)

            if len(horizontal_y) < 2:
                logger.warning("Not enough horizontal grid lines")
                return

            # Calculate average spacing
            horizontal_y = sorted(horizontal_y)
            spacings = np.diff(horizontal_y)
            avg_spacing = np.median(spacings[spacings > 0])
            
            self.grid_spacing_px = int(avg_spacing)
            logger.info(f"Detected grid spacing: {self.grid_spacing_px} pixels")

        except Exception as e:
            logger.warning(f"Grid detection failed: {e}")

    def _extract_waveforms(self) -> Dict[str, np.ndarray]:
        """Extract waveform curves for each lead."""
        try:
            # Pre-process image to enhance waveforms
            # Apply Gaussian blur to smooth
            blurred = cv2.GaussianBlur(self.gray, (3, 3), 0)
            
            # Threshold to isolate waveform (usually dark lines)
            _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours (waveforms)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No waveforms detected")
                return {}

            # Extract curve points
            leads_pixels = {}
            for i, contour in enumerate(contours[:12]):  # 12 leads max
                # Approximate contour to points
                epsilon = 0.01 * cv2.arcLength(contour, False)
                approx = cv2.approxPolyDP(contour, epsilon, False)
                
                # Extract (x, y) coordinates
                points = approx.reshape(-1, 2).astype(float)
                
                lead_name = self.LEAD_ORDER[i] if i < len(self.LEAD_ORDER) else f"Unknown_{i}"
                leads_pixels[lead_name] = points
                
                logger.debug(f"Lead {lead_name}: {len(points)} points")

            return leads_pixels

        except Exception as e:
            logger.error(f"Waveform extraction failed: {e}")
            return {}

    def _calibrate_signal(self, leads_pixels: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Convert pixel coordinates to mV signal.

        Standard ECG paper:
          - 1 small square = 1mm = 0.1mV (vertical)
          - 1 small square = 1mm = 0.04sec at 25mm/sec (horizontal)
        """
        try:
            if self.grid_spacing_px is None:
                logger.warning("Grid spacing unknown, using default calibration")
                grid_spacing_px = 10  # Default assumption

            # Calibration factors
            mm_per_px = self.GRID_SPACING_MM / self.grid_spacing_px
            mv_per_px = self.VOLTAGE_PER_MM * mm_per_px
            
            # Standard ECG baseline is center of paper
            # Assume image center is baseline (0mV)
            baseline_y = self.gray.shape[0] / 2

            # Extract signal for each lead
            signals = []
            for lead_name in self.LEAD_ORDER:
                if lead_name not in leads_pixels:
                    # Use zeros if lead not detected
                    signals.append(np.zeros(1000))
                    continue

                points = leads_pixels[lead_name]
                
                # Sort by x coordinate (time)
                points = points[np.argsort(points[:, 0])]
                
                # Convert pixel Y to voltage (Y increases downward in images)
                # Negative because image Y increases downward
                voltage = -(points[:, 1] - baseline_y) * mv_per_px
                
                # Interpolate to standard length (1000 samples @ 100 Hz)
                # Assume image width represents ~10 seconds
                x_normalized = np.linspace(0, 1, len(voltage))
                x_standard = np.linspace(0, 1, 1000)
                
                signal_interp = np.interp(x_standard, x_normalized, voltage)
                signals.append(signal_interp)

            # Stack into 2D array (12, samples)
            signal_array = np.array(signals, dtype=np.float32)
            return signal_array

        except Exception as e:
            logger.error(f"Signal calibration failed: {e}")
            return None

    def detect_image_quality(self) -> Tuple[bool, str]:
        """
        Validate image quality for ECG analysis.

        Returns:
            (is_valid, error_message)
        """
        if self.image is None:
            return False, "Image not loaded"

        # Check image size
        height, width = self.image.shape[:2]
        if width < 100 or height < 100:
            return False, "Image too small"

        # Check contrast
        contrast = np.std(self.gray)
        if contrast < 20:
            return False, "Low contrast (grid not visible)"

        # Check for common artifacts
        # Too much white (overexposed) or black (underexposed)
        mean_brightness = np.mean(self.gray)
        if mean_brightness < 30 or mean_brightness > 220:
            return False, "Exposure issue (too dark or too bright)"

        return True, ""

    def get_metadata(self) -> Dict[str, Any]:
        """Get image metadata."""
        if self.image is None:
            return {}

        height, width = self.image.shape[:2]
        return {
            "image_size": (width, height),
            "dpi": self.dpi,
            "grid_spacing_px": self.grid_spacing_px,
            "signal_extracted": self.signal is not None,
            "signal_shape": self.signal.shape if self.signal is not None else None,
        }


def parse_ecg_image(image_path: str, dpi: int = 300) -> Optional[np.ndarray]:
    """
    Convenience function to extract ECG from image.

    Args:
        image_path: Path to ECG image
        dpi: Image resolution

    Returns:
        12-lead signal or None on error
    """
    try:
        parser = ECGImageParser(image_path, dpi=dpi)
        
        # Validate image
        is_valid, error = parser.detect_image_quality()
        if not is_valid:
            logger.warning(f"Image quality issue: {error}")
            return None

        # Extract signal
        signal = parser.extract_signal()
        return signal

    except Exception as e:
        logger.error(f"ECG image parsing failed: {e}")
        return None
