"""
Phase 4.6: ECG Image Enhancement

Advanced image processing for scanned ECG paper:
- Automatic rotation detection
- Multi-page handling
- Image quality improvement
- Perspective correction
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ECGImageEnhancer:
    """Enhance scanned ECG images for better signal extraction."""

    def __init__(self, image):
        """
        Initialize enhancer.

        Args:
            image: numpy array (BGR or grayscale)
        """
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    def detect_rotation(self) -> float:
        """
        Detect image rotation using Hough line detection.

        Returns:
            Rotation angle in degrees (0-90, or None if not detected)
        """
        try:
            # Edge detection
            edges = cv2.Canny(self.gray, 50, 150)

            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

            if lines is None or len(lines) == 0:
                logger.warning("No lines detected for rotation")
                return 0.0

            # Extract angles from horizontal lines
            angles = []
            for line in lines:
                rho, theta = line[0]
                # Convert to degrees
                angle_deg = np.degrees(theta)

                # Only consider roughly horizontal lines (0-10 or 170-180 degrees)
                if (angle_deg < 10 or angle_deg > 170):
                    angles.append(angle_deg)
                # Or roughly vertical lines (80-100 degrees)
                elif (80 < angle_deg < 100):
                    angles.append(angle_deg - 90)

            if not angles:
                return 0.0

            # Calculate median angle (robust to outliers)
            median_angle = np.median(angles)

            # Normalize to -45 to 45 range
            if median_angle > 45:
                median_angle -= 90
            elif median_angle < -45:
                median_angle += 90

            logger.info(f"Detected rotation: {median_angle:.1f}°")
            return float(median_angle)

        except Exception as e:
            logger.warning(f"Rotation detection failed: {e}")
            return 0.0

    def correct_rotation(self, angle: Optional[float] = None) -> np.ndarray:
        """
        Correct image rotation.

        Args:
            angle: Rotation angle in degrees. If None, auto-detect.

        Returns:
            Rotated image
        """
        try:
            if angle is None:
                angle = self.detect_rotation()

            if abs(angle) < 0.5:
                return self.image

            # Rotate image
            height, width = self.image.shape[:2]
            center = (width // 2, height // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                self.image,
                rotation_matrix,
                (width, height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

            logger.info(f"Image rotated by {angle:.1f}°")
            return rotated

        except Exception as e:
            logger.error(f"Rotation correction failed: {e}")
            return self.image

    def enhance_contrast(self) -> np.ndarray:
        """
        Enhance image contrast for better grid detection.

        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space for better contrast enhancement
            if len(self.image.shape) == 3:
                lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
            else:
                l = self.image
                a = np.zeros_like(l)
                b = np.zeros_like(l)

            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            # Merge back
            if len(self.image.shape) == 3:
                lab_enhanced = cv2.merge([l_enhanced, a, b])
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            else:
                enhanced = l_enhanced

            logger.info("Contrast enhanced")
            return enhanced

        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return self.image

    def remove_noise(self, kernel_size: int = 5) -> np.ndarray:
        """
        Remove noise while preserving edges.

        Args:
            kernel_size: Bilateral filter kernel size

        Returns:
            Denoised image
        """
        try:
            if len(self.image.shape) == 3:
                denoised = cv2.bilateralFilter(self.image, kernel_size, 75, 75)
            else:
                denoised = cv2.bilateralFilter(self.image, kernel_size, 75, 75)

            logger.info("Noise removed")
            return denoised

        except Exception as e:
            logger.warning(f"Noise removal failed: {e}")
            return self.image

    def detect_paper_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the actual ECG paper region (bounding box).

        Returns:
            (x, y, w, h) bounding box or None
        """
        try:
            # Threshold to find paper (mostly white background)
            _, binary = cv2.threshold(self.gray, 200, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Find largest contour (should be the paper)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            logger.info(f"Paper region detected: ({x}, {y}, {w}, {h})")
            return (x, y, w, h)

        except Exception as e:
            logger.warning(f"Paper region detection failed: {e}")
            return None

    def crop_to_paper(self) -> np.ndarray:
        """
        Crop image to paper region only.

        Returns:
            Cropped image
        """
        try:
            bbox = self.detect_paper_region()
            if bbox is None:
                return self.image

            x, y, w, h = bbox
            cropped = self.image[y:y+h, x:x+w]

            logger.info(f"Image cropped to paper region")
            return cropped

        except Exception as e:
            logger.warning(f"Cropping failed: {e}")
            return self.image

    def full_enhancement(self) -> np.ndarray:
        """
        Apply all enhancements in sequence.

        Returns:
            Enhanced image
        """
        try:
            # 1. Detect and correct rotation
            rotated = self.correct_rotation()
            enhancer = ECGImageEnhancer(rotated)

            # 2. Crop to paper
            cropped = enhancer.crop_to_paper()
            enhancer = ECGImageEnhancer(cropped)

            # 3. Remove noise
            denoised = enhancer.remove_noise()
            enhancer = ECGImageEnhancer(denoised)

            # 4. Enhance contrast
            enhanced = enhancer.enhance_contrast()

            logger.info("Full enhancement completed")
            return enhanced

        except Exception as e:
            logger.error(f"Full enhancement failed: {e}")
            return self.image


class MultiPageHandler:
    """Handle multi-page ECG scans."""

    def __init__(self, image_paths: list):
        """
        Initialize multi-page handler.

        Args:
            image_paths: List of image file paths
        """
        self.image_paths = image_paths
        self.images = []
        self._load_images()

    def _load_images(self):
        """Load all images."""
        for path in self.image_paths:
            try:
                img = cv2.imread(path)
                if img is not None:
                    self.images.append(img)
                    logger.info(f"Loaded: {path}")
                else:
                    logger.warning(f"Failed to load: {path}")
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")

    def merge_horizontally(self) -> Optional[np.ndarray]:
        """
        Merge images horizontally (side by side).

        Returns:
            Merged image
        """
        if len(self.images) < 2:
            logger.warning("Need at least 2 images to merge")
            return None

        try:
            # Resize all images to same height
            max_height = max(img.shape[0] for img in self.images)
            resized = []
            for img in self.images:
                ratio = max_height / img.shape[0]
                new_width = int(img.shape[1] * ratio)
                resized_img = cv2.resize(img, (new_width, max_height))
                resized.append(resized_img)

            # Concatenate horizontally
            merged = cv2.hconcat(resized)
            logger.info(f"Merged {len(self.images)} images horizontally")
            return merged

        except Exception as e:
            logger.error(f"Horizontal merge failed: {e}")
            return None

    def merge_vertically(self) -> Optional[np.ndarray]:
        """
        Merge images vertically (top to bottom).

        Returns:
            Merged image
        """
        if len(self.images) < 2:
            logger.warning("Need at least 2 images to merge")
            return None

        try:
            # Resize all images to same width
            max_width = max(img.shape[1] for img in self.images)
            resized = []
            for img in self.images:
                ratio = max_width / img.shape[1]
                new_height = int(img.shape[0] * ratio)
                resized_img = cv2.resize(img, (max_width, new_height))
                resized.append(resized_img)

            # Concatenate vertically
            merged = cv2.vconcat(resized)
            logger.info(f"Merged {len(self.images)} images vertically")
            return merged

        except Exception as e:
            logger.error(f"Vertical merge failed: {e}")
            return None

    def detect_layout(self) -> str:
        """
        Auto-detect merge layout (horizontal or vertical).

        Returns:
            'horizontal' or 'vertical'
        """
        if len(self.images) < 2:
            return 'horizontal'

        # If images are wider than tall, merge vertically
        # If images are taller than wide, merge horizontally
        avg_width = np.mean([img.shape[1] for img in self.images])
        avg_height = np.mean([img.shape[0] for img in self.images])

        if avg_width > avg_height:
            layout = 'vertical'
        else:
            layout = 'horizontal'

        logger.info(f"Detected layout: {layout}")
        return layout

    def merge_auto(self) -> Optional[np.ndarray]:
        """
        Automatically merge images with detected layout.

        Returns:
            Merged image
        """
        layout = self.detect_layout()
        if layout == 'horizontal':
            return self.merge_horizontally()
        else:
            return self.merge_vertically()
