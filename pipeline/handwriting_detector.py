"""Phase 4.7: Handwriting & Annotation Detection"""
import cv2
import numpy as np

class HandwritingDetector:
    """Detect and preserve handwritten notes on ECG."""

    def __init__(self, image):
        self.image = image
        self.annotations = []

    def find_annotations(self):
        """Find handwritten annotations (doctor's notes)."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) \
            if len(self.image.shape) == 3 else self.image

        # Find dark pixels (handwriting is typically dark)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Find contours (handwritten strokes)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        annotations = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Handwriting typically has medium-sized contours
            if 100 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                annotations.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'type': 'handwriting'
                })

        self.annotations = annotations
        return annotations

    def preserve_waveform(self):
        """Remove handwritten annotations, preserve waveform."""
        # Create mask of handwritten areas
        mask = np.ones(self.image.shape[:2], dtype=np.uint8) * 255

        for ann in self.annotations:
            x, y, w, h = ann['bbox']
            mask[y:y+h, x:x+w] = 0

        # Apply mask to image
        result = self.image.copy()
        result[mask == 0] = 255  # Make annotation areas white

        return result

    def extract_annotation_regions(self):
        """Extract regions with handwritten text for OCR."""
        regions = []
        for ann in self.annotations:
            x, y, w, h = ann['bbox']
            region = self.image[y:y+h, x:x+w]
            regions.append({
                'image': region,
                'bbox': (x, y, w, h)
            })
        return regions

    def confidence_score(self) -> float:
        """Score likelihood of annotations interfering with ECG."""
        if not self.annotations:
            return 1.0  # No annotations = high confidence

        # If many annotations, lower confidence
        # ECG image should be mostly waveform, not notes
        image_area = self.image.shape[0] * self.image.shape[1]
        annotation_area = sum(a['area'] for a in self.annotations)

        interference_ratio = annotation_area / image_area

        # Return confidence (1.0 = no interference, 0.0 = heavy interference)
        return max(0.0, 1.0 - interference_ratio)
