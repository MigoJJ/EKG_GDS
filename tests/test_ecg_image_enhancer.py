"""Phase 4.6: ECG Image Enhancer Tests"""
import pytest
from pipeline.ecg_image_enhancer import ECGImageEnhancer, MultiPageHandler
import numpy as np

class TestRotationDetection:
    def test_rotation_detection(self):
        """Rotation detection should work."""
        # Create test image with grid
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        enhancer = ECGImageEnhancer(img)
        angle = enhancer.detect_rotation()
        assert isinstance(angle, float)
        assert -45 <= angle <= 45

    def test_rotation_correction(self):
        """Image rotation correction should work."""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        enhancer = ECGImageEnhancer(img)
        rotated = enhancer.correct_rotation(angle=5.0)
        assert rotated.shape == img.shape

class TestImageEnhancement:
    def test_contrast_enhancement(self):
        """Contrast enhancement should not crash."""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 128
        enhancer = ECGImageEnhancer(img)
        enhanced = enhancer.enhance_contrast()
        assert enhanced is not None

    def test_noise_removal(self):
        """Noise removal should work."""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 200
        enhancer = ECGImageEnhancer(img)
        denoised = enhancer.remove_noise()
        assert denoised is not None

    def test_full_enhancement(self):
        """Full enhancement pipeline should work."""
        img = np.ones((600, 800, 3), dtype=np.uint8) * 200
        enhancer = ECGImageEnhancer(img)
        enhanced = enhancer.full_enhancement()
        assert enhanced is not None

class TestMultiPageHandler:
    def test_horizontal_merge(self, tmp_path):
        """Horizontal merge should work."""
        # Skip if cv2 not available
        pytest.importorskip("cv2")
        
        # Create test images
        img1 = np.ones((400, 600, 3), dtype=np.uint8) * 200
        img2 = np.ones((400, 600, 3), dtype=np.uint8) * 150
        
        import cv2
        from pathlib import Path
        cv2.imwrite(str(tmp_path / "page1.png"), img1)
        cv2.imwrite(str(tmp_path / "page2.png"), img2)
        
        handler = MultiPageHandler([
            str(tmp_path / "page1.png"),
            str(tmp_path / "page2.png")
        ])
        
        merged = handler.merge_horizontally()
        assert merged is not None

    def test_auto_layout_detection(self, tmp_path):
        """Auto layout detection should work."""
        pytest.importorskip("cv2")
        
        img1 = np.ones((300, 600, 3), dtype=np.uint8) * 200
        img2 = np.ones((300, 600, 3), dtype=np.uint8) * 150
        
        import cv2
        cv2.imwrite(str(tmp_path / "p1.png"), img1)
        cv2.imwrite(str(tmp_path / "p2.png"), img2)
        
        handler = MultiPageHandler([
            str(tmp_path / "p1.png"),
            str(tmp_path / "p2.png")
        ])
        
        layout = handler.detect_layout()
        assert layout in ['horizontal', 'vertical']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
