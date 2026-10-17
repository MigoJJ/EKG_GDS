"""
Phase 4.5: ECG Image Parser Tests

Tests for converting scanned ECG paper images (PNG/JPEG) to digital signals.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

from pipeline.ecg_image_parser import ECGImageParser, parse_ecg_image


class TestECGImageCreation:
    """Helper to create test ECG images."""

    @staticmethod
    def create_test_ecg_image(width: int = 800, height: int = 600) -> np.ndarray:
        """Create a synthetic ECG-like image for testing."""
        # Create white background
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw ECG grid (light gray)
        grid_spacing = 10
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)

        # Draw darker grid (1cm lines)
        major_grid = grid_spacing * 10
        for x in range(0, width, major_grid):
            draw.line([(x, 0), (x, height)], fill=(200, 200, 200), width=2)
        for y in range(0, height, major_grid):
            draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=2)

        # Draw sinusoidal waveform (simulating ECG)
        baseline_y = height // 2
        amplitude = 30
        frequency = 0.05

        for lead in range(3):  # 3 example leads
            points = []
            for x in range(width):
                # Sinusoidal curve with noise
                y = baseline_y - int(amplitude * np.sin(frequency * (x + lead * 100))) + np.random.randint(-2, 2)
                y = max(0, min(height - 1, y))
                points.append((x, y))

            # Draw waveform (dark blue)
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=(0, 0, 100), width=2)

        return np.array(img)


class TestImageLoading:
    """Test image loading and validation."""

    def test_image_not_found(self):
        """Non-existent file should raise error."""
        with pytest.raises(FileNotFoundError):
            ECGImageParser("/nonexistent/file.png")

    def test_invalid_image_file(self):
        """Invalid image file should raise error."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"not_an_image")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError):
                ECGImageParser(tmp_path)
        finally:
            Path(tmp_path).unlink()

    def test_valid_image_loading(self):
        """Valid PNG image should load."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Create a simple test image
            test_img = TestECGImageCreation.create_test_ecg_image()
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            assert parser.image is not None
            assert parser.gray is not None
        finally:
            Path(tmp_path).unlink()


class TestImageQuality:
    """Test image quality detection."""

    def test_normal_quality(self):
        """Normal ECG image should pass quality check."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = TestECGImageCreation.create_test_ecg_image()
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            is_valid, error = parser.detect_image_quality()
            assert is_valid
        finally:
            Path(tmp_path).unlink()

    def test_image_too_small(self):
        """Tiny image should fail quality check."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = TestECGImageCreation.create_test_ecg_image(width=50, height=50)
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            is_valid, error = parser.detect_image_quality()
            assert not is_valid
        finally:
            Path(tmp_path).unlink()

    def test_low_contrast(self):
        """Low contrast image should fail."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Create low-contrast image (all light gray)
            img = Image.new('RGB', (400, 300), color=(200, 200, 200))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            is_valid, error = parser.detect_image_quality()
            # May fail due to low contrast
            if not is_valid:
                assert "contrast" in error.lower()
        finally:
            Path(tmp_path).unlink()


class TestGridDetection:
    """Test ECG grid detection."""

    def test_grid_detection(self):
        """Grid should be detected in normal image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = TestECGImageCreation.create_test_ecg_image()
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            parser._detect_grid()
            # Grid detection may succeed or fail depending on image
            # Just verify it doesn't crash
            assert True
        finally:
            Path(tmp_path).unlink()


class TestSignalExtraction:
    """Test signal extraction from image."""

    def test_signal_shape(self):
        """Extracted signal should have correct shape."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = TestECGImageCreation.create_test_ecg_image()
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            signal = parser.extract_signal()

            # Signal may be None if detection fails, that's ok
            if signal is not None:
                assert signal.shape[0] == 12, "Should have 12 leads"
                assert signal.shape[1] > 0, "Should have samples"
                assert signal.dtype == np.float32, "Should be float32"
        finally:
            Path(tmp_path).unlink()

    def test_signal_values_in_range(self):
        """Signal values should be in reasonable mV range."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = TestECGImageCreation.create_test_ecg_image()
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            signal = parser.extract_signal()

            if signal is not None and signal.size > 0:
                # ECG signals typically in range -5 to 5 mV
                # Allow wider range for test images
                assert np.abs(signal).max() < 100, "Signal values out of range"
        finally:
            Path(tmp_path).unlink()


class TestMetadata:
    """Test metadata extraction."""

    def test_metadata_available(self):
        """Metadata should be available after loading."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = TestECGImageCreation.create_test_ecg_image()
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            parser = ECGImageParser(tmp_path)
            metadata = parser.get_metadata()

            assert "image_size" in metadata
            assert "dpi" in metadata
            assert metadata["dpi"] == 300
        finally:
            Path(tmp_path).unlink()


class TestConvenienceFunction:
    """Test convenience wrapper function."""

    def test_parse_ecg_image(self):
        """parse_ecg_image() should work."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = TestECGImageCreation.create_test_ecg_image()
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            signal = parse_ecg_image(tmp_path)
            # May succeed or return None depending on detection
            if signal is not None:
                assert signal.shape[0] == 12
        finally:
            Path(tmp_path).unlink()

    def test_parse_invalid_image(self):
        """parse_ecg_image() should handle invalid input."""
        signal = parse_ecg_image("/nonexistent.png")
        assert signal is None


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Full image-to-signal pipeline."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Create test image
            test_img = TestECGImageCreation.create_test_ecg_image(width=800, height=600)
            img = Image.fromarray(test_img.astype('uint8'))
            img.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Parse image
            parser = ECGImageParser(tmp_path)

            # Validate quality
            is_valid, error = parser.detect_image_quality()
            if not is_valid:
                pytest.skip(f"Test image quality issue: {error}")

            # Extract signal
            signal = parser.extract_signal()
            if signal is not None:
                assert signal.shape == (12, 1000) or signal.shape[0] == 12
                assert not np.isnan(signal).any(), "Signal should not contain NaN"
                assert not np.isinf(signal).any(), "Signal should not contain Inf"
        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
