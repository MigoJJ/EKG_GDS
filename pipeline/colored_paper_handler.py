"""Phase 4.7: Colored Paper Support"""
import cv2
import numpy as np

class ColoredPaperHandler:
    """Handle colored ECG paper (blue, green, pink)."""

    PAPER_COLORS = {
        'white': (200, 200, 200),
        'blue': (100, 100, 180),
        'green': (100, 180, 100),
        'pink': (180, 100, 150),
    }

    def __init__(self, image):
        self.image = image
        self.detected_color = None

    def detect_paper_color(self) -> str:
        """Detect paper color using histogram analysis."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Calculate color histogram
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])

        # Find dominant hue
        dominant_hue = np.argmax(h_hist)

        # Map hue to color
        if dominant_hue < 10 or dominant_hue > 170:
            color = 'red'  # Red/pink
        elif 10 <= dominant_hue < 35:
            color = 'yellow'
        elif 35 <= dominant_hue < 90:
            color = 'green'
        elif 90 <= dominant_hue < 130:
            color = 'cyan'
        elif 130 <= dominant_hue < 170:
            color = 'blue'
        else:
            color = 'white'

        self.detected_color = color
        return color

    def adjust_thresholds(self, color: str = None):
        """Auto-adjust thresholds based on paper color."""
        if color is None:
            color = self.detect_paper_color()

        # Adjust threshold values per color
        if color == 'white':
            lower, upper = 180, 255
        elif color == 'blue':
            lower, upper = 170, 240
        elif color == 'green':
            lower, upper = 160, 230
        elif color == 'pink':
            lower, upper = 150, 220
        else:
            lower, upper = 150, 255

        return lower, upper

    def extract_grid_colored(self):
        """Extract grid from colored paper."""
        color = self.detect_paper_color()
        lower, upper = self.adjust_thresholds(color)

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold for colored paper
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary
