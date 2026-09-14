import numpy as np

from src.vision.classical_detector import ClassicalStainDetector


def test_clean_image_low_coverage():
    detector = ClassicalStainDetector()
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    grid = detector.predict_grid(image, rows=4, cols=4)
    assert grid.mean() < 0.05


def test_stained_patch_detected():
    detector = ClassicalStainDetector(color_distance_threshold=20)
    image = np.full((40, 40, 3), 200, dtype=np.uint8)
    image[0:20, 0:20] = [40, 40, 40]  # dark stain in the top-left quadrant
    grid = detector.predict_grid(image, rows=2, cols=2)
    assert grid[0, 0] > grid[1, 1]
    assert grid[0, 0] > 0.5
