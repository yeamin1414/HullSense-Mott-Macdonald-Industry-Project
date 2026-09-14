"""Classical computer-vision baseline for hull stain detection.

Implemented with plain NumPy colour-distance thresholding (no heavyweight
dependency required), so this always works even in a minimal environment and
acts as a sanity-check / fallback for the CNN in `cnn_model.py`. This mirrors
a real deployment strategy: never trust a single model — cross-check it.
"""
from __future__ import annotations

import numpy as np

from src.vision.synthetic_data import grid_coverage


class ClassicalStainDetector:
    """Flags a pixel as 'stained' if it deviates from the dominant hull colour
    by more than `color_distance_threshold`, then aggregates into a grid."""

    def __init__(self, color_distance_threshold: float = 28.0) -> None:
        self.color_distance_threshold = color_distance_threshold

    def _pixel_mask(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        # Estimate the "clean hull" colour as the median of the outer border
        # pixels, which are far more likely to be unstained than the centre.
        border = np.concatenate(
            [img[0, :, :], img[-1, :, :], img[:, 0, :], img[:, -1, :]], axis=0
        )
        base_color = np.median(border, axis=0)
        dist = np.linalg.norm(img - base_color, axis=2)
        return (dist > self.color_distance_threshold).astype(np.float32)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return self._pixel_mask(image)

    def predict_grid(self, image: np.ndarray, rows: int, cols: int) -> np.ndarray:
        mask = self._pixel_mask(image)
        return grid_coverage(mask, rows, cols)
