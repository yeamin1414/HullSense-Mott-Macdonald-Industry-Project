"""Unified stain-detection facade.

Picks the best available backend at runtime — trained CNN > classical CV —
and exposes a single `StainDetector.predict()` used by the orchestrator and
dashboard. This is the "Find the dirt" step from the concept design.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.vision.classical_detector import ClassicalStainDetector
from src.vision.cnn_model import DEFAULT_WEIGHTS_PATH, TORCH_AVAILABLE, CNNStainDetector


@dataclass
class DetectionResult:
    grid: np.ndarray  # (rows, cols) dirtiness score in [0, 1]
    backend: str  # "cnn" or "classical"
    dirty_cells: list[tuple[int, int]]
    coverage_pct: float  # % of hull area estimated dirty


class StainDetector:
    def __init__(self, rows: int = 4, cols: int = 6, dirty_threshold: float = 0.35,
                 weights_path: str | Path = DEFAULT_WEIGHTS_PATH, prefer_cnn: bool = True):
        self.rows, self.cols = rows, cols
        self.dirty_threshold = dirty_threshold
        self.classical = ClassicalStainDetector()
        self.cnn: CNNStainDetector | None = None
        self.backend = "classical"

        if prefer_cnn and TORCH_AVAILABLE:
            try:
                candidate = CNNStainDetector(rows, cols, weights_path)
                if candidate.trained:
                    self.cnn = candidate
                    self.backend = "cnn"
            except Exception:
                self.cnn = None

    def predict(self, image: np.ndarray) -> DetectionResult:
        if self.cnn is not None:
            grid = self.cnn.predict_grid(image)
            backend = "cnn"
        else:
            grid = self.classical.predict_grid(image, self.rows, self.cols)
            backend = "classical"

        dirty_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                        if grid[r, c] >= self.dirty_threshold]
        coverage_pct = float(grid.mean() * 100)
        return DetectionResult(grid=grid, backend=backend, dirty_cells=dirty_cells, coverage_pct=coverage_pct)

    def predict_ensemble(self, image: np.ndarray) -> DetectionResult:
        """Averages CNN + classical predictions when both are available —
        the "cross-checking sensor readings" strategy from the concept
        design, applied to vision instead of sensors."""
        classical_grid = self.classical.predict_grid(image, self.rows, self.cols)
        if self.cnn is not None:
            cnn_grid = self.cnn.predict_grid(image)
            grid = 0.5 * classical_grid + 0.5 * cnn_grid
            backend = "ensemble(cnn+classical)"
        else:
            grid = classical_grid
            backend = "classical"
        dirty_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                        if grid[r, c] >= self.dirty_threshold]
        coverage_pct = float(grid.mean() * 100)
        return DetectionResult(grid=grid, backend=backend, dirty_cells=dirty_cells, coverage_pct=coverage_pct)
