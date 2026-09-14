"""Maps the CV dirtiness grid onto physical spray commands for the two
powered spray bars + hand-wand zone described in the concept design's
section view — "pumps focus water on the dirtiest sections, which saves
water"."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SprayCommand:
    row: int
    col: int
    bar: str  # "bar_1" (left half) | "bar_2" (right half)
    pressure_pct: int  # 0, 50 (light rinse) or 100 (full pressure)


class SprayController:
    def __init__(self, dirty_threshold: float = 0.35, light_rinse_threshold: float = 0.12):
        self.dirty_threshold = dirty_threshold
        self.light_rinse_threshold = light_rinse_threshold

    def plan(self, grid: np.ndarray) -> list[SprayCommand]:
        rows, cols = grid.shape
        mid = cols / 2
        commands: list[SprayCommand] = []
        for r in range(rows):
            for c in range(cols):
                score = float(grid[r, c])
                if score >= self.dirty_threshold:
                    pressure = 100
                elif score >= self.light_rinse_threshold:
                    pressure = 50
                else:
                    pressure = 0
                bar = "bar_1" if c < mid else "bar_2"
                commands.append(SprayCommand(r, c, bar, pressure))
        return commands

    @staticmethod
    def spray_fraction(commands: list[SprayCommand]) -> float:
        """Fraction of a full blanket wash this plan actually uses — pressure
        weighted so a 50%-rinse cell costs half of a 100% cell."""
        if not commands:
            return 0.0
        return sum(c.pressure_pct for c in commands) / (100 * len(commands))
