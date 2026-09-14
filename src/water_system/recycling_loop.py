"""Closed-loop water recycling state machine — implements the 4-step loop
from the concept design: wash runs -> drains to bund -> sensors check
quality -> back to tank (or held for disposal if it fails the gate)."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.config import WaterSystemConfig


@dataclass
class RecyclingCycleResult:
    tick: int
    volume_used_l: float
    volume_recycled_l: float
    volume_held_for_disposal_l: float
    tank_level_l: float
    reused: bool


@dataclass
class WaterRecyclingLoop:
    config: WaterSystemConfig
    tank_level_l: float = field(init=False)
    total_used_l: float = field(default=0.0, init=False)
    total_recycled_l: float = field(default=0.0, init=False)
    total_held_l: float = field(default=0.0, init=False)
    total_baseline_blanket_l: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.tank_level_l = self.config.tank_capacity_l

    def run_cycle(self, tick: int, spray_fraction: float, water_safe_to_reuse: bool) -> RecyclingCycleResult:
        """`spray_fraction` in [0, 1] is the fraction of a full blanket wash
        actually needed this cycle (driven by the CV targeted-spray decision)."""
        volume_used = self.config.wash_volume_per_cycle_l * spray_fraction
        self.tank_level_l = max(0.0, self.tank_level_l - volume_used)

        if water_safe_to_reuse:
            volume_recycled = volume_used
            volume_held = 0.0
            self.tank_level_l = min(self.config.tank_capacity_l, self.tank_level_l + volume_recycled)
        else:
            volume_recycled = 0.0
            volume_held = volume_used
            # Water fails the quality gate: it is held for controlled disposal
            # rather than re-entering the tank; top up from mains instead.
            self.tank_level_l = min(self.config.tank_capacity_l, self.tank_level_l + volume_used * 0.0)

        self.total_used_l += volume_used
        self.total_recycled_l += volume_recycled
        self.total_held_l += volume_held
        self.total_baseline_blanket_l += self.config.blanket_wash_volume_l

        return RecyclingCycleResult(
            tick=tick,
            volume_used_l=round(volume_used, 1),
            volume_recycled_l=round(volume_recycled, 1),
            volume_held_for_disposal_l=round(volume_held, 1),
            tank_level_l=round(self.tank_level_l, 1),
            reused=water_safe_to_reuse,
        )

    @property
    def recycled_rate_pct(self) -> float:
        if self.total_used_l == 0:
            return 100.0
        return round(100 * self.total_recycled_l / self.total_used_l, 1)

    @property
    def water_saved_vs_blanket_pct(self) -> float:
        if self.total_baseline_blanket_l == 0:
            return 0.0
        saved = self.total_baseline_blanket_l - self.total_used_l
        return round(100 * saved / self.total_baseline_blanket_l, 1)
