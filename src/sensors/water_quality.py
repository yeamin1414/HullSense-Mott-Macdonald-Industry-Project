"""Water-quality gate: the simple threshold/comparison rules described as
"Sensors check quality ... clean water is cleared for reuse" in the concept
design's closed-loop diagram."""
from __future__ import annotations

from dataclasses import dataclass

from src.sensors.simulator import WaterQualityReading
from src.utils.config import WaterQualityThresholds


@dataclass
class GateDecision:
    safe_to_reuse: bool
    reasons: list[str]


class WaterQualityGate:
    def __init__(self, thresholds: WaterQualityThresholds):
        self.t = thresholds

    def evaluate(self, reading: WaterQualityReading) -> GateDecision:
        reasons: list[str] = []

        if not (self.t.ph_min <= reading.ph <= self.t.ph_max):
            reasons.append(f"pH {reading.ph} outside [{self.t.ph_min}, {self.t.ph_max}]")
        if reading.turbidity_ntu > self.t.turbidity_max_ntu:
            reasons.append(f"turbidity {reading.turbidity_ntu} NTU > {self.t.turbidity_max_ntu}")
        if not (self.t.chlorine_min_ppm <= reading.chlorine_ppm <= self.t.chlorine_max_ppm):
            reasons.append(f"chlorine {reading.chlorine_ppm} ppm outside "
                            f"[{self.t.chlorine_min_ppm}, {self.t.chlorine_max_ppm}]")
        if reading.tds_ppm > self.t.tds_max_ppm:
            reasons.append(f"TDS {reading.tds_ppm} ppm > {self.t.tds_max_ppm}")
        if reading.faulty:
            reasons.append("sensor flagged a fault on this reading")

        return GateDecision(safe_to_reuse=(len(reasons) == 0), reasons=reasons)
