"""The Orchestrator: ties computer vision, sensors and the water loop
together every wash cycle. The concept design describes this exact
component and is candid that it "still needs more research" beyond simple
threshold/comparison rules — so, true to that scope, this is a transparent
rule engine (not a black box), which also makes every decision explainable
for an environmental audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.orchestrator.spray_controller import SprayCommand, SprayController
from src.sensors.pump_health import PumpHealthAlert, PumpHealthMonitor
from src.sensors.simulator import PumpReading, SensorFleet, WaterQualityReading
from src.sensors.water_quality import GateDecision, WaterQualityGate
from src.utils.config import SystemConfig
from src.vision.detector import DetectionResult, StainDetector
from src.water_system.recycling_loop import RecyclingCycleResult, WaterRecyclingLoop


@dataclass
class CycleResult:
    tick: int
    vision: DetectionResult
    spray_commands: list[SprayCommand]
    water_reading: WaterQualityReading
    water_gate: GateDecision
    pump_readings: list[PumpReading]
    pump_alerts: list[PumpHealthAlert]
    recycling: RecyclingCycleResult
    alerts: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, config: SystemConfig, seed: int | None = None):
        self.config = config
        self.detector = StainDetector(
            rows=config.grid.rows, cols=config.grid.cols, dirty_threshold=config.grid.dirty_threshold
        )
        self.spray_controller = SprayController(dirty_threshold=config.grid.dirty_threshold)
        self.water_gate = WaterQualityGate(config.water_quality)
        self.sensors = SensorFleet(config, seed=seed)
        self.pump_monitors = {
            pump.pump_id: PumpHealthMonitor(pump.pump_id, config.pump, seed=seed or 0)
            for pump in self.sensors.pumps
        }
        self.recycling_loop = WaterRecyclingLoop(config.water_system)
        self.log: list[CycleResult] = []

    def run_cycle(self, tick: int, hull_image: np.ndarray) -> CycleResult:
        alerts: list[str] = []

        # 1. Find the dirt (computer vision).
        vision = self.detector.predict_ensemble(hull_image)
        spray_commands = self.spray_controller.plan(vision.grid)
        spray_fraction = self.spray_controller.spray_fraction(spray_commands)

        # 2. Watch the pumps.
        water_reading, pump_readings = self.sensors.read(tick)
        pump_alerts = []
        for reading in pump_readings:
            alert = self.pump_monitors[reading.pump_id].evaluate(reading)
            pump_alerts.append(alert)
            if alert.is_anomaly:
                alerts.append(f"pump health: {reading.pump_id} anomalous "
                               f"(pressure={reading.pressure_bar}bar, flow={reading.flow_lpm}Lpm, "
                               f"score={alert.anomaly_score})")

        # 3. Watch the water / gate reuse.
        gate = self.water_gate.evaluate(water_reading)
        if not gate.safe_to_reuse:
            alerts.append(f"water quality gate FAILED: {'; '.join(gate.reasons)}")

        # 4. Run the recycling loop with the targeted-spray volume.
        recycling = self.recycling_loop.run_cycle(tick, spray_fraction, gate.safe_to_reuse)

        result = CycleResult(
            tick=tick, vision=vision, spray_commands=spray_commands, water_reading=water_reading,
            water_gate=gate, pump_readings=pump_readings, pump_alerts=pump_alerts,
            recycling=recycling, alerts=alerts,
        )
        self.log.append(result)
        return result

    def summary(self) -> dict:
        n = len(self.log)
        if n == 0:
            return {}
        n_anomalies = sum(1 for r in self.log for a in r.pump_alerts if a.is_anomaly)
        n_gate_fails = sum(1 for r in self.log if not r.water_gate.safe_to_reuse)
        avg_coverage = float(np.mean([r.vision.coverage_pct for r in self.log]))
        return {
            "cycles_run": n,
            "avg_hull_dirtiness_pct": round(avg_coverage, 1),
            "pump_anomalies_flagged": n_anomalies,
            "water_quality_gate_failures": n_gate_fails,
            "water_recycled_pct": self.recycling_loop.recycled_rate_pct,
            "water_saved_vs_blanket_wash_pct": self.recycling_loop.water_saved_vs_blanket_pct,
            "vision_backend": self.log[-1].vision.backend,
        }
