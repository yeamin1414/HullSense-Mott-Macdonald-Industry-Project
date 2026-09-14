"""Realistic sensor simulators for water quality and pump telemetry.

Standing in for the physical instruments named in the concept design (Hanna
HI98194 water-quality probe, IFM SF5700 flow/pressure sensor): Gaussian
sensor noise, slow drift, and a small chance of an outright fault — modelling
the documented risk that "sensors wear out in the salty air and give wrong
readings."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.config import PumpThresholds, WaterQualityThresholds


@dataclass
class WaterQualityReading:
    tick: int
    ph: float
    turbidity_ntu: float
    chlorine_ppm: float
    tds_ppm: float
    faulty: bool = False


@dataclass
class PumpReading:
    tick: int
    pump_id: str
    pressure_bar: float
    flow_lpm: float
    energy_kw: float
    faulty: bool = False


class WaterQualitySensor:
    def __init__(self, thresholds: WaterQualityThresholds, fault_probability: float = 0.03,
                 rng: np.random.Generator | None = None):
        self.t = thresholds
        self.fault_probability = fault_probability
        self.rng = rng or np.random.default_rng()
        self._drift = 0.0

    def read(self, tick: int, contamination_event: bool = False) -> WaterQualityReading:
        self._drift += self.rng.normal(0, 0.01)
        ph = 7.4 + self._drift + self.rng.normal(0, 0.12)
        turbidity = max(0.0, 1.8 + self.rng.normal(0, 0.5))
        chlorine = max(0.0, 1.2 + self.rng.normal(0, 0.15))
        tds = max(0.0, 420 + self.rng.normal(0, 30))

        if contamination_event:
            turbidity += self.rng.uniform(6, 14)
            tds += self.rng.uniform(300, 700)
            ph += self.rng.choice([-1, 1]) * self.rng.uniform(1.2, 2.0)

        faulty = bool(self.rng.random() < self.fault_probability)
        if faulty:
            # A stuck / drifted probe reports an implausible spike.
            ph += self.rng.choice([-1, 1]) * self.rng.uniform(2.5, 4.0)

        return WaterQualityReading(tick, round(ph, 2), round(turbidity, 2), round(chlorine, 2), round(tds, 1), faulty)


class PumpSensor:
    def __init__(self, pump_id: str, thresholds: PumpThresholds, fault_probability: float = 0.03,
                 degrade_after_tick: int | None = None, rng: np.random.Generator | None = None):
        self.pump_id = pump_id
        self.t = thresholds
        self.fault_probability = fault_probability
        self.degrade_after_tick = degrade_after_tick
        self.rng = rng or np.random.default_rng()

    def read(self, tick: int) -> PumpReading:
        pressure = self.t.pressure_nominal_bar + self.rng.normal(0, 0.15)
        flow = self.t.flow_nominal_lpm + self.rng.normal(0, 1.5)
        energy = self.t.energy_nominal_kw + self.rng.normal(0, 0.08)

        if self.degrade_after_tick is not None and tick >= self.degrade_after_tick:
            # Simulates a bearing/seal wearing out: pressure & flow sag,
            # energy draw creeps up to compensate — a classic predictive-
            # maintenance signature.
            progress = min(1.0, (tick - self.degrade_after_tick) / 8.0)
            pressure -= 1.1 * progress
            flow -= 12.0 * progress
            energy += 0.9 * progress

        faulty = bool(self.rng.random() < self.fault_probability)
        if faulty:
            pressure += self.rng.choice([-1, 1]) * self.rng.uniform(1.5, 2.5)

        return PumpReading(tick, self.pump_id, round(pressure, 2), round(flow, 2), round(energy, 2), faulty)


class SensorFleet:
    """Bundles the water-quality probe and both wash-bay pump sensors, and
    can be told to inject a contamination event or start a pump degrading —
    used by the demo script and dashboard to show the system reacting."""

    def __init__(self, config, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.water = WaterQualitySensor(config.water_quality, rng=rng)
        self.pumps = [
            PumpSensor("pump_A", config.pump, rng=rng),
            PumpSensor("pump_B", config.pump, rng=rng),
        ]
        self._contamination_tick: int | None = None

    def inject_contamination(self, tick: int) -> None:
        self._contamination_tick = tick

    def degrade_pump(self, pump_id: str, from_tick: int) -> None:
        for pump in self.pumps:
            if pump.pump_id == pump_id:
                pump.degrade_after_tick = from_tick

    def read(self, tick: int) -> tuple[WaterQualityReading, list[PumpReading]]:
        contamination = self._contamination_tick is not None and tick == self._contamination_tick
        water_reading = self.water.read(tick, contamination_event=contamination)
        pump_readings = [p.read(tick) for p in self.pumps]
        return water_reading, pump_readings
