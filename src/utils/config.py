"""Central configuration for the Boat Wash AI System.

All tunable thresholds live here so the rest of the codebase never hard-codes
a "magic number" — this mirrors how the thresholds were specified in the
Building R13 concept design (see docs/architecture.md).
"""
from dataclasses import dataclass, field


@dataclass
class HullGridConfig:
    rows: int = 4
    cols: int = 6
    image_size: int = 128  # square synthetic hull image, pixels
    dirty_threshold: float = 0.35  # section dirtiness score >= this => spray


@dataclass
class WaterQualityThresholds:
    ph_min: float = 6.5
    ph_max: float = 8.5
    turbidity_max_ntu: float = 5.0
    chlorine_min_ppm: float = 0.2
    chlorine_max_ppm: float = 4.0
    tds_max_ppm: float = 1000.0


@dataclass
class PumpThresholds:
    pressure_nominal_bar: float = 4.0
    pressure_tolerance_bar: float = 0.6
    flow_nominal_lpm: float = 45.0
    flow_tolerance_lpm: float = 6.0
    energy_nominal_kw: float = 2.2
    energy_tolerance_kw: float = 0.4
    anomaly_contamination: float = 0.06  # expected fraction of anomalous ticks


@dataclass
class WaterSystemConfig:
    tank_capacity_l: float = 5000.0
    wash_volume_per_cycle_l: float = 220.0
    blanket_wash_volume_l: float = 420.0  # baseline non-targeted hose-down volume


@dataclass
class SystemConfig:
    grid: HullGridConfig = field(default_factory=HullGridConfig)
    water_quality: WaterQualityThresholds = field(default_factory=WaterQualityThresholds)
    pump: PumpThresholds = field(default_factory=PumpThresholds)
    water_system: WaterSystemConfig = field(default_factory=WaterSystemConfig)
    random_seed: int = 42


DEFAULT_CONFIG = SystemConfig()
