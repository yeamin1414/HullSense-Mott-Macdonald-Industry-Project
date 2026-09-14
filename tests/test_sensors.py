import numpy as np

from src.sensors.pump_health import PumpHealthMonitor
from src.sensors.simulator import PumpSensor, WaterQualitySensor
from src.sensors.water_quality import WaterQualityGate
from src.utils.config import DEFAULT_CONFIG


def test_water_quality_sensor_within_plausible_range():
    sensor = WaterQualitySensor(DEFAULT_CONFIG.water_quality, rng=np.random.default_rng(0))
    reading = sensor.read(tick=1)
    assert 0 <= reading.turbidity_ntu < 20
    assert 0 <= reading.tds_ppm < 3000


def test_water_quality_gate_passes_clean_reading():
    gate = WaterQualityGate(DEFAULT_CONFIG.water_quality)
    sensor = WaterQualitySensor(DEFAULT_CONFIG.water_quality, fault_probability=0.0, rng=np.random.default_rng(1))
    reading = sensor.read(tick=1)
    decision = gate.evaluate(reading)
    assert decision.safe_to_reuse is True
    assert decision.reasons == []


def test_water_quality_gate_blocks_contamination_event():
    gate = WaterQualityGate(DEFAULT_CONFIG.water_quality)
    sensor = WaterQualitySensor(DEFAULT_CONFIG.water_quality, fault_probability=0.0, rng=np.random.default_rng(2))
    reading = sensor.read(tick=1, contamination_event=True)
    decision = gate.evaluate(reading)
    assert decision.safe_to_reuse is False
    assert len(decision.reasons) > 0


def test_pump_sensor_degrades_over_time():
    sensor = PumpSensor("pump_A", DEFAULT_CONFIG.pump, fault_probability=0.0,
                         degrade_after_tick=5, rng=np.random.default_rng(3))
    early = sensor.read(tick=1)
    late = sensor.read(tick=13)
    assert late.pressure_bar < early.pressure_bar
    assert late.flow_lpm < early.flow_lpm


def test_pump_health_monitor_flags_extreme_reading():
    monitor = PumpHealthMonitor("pump_A", DEFAULT_CONFIG.pump, seed=0)
    from src.sensors.simulator import PumpReading

    normal = PumpReading(tick=1, pump_id="pump_A", pressure_bar=4.0, flow_lpm=45.0, energy_kw=2.2)
    extreme = PumpReading(tick=2, pump_id="pump_A", pressure_bar=0.5, flow_lpm=5.0, energy_kw=6.0)

    normal_alert = monitor.evaluate(normal)
    extreme_alert = monitor.evaluate(extreme)
    assert extreme_alert.anomaly_score >= normal_alert.anomaly_score
    assert extreme_alert.is_anomaly is True
