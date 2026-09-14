import numpy as np

from src.orchestrator.decision_engine import Orchestrator
from src.utils.config import DEFAULT_CONFIG
from src.vision.synthetic_data import generate_hull_image


def test_orchestrator_runs_cycles_and_logs():
    orchestrator = Orchestrator(DEFAULT_CONFIG, seed=0)
    rng = np.random.default_rng(0)
    for tick in range(1, 6):
        image = generate_hull_image(size=DEFAULT_CONFIG.grid.image_size, rng=rng).image
        result = orchestrator.run_cycle(tick, image)
        assert result.vision.grid.shape == (DEFAULT_CONFIG.grid.rows, DEFAULT_CONFIG.grid.cols)
        assert 0.0 <= result.vision.coverage_pct <= 100.0
        assert len(result.pump_readings) == 2
        assert len(result.pump_alerts) == 2

    assert len(orchestrator.log) == 5
    summary = orchestrator.summary()
    assert summary["cycles_run"] == 5


def test_orchestrator_reacts_to_contamination_event():
    orchestrator = Orchestrator(DEFAULT_CONFIG, seed=1)
    orchestrator.sensors.inject_contamination(tick=1)
    rng = np.random.default_rng(1)
    image = generate_hull_image(size=DEFAULT_CONFIG.grid.image_size, rng=rng).image
    result = orchestrator.run_cycle(1, image)

    assert result.water_gate.safe_to_reuse is False
    assert result.recycling.reused is False
    assert any("water quality gate" in a for a in result.alerts)


def test_orchestrator_spray_fraction_never_exceeds_one():
    orchestrator = Orchestrator(DEFAULT_CONFIG, seed=2)
    rng = np.random.default_rng(2)
    for tick in range(1, 4):
        image = generate_hull_image(size=DEFAULT_CONFIG.grid.image_size, num_stains=8, rng=rng).image
        result = orchestrator.run_cycle(tick, image)
        fraction = result.recycling.volume_used_l / DEFAULT_CONFIG.water_system.wash_volume_per_cycle_l
        assert 0.0 <= fraction <= 1.0
