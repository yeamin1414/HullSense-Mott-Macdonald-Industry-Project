from src.utils.config import DEFAULT_CONFIG
from src.water_system.recycling_loop import WaterRecyclingLoop


def test_full_reuse_keeps_tank_level_stable():
    loop = WaterRecyclingLoop(DEFAULT_CONFIG.water_system)
    start_level = loop.tank_level_l
    result = loop.run_cycle(tick=1, spray_fraction=0.5, water_safe_to_reuse=True)
    assert result.reused is True
    assert result.volume_held_for_disposal_l == 0.0
    assert loop.tank_level_l == start_level  # used, then fully recycled back


def test_failed_gate_holds_water_and_drains_tank():
    loop = WaterRecyclingLoop(DEFAULT_CONFIG.water_system)
    start_level = loop.tank_level_l
    result = loop.run_cycle(tick=1, spray_fraction=1.0, water_safe_to_reuse=False)
    assert result.reused is False
    assert result.volume_held_for_disposal_l > 0
    assert loop.tank_level_l < start_level


def test_spray_fraction_zero_uses_no_water():
    loop = WaterRecyclingLoop(DEFAULT_CONFIG.water_system)
    result = loop.run_cycle(tick=1, spray_fraction=0.0, water_safe_to_reuse=True)
    assert result.volume_used_l == 0.0


def test_water_saved_metric_improves_with_lower_spray_fraction():
    loop = WaterRecyclingLoop(DEFAULT_CONFIG.water_system)
    for tick in range(1, 6):
        loop.run_cycle(tick, spray_fraction=0.2, water_safe_to_reuse=True)
    assert loop.water_saved_vs_blanket_pct > 0
