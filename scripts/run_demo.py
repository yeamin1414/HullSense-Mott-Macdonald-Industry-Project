"""End-to-end demo: runs the full Boat Wash AI System for N wash cycles —
computer vision stain detection, sensor simulation, ML pump-health
monitoring, the rule-based orchestrator, and the closed-loop water system —
then prints a report and saves a visualisation + CSV audit log.

Usage:
    python scripts/run_demo.py --cycles 10
    python scripts/run_demo.py --cycles 10 --inject-fault
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.orchestrator.decision_engine import Orchestrator
from src.utils.config import DEFAULT_CONFIG
from src.utils.logger import AuditLog, get_logger
from src.vision.synthetic_data import generate_hull_image

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
logger = get_logger()


def run(cycles: int, inject_fault: bool, seed: int) -> None:
    config = DEFAULT_CONFIG
    orchestrator = Orchestrator(config, seed=seed)
    rng = np.random.default_rng(seed)
    audit = AuditLog()

    if inject_fault:
        orchestrator.sensors.inject_contamination(tick=max(1, cycles // 3))
        orchestrator.sensors.degrade_pump("pump_A", from_tick=max(1, cycles // 2))
        logger.info(f"Fault injection armed: contamination at tick {cycles // 3}, "
                     f"pump_A degrading from tick {cycles // 2}")

    logger.info(f"Vision backend in use: {orchestrator.detector.backend}")
    logger.info(f"Running {cycles} wash cycles...\n")

    for tick in range(1, cycles + 1):
        hull_image = generate_hull_image(size=config.grid.image_size, rng=rng).image
        result = orchestrator.run_cycle(tick, hull_image)
        audit.add(result)

        status = "OK" if not result.alerts else "ALERT"
        logger.info(
            f"tick {tick:02d} | dirt={result.vision.coverage_pct:5.1f}% "
            f"({len(result.vision.dirty_cells)} cells) | spray_used="
            f"{result.recycling.volume_used_l:6.1f}L | reused={result.recycling.reused!s:5} | {status}"
        )
        for a in result.alerts:
            logger.warning(f"  -> {a}")

    _print_summary(orchestrator)
    _save_outputs(orchestrator, audit)


def _print_summary(orchestrator: Orchestrator) -> None:
    summary = orchestrator.summary()
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Run summary")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for k, v in summary.items():
            table.add_row(k, str(v))
        console.print(table)
    else:
        print("\n=== Run summary ===")
        for k, v in summary.items():
            print(f"  {k:35s} {v}")


def _save_outputs(orchestrator: Orchestrator, audit) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = audit.to_csv(OUT_DIR / "run_log.csv")
    logger.info(f"Audit log written -> {csv_path}")

    log = orchestrator.log
    ticks = [r.tick for r in log]
    dirt = [r.vision.coverage_pct for r in log]
    tank = [r.recycling.tank_level_l for r in log]
    used = [r.recycling.volume_used_l for r in log]
    pressure_a = [r.pump_readings[0].pressure_bar for r in log]
    anomaly_a = [1 if r.pump_alerts[0].is_anomaly else 0 for r in log]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(ticks, dirt, marker="o", color="firebrick")
    axes[0].set_ylabel("Hull dirtiness %")
    axes[0].set_title("Boat Wash AI System — demo run")

    axes[1].plot(ticks, tank, marker="o", color="steelblue", label="tank level (L)")
    axes[1].bar(ticks, used, alpha=0.35, color="gray", label="volume used (L)")
    axes[1].set_ylabel("Water (L)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(ticks, pressure_a, marker="o", color="darkorange", label="pump_A pressure (bar)")
    anomaly_ticks = [t for t, a in zip(ticks, anomaly_a) if a]
    if anomaly_ticks:
        axes[2].scatter(anomaly_ticks, [pressure_a[ticks.index(t)] for t in anomaly_ticks],
                         color="red", zorder=5, label="anomaly flagged")
    axes[2].set_ylabel("Pump pressure (bar)")
    axes[2].set_xlabel("wash cycle")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig_path = OUT_DIR / "run_report.png"
    fig.savefig(fig_path, dpi=130)
    logger.info(f"Report chart written -> {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--inject-fault", action="store_true",
                         help="inject a water-contamination event and a slowly-degrading pump")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.cycles, args.inject_fault, args.seed)
