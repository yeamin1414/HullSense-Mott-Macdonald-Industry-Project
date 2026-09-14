"""Pump-health anomaly detection: "catch a failing pump early" from the
concept design, implemented as genuine unsupervised ML (IsolationForest)
trained on simulated normal operation, with a dependency-free statistical
fallback (rolling z-score) so the system still runs without scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sensors.simulator import PumpReading
from src.utils.config import PumpThresholds

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


@dataclass
class PumpHealthAlert:
    pump_id: str
    tick: int
    is_anomaly: bool
    anomaly_score: float
    method: str


def _simulate_normal_operation(thresholds: PumpThresholds, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pressure = rng.normal(thresholds.pressure_nominal_bar, 0.15, n)
    flow = rng.normal(thresholds.flow_nominal_lpm, 1.5, n)
    energy = rng.normal(thresholds.energy_nominal_kw, 0.08, n)
    return np.column_stack([pressure, flow, energy])


class PumpHealthMonitor:
    """One monitor per pump. Fits an IsolationForest on simulated healthy
    telemetry at start-up (no real historical data required), then scores
    each new reading live. Falls back to a 3-sigma rolling z-score rule if
    scikit-learn isn't installed."""

    def __init__(self, pump_id: str, thresholds: PumpThresholds, seed: int = 0):
        self.pump_id = pump_id
        self.t = thresholds
        self.history: list[np.ndarray] = []
        self.method = "isolation_forest" if SKLEARN_AVAILABLE else "zscore"

        if SKLEARN_AVAILABLE:
            training_data = _simulate_normal_operation(thresholds, n=300, seed=seed)
            self.model = IsolationForest(
                n_estimators=100, contamination=thresholds.anomaly_contamination, random_state=seed
            )
            self.model.fit(training_data)
        else:
            self.model = None

    def evaluate(self, reading: PumpReading) -> PumpHealthAlert:
        vec = np.array([[reading.pressure_bar, reading.flow_lpm, reading.energy_kw]])
        self.history.append(vec[0])

        if self.model is not None:
            pred = self.model.predict(vec)[0]  # -1 anomaly, 1 normal
            score = float(-self.model.score_samples(vec)[0])  # higher = more anomalous
            is_anomaly = pred == -1
        else:
            is_anomaly, score = self._zscore_fallback()

        return PumpHealthAlert(self.pump_id, reading.tick, bool(is_anomaly), round(score, 4), self.method)

    def _zscore_fallback(self) -> tuple[bool, float]:
        arr = np.array(self.history)
        if len(arr) < 8:
            return False, 0.0
        mean, std = arr[:-1].mean(axis=0), arr[:-1].std(axis=0) + 1e-6
        z = np.abs((arr[-1] - mean) / std)
        score = float(z.max())
        return score > 3.0, score
