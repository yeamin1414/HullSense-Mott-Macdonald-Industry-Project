"""Lightweight structured logging shared across the system.

Every orchestrator decision is written here so the run produces an auditable
record — matching the "Closed-loop proof: logged sensor data provides an
auditable record for environmental reporting" requirement from the concept
design.
"""
from __future__ import annotations

import csv
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def get_logger(name: str = "boat_wash_ai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class AuditLog:
    """Appends flattened records (dataclasses or dicts) to an in-memory list
    and can dump the run to CSV for environmental / compliance reporting."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def add(self, record: Any) -> None:
        if is_dataclass(record):
            record = asdict(record)
        self._records.append(_flatten(record))

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            path.write_text("")
            return path
        fieldnames = sorted({k for r in self._records for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)
        return path

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)


def _flatten(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
            for i, item in enumerate(v):
                items.update(_flatten(item, f"{key}[{i}]"))
        else:
            items[key] = v
    return items
