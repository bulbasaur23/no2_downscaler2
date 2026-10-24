"""Configuration loading.

Config values are *locked decisions* from the plan (section 3) plus Stage-0
placeholders. Loading is centralised so live inference and the training
pipeline read the SAME numbers (one shared source of truth, see parity test).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ("grid", "observed_cell_rule", "study_areas", "sources", "weather", "quota")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"config missing required sections: {missing}")
    if abs(cfg["grid"]["cell_deg"] - 0.05) > 1e-9:
        raise ValueError("grid.cell_deg is locked at 0.05 (decision D8); regridding needs a plan change")
    return cfg


def study_area(cfg: dict, name: str) -> dict:
    areas = cfg["study_areas"]
    if name not in areas:
        raise KeyError(f"unknown study area {name!r}; available: {sorted(areas)}")
    return areas[name]
