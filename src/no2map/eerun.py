"""Earth Engine session initialization for all pipeline entrypoints.

Every script/module that talks to Earth Engine initializes through this one
function so the project id (decision D5: ONE project) and auth state are
handled in exactly one place."""
from __future__ import annotations

import ee


def initialize_ee(cfg: dict | None = None) -> dict:
    """Initialize Earth Engine with the configured project; returns the config.

    Pass a config dict to skip re-loading (config.load_config is cached)."""
    if cfg is None:
        from no2map.config import load_config

        cfg = load_config()
    ee.Initialize(project=cfg["quota"]["project_id"])
    return cfg
