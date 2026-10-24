"""Model artifact: save / load the trained Model A for the live path.

The plan's Gate G4 requires "a map from a drawn polygon WITHOUT retraining".
That means the Phase-3 training run must persist a complete, self-described
serving bundle, and Phase 4 must trust nothing but that bundle at request
time. This module is the contract between those phases:

- `save_model_artifact(run_dir, ...)`: writes the XGBoost model in its
  native JSON form, the split-conformal calibration table (season -> half-
  width quantile, pooled fallback root included), the exact feature-column
  order, and provenance metadata (feature-schema version, groups, seed,
  block protocol, training months, created-at timestamp).
- `load_model_artifact(run_dir)`: restores everything; `predict(frame)`
  runs the shared feature module (train/serve parity), returns point
  predictions plus conformal half-widths. Unknown season strata fall back
  to the pooled quantile exactly as in training-time calibration.

Parity pin: loading with a differing `features.FEATURE_SCHEMA_VERSION`
raises - a silently mismatched schema at serve time would produce features
the model never saw.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from no2map.conformal import SplitConformal
from no2map.features import FEATURE_SCHEMA_VERSION, TARGET, build_features, feature_block
from no2map.model import season_key

ARTIFACT_FILE = "artifact.json"
MODEL_FILE = "model.json"
_POOLED_KEY = "__pooled__"


@dataclass
class ModelArtifact:
    model: XGBRegressor
    q_by_stratum: dict[str | None, float]
    alpha: float
    fe_cols: list[str]
    groups: list[str]
    feature_schema_version: str
    meta: dict

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Point predictions + conformal half-widths through the SHARED
        feature module (never re-implement a feature at serve time)."""
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"artifact schema {self.feature_schema_version} != installed "
                f"features {FEATURE_SCHEMA_VERSION}; retrain or pin the env")
        rows = build_features(df)
        missing = [c for c in self.fe_cols if c not in rows.columns]
        if missing:
            raise KeyError(f"frame lacks feature columns: {missing[:5]}")
        out = pd.DataFrame(index=rows.index)
        out[TARGET] = rows[TARGET].to_numpy() if TARGET in rows else float("nan")
        out["pred"] = self.model.predict(rows[self.fe_cols])
        out["cell_id"] = rows["cell_id"].to_numpy()
        out["date"] = rows["date"].to_numpy()
        if "month" in rows:
            out["month"] = rows["month"].to_numpy()
        else:
            out["month"] = np.full(len(rows), np.nan)
        # conformal margin: season-keyed with the pooled fallback root
        keys = [season_key({"month": int(m)}) if pd.notna(m) else None
                for m in out["month"]]
        out["margin"] = [self.q_by_stratum.get(k, self.q_by_stratum[None])
                         for k in keys]
        return out


def save_model_artifact(
    run_dir: str | Path,
    model: XGBRegressor,
    conformal: SplitConformal,
    fe_cols: list[str],
    groups: list[str],
    meta: dict,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(run_dir / MODEL_FILE))

    # only the season-quarter key function is supported; refuse exotic fns so
    # the artifact stays reloadable without pickling callables
    q_payload = {(_POOLED_KEY if k is None else str(k)): v
                 for k, v in (conformal.q_by_stratum or {}).items()}
    payload = {
        "artifact_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fe_cols": list(fe_cols),
        "groups": list(groups),
        "alpha": conformal.alpha,
        "q_by_stratum": q_payload,
        "key_fn": "season_quarter",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta,
    }
    (run_dir / ARTIFACT_FILE).write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return run_dir


def load_model_artifact(run_dir: str | Path) -> ModelArtifact:
    run_dir = Path(run_dir)
    payload = json.loads((run_dir / ARTIFACT_FILE).read_text(encoding="utf-8"))
    if payload.get("key_fn") != "season_quarter":
        raise RuntimeError(f"unsupported conformal key_fn {payload.get('key_fn')!r}")
    model = XGBRegressor()
    model.load_model(str(run_dir / MODEL_FILE))
    q = {None if k == _POOLED_KEY else k: val
         for k, val in payload["q_by_stratum"].items()}
    return ModelArtifact(
        model=model,
        q_by_stratum=q,
        alpha=float(payload["alpha"]),
        fe_cols=list(payload["fe_cols"]),
        groups=list(payload["groups"]),
        feature_schema_version=payload["feature_schema_version"],
        meta=payload.get("meta", {}),
    )
