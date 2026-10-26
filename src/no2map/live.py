"""Live polygon compute (Phase 4): EE fetch -> shared features -> artifact.

The parity contract from the plan (risk 10) is structural here: the live path
calls the SAME acquisition (`day_no2_cell_sums` -> `day_rows`) and
weather (`day_weather_stack`, which itself builds the 24/48 h lags) functions
that produced the training parquet, then the SAME feature module
(`features.build_features`) and the SAME schema-pinned artifact. No feature is
recomputed by a second implementation.

Flow for one request:
  1. validate the polygon and map it to 0.05 deg cells with EXACT overlap
     fractions (edge cells are weighted, never rounded away);
  2. fetch NO2 for the request date over exactly the polygon's cells, so each
     cell keeps its D6 state (observed / insufficient / no_overpass);
  3. fetch weather (+ lags) and join the cached statics;
  4. score every cell with the loaded artifact (point + conformal margin);
  5. assemble per-cell rows and a weighted summary. Observed cells report the
     observation; non-observed cells are labelled `reconstructed` (the plan's
     Phase-4 rule until Phase 6 sets thresholds).

EE calls are injected (`no2_fn`, `weather_fn`, `version_fn`) so the whole
assembly and labelling logic is testable without Earth Engine; the defaults
are the real EE functions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from no2map.artifact import ModelArtifact, load_model_artifact
from no2map.cache import CellCache, cell_key
from no2map.features import FEATURE_SCHEMA_VERSION, TARGET, build_features
from no2map.grid import cell_from_row_col, parse_cell_id
from no2map.polygon import (
    cells_for_polygon,
    polygon_bbox_buffered,
    resolution_warning,
    validate_polygon,
)

GRID_CELL_DEG = 0.05
WEATHER_SCALE_M = 11132.0
# D4: both ERA5 collections were measured ~6 days behind real time at Stage 0;
# a request date must not be newer than that or the features are fabricated.
DATE_CAP_LAG_DAYS = 6
STATIC_FEATURE_COLS = ["elevation_mean_m", "elevation_std_m", "elevation_min_m",
                       "elevation_max_m", "water_fraction"]
OBSERVED = "observed"


def latest_available_date(today: _date | None = None,
                          lag_days: int = DATE_CAP_LAG_DAYS) -> str:
    """Newest date the date picker may offer (D4 cap)."""
    today = today or _date.today()
    return (today - timedelta(days=lag_days)).isoformat()


def _cells_fc(cell_ids: list[str]):
    """GEE FeatureCollection covering exactly these grid cells."""
    import ee

    feats = []
    for cid in cell_ids:
        row, col = parse_cell_id(cid)
        c = cell_from_row_col(row, col)
        geom = ee.Geometry.Rectangle([c.lon_min, c.lat_min, c.lon_max, c.lat_max])
        feats.append(ee.Feature(geom, {"cell_id": cid}))
    return ee.FeatureCollection(feats)


def _ee_no2(city_cfg: dict, date: str, cell_ids: list[str]) -> list[dict]:
    from no2map.acquisition import (day_no2_cell_sums, day_rows,
                                    no2_projection_scale)

    scale = _ee_no2.scale  # cached process-wide (one EE round trip)
    if scale is None:
        scale = _ee_no2.scale = no2_projection_scale()
    sums = day_no2_cell_sums(city_cfg, date, _cells_fc(cell_ids), scale)
    # same row contract as the month exporter: an empty overpass day becomes
    # `no_overpass` rows for every cell (D6), never an empty frame
    return [{"date": date, **r} for r in day_rows(sums, cell_ids,
                                                  _ee_no2.threshold)]


_ee_no2.scale = None
_ee_no2.threshold = 0.5


def _ee_weather(city_cfg: dict, date: str, cell_ids: list[str]) -> list[dict]:
    from no2map.weather import day_weather_stack

    return day_weather_stack(date, WEATHER_SCALE_M, _cells_fc(cell_ids))


def _ee_versions(city_cfg: dict, date: str) -> list[str]:
    from no2map.acquisition import month_processor_versions

    return month_processor_versions(int(date[:4]), int(date[5:7]))


def load_statics(exports_dir: str | Path, city: str) -> pd.DataFrame:
    p = Path(exports_dir) / f"static_{city}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"statics cache missing for {city}: {p} (run a month export first)")
    df = pd.read_parquet(p)
    missing = [c for c in ["cell_id", *STATIC_FEATURE_COLS] if c not in df.columns]
    if missing:
        raise KeyError(f"statics cache lacks columns: {missing}")
    return df[["cell_id", *STATIC_FEATURE_COLS]]


def load_latest_artifact(runs_dir: str | Path, city: str) -> tuple[ModelArtifact, str]:
    """Newest run for this city that carries a complete artifact bundle.

    Refuses (rather than silently degrading) when the artifact's feature
    schema differs from the installed code: a live map from mismatched
    features would be confident nonsense."""
    runs = sorted(Path(runs_dir).glob(f"{city}_modelA_*"))
    complete = [d for d in runs if (d / "artifact.json").exists()
                and (d / "model.json").exists()]
    if not complete:
        raise FileNotFoundError(f"no complete model artifact for {city} in {runs_dir}")
    run_dir = complete[-1]
    art = load_model_artifact(run_dir)
    if art.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"artifact {run_dir.name} schema {art.feature_schema_version} != "
            f"installed {FEATURE_SCHEMA_VERSION}; retrain before serving")
    return art, run_dir.name


def load_g3_summary(exports_dir: str | Path, city: str) -> dict | None:
    """Compact G3 evidence for the app (G3 returned `honest_report`).

    The plan's fallback for a model that does not beat climatology is to keep
    serving, but label estimates as climatology-anchored and report the skill
    as it stands. The app therefore shows this block verbatim; a missing
    report is reported as missing rather than quietly omitted.
    """
    p = Path(exports_dir) / f"g3_report_{city}.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    b = rep.get("blocked", {}).get("seed_delta", {})
    c = rep.get("chronological", {}).get("seed_delta", {})
    verdict = rep.get("verdict", {})
    return {
        "g3_mode": verdict.get("g3_mode"),
        "generated_at": rep.get("generated_at"),
        "blocked": {
            "model_minus_climatology_rmse": b.get("delta_mean"),
            "delta_sd": b.get("delta_sd"),
            "beats_beyond_noise": verdict.get("beats_blocked_beyond_noise"),
        },
        "chronological": {
            "model_minus_climatology_rmse": c.get("delta_mean"),
            "delta_sd": c.get("delta_sd"),
            "beats_beyond_noise": verdict.get("beats_chronological_beyond_noise"),
        },
        "required_actions": verdict.get("required_actions", []),
        "note": verdict.get("note"),
        "report_path": str(p),
    }


@dataclass
class LivePredictor:
    city: str
    city_cfg: dict
    artifact: ModelArtifact
    statics: pd.DataFrame
    run_name: str = ""
    main_threshold: float = 0.5
    cache: CellCache | None = None
    g3: dict | None = None
    no2_fn: Callable = _ee_no2
    weather_fn: Callable = _ee_weather
    version_fn: Callable = _ee_versions
    _versions_memo: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # the default EE fetcher applies the D2 observed threshold client-side
        # (same pure function the export worker uses); a per-request threshold
        # would be a different rule, so it is fixed at construction
        if self.no2_fn is _ee_no2:
            _ee_no2.threshold = self.main_threshold

    def _versions(self, date: str) -> list[str]:
        """Processor versions for the date's month, memoized per process:
        an EE aggregate call per request would dominate the quota, and the
        value cannot change for a past month."""
        month = date[:7]
        if month not in self._versions_memo:
            self._versions_memo[month] = list(self.version_fn(self.city_cfg, date))
        return self._versions_memo[month]

    def _cache_keys(self, date: str, versions: list[str],
                    cell_ids: list[str]) -> dict[str, str]:
        if self.cache is None:
            return {}
        pv = ",".join(sorted(str(v) for v in versions))
        fv = self.artifact.feature_schema_version
        return {cid: cell_key(cid, date, fv, pv, self.run_name)
                for cid in cell_ids}

    def compute(self, polygon: list[tuple[float, float]], date: str) -> dict:
        info = validate_polygon(polygon)
        overlaps = cells_for_polygon(polygon)
        cell_ids = sorted(overlaps)
        window = polygon_bbox_buffered(polygon)
        versions = self._versions(date)

        keys = self._cache_keys(date, versions, cell_ids)
        if keys:
            hits = self.cache.get_many(list(keys.values()))
            if len(hits) == len(cell_ids):
                # full hit: no Earth Engine work at all
                rows = [hits[keys[cid]] for cid in cell_ids]
                return self._response(info, overlaps, window, date, versions,
                                      rows, cache_hit=True)

        df = self.assemble(date, cell_ids)
        rows = self._rows(df, overlaps)

        if keys:
            pv = ",".join(sorted(str(v) for v in versions))
            fv = self.artifact.feature_schema_version
            self.cache.put_many([
                (keys[r["cell_id"]], r["cell_id"], date, fv, pv,
                 self.run_name, r) for r in rows])
        return self._response(info, overlaps, window, date, versions,
                              rows, cache_hit=False)

    def assemble(self, date: str, cell_ids: list[str]) -> pd.DataFrame:
        """Request frame for these cells/dates: NO2 state + weather/lags +
        statics, merged on (cell_id, date).

        Public because it is the G4 parity hook: feeding it the same raw
        values that training consumed must produce the same feature frame.
        It calls the shared acquisition/weather fetchers and never derives a
        feature itself - `artifact.predict` applies `build_features` once."""
        no2_rows = self.no2_fn(self.city_cfg, date, cell_ids)
        weather_rows = self.weather_fn(self.city_cfg, date, cell_ids)

        no2 = pd.DataFrame(no2_rows, columns=["cell_id", "date", TARGET, "state",
                                              "reason_code", "n_valid_fraction",
                                              "subpixel_spread_umol_m2",
                                              "n_orbit_overlap"])
        # a fetcher that omits the date still keys rows by the request date
        if not no2.empty and no2["date"].isna().all():
            no2["date"] = date
        # start from the polygon's cell set: a cell the fetch never returned
        # must surface as `unavailable`, not silently shrink the map
        df = pd.DataFrame({"cell_id": cell_ids, "date": date}).merge(
            no2, on=["cell_id", "date"], how="left")
        if len(df) != len(cell_ids):
            raise RuntimeError(f"NO2 join produced {len(df)} rows for "
                               f"{len(cell_ids)} polygon cells")
        wx = pd.DataFrame(weather_rows)
        if not wx.empty:
            df = df.merge(wx, on=["cell_id", "date"], how="left",
                          suffixes=("", "_wx"))
            if len(df) != len(cell_ids):
                raise RuntimeError(f"weather join produced {len(df)} rows for "
                                   f"{len(cell_ids)} polygon cells")
        # statics define the cell's static features; polygon cells always exist
        # in the city grid, so a miss is a configuration bug, not a data gap
        df = df.merge(self.statics, on="cell_id", how="left")
        if len(df) != len(cell_ids):
            raise RuntimeError(f"cell join produced {len(df)} rows for "
                               f"{len(cell_ids)} polygon cells")

        # cells the fetch did not return at all: never fabricate a state
        df["state"] = df["state"].fillna("unavailable")
        df.loc[df["state"] == "unavailable", "reason_code"] = "fetch_missing"

        preds = self.artifact.predict(df)
        df["pred"] = preds["pred"].to_numpy()
        df["margin"] = preds["margin"].to_numpy()
        # G3 was reported honestly: the climatology anchor rides along so the
        # UI can show "model estimate" beside "what this cell usually is".
        df["climatology"] = preds["climatology"].to_numpy()

        observed = df["state"].to_numpy() == OBSERVED
        df["value"] = np.where(observed, df[TARGET].to_numpy(dtype=float),
                               df["pred"].to_numpy(dtype=float))
        df["source"] = np.where(observed, "observed", "reconstructed")
        return df

    def _response(self, info: dict, overlaps: dict[str, float], window: dict,
                  date: str, versions: list[str], rows: list[dict],
                  cache_hit: bool) -> dict:
        weights = np.array([float(overlaps[r["cell_id"]]) for r in rows],
                           dtype=float)
        warnings = list(info["warnings"])
        w = resolution_warning(len(rows),
                               float(weights.min()) if len(weights) else 0.0)
        if w:
            warnings.append(w)
        months = self.artifact.meta.get("months") or []
        if date[:7] in months:
            warnings.append(
                "requested date falls inside the model's training window: "
                "reconstructed values are in-sample and likely optimistic; "
                "use observed cells and the climatology anchor as the "
                "trustworthy reading")
        summary = self._summary(rows, weights)
        if summary["observed_fraction"] == 0.0:
            warnings.append(
                "no valid TROPOMI retrieval covers this polygon on this date: "
                "every cell shown is model output (reconstructed), not an "
                "observation")
        return {
            "city": self.city,
            "date": date,
            "grid_cell_deg": GRID_CELL_DEG,
            "polygon": {"area_m2": info["area_m2"],
                        "n_vertices": info["n_vertices"]},
            "fetch_window": window,
            "model": {"run": self.run_name,
                      "feature_schema_version": self.artifact.feature_schema_version,
                      "alpha": self.artifact.alpha,
                      "groups": self.artifact.groups},
            "data": {"processor_versions": list(versions),
                     "no2_source": "COPERNICUS/S5P/OFFL/L3_NO2"},
            "summary": summary,
            "warnings": warnings,
            "cache_hit": cache_hit,
            # G3 evidence (honest report), so the UI never presents a
            # climatology-anchored estimate as validated skill
            "g3": self.g3,
            "climatology_anchor": {"available":
                                   self.artifact.climatology is not None,
                                   "definition":
                                   "median observed value per (cell, month) "
                                   "over the training window"},
            "cells": rows,
        }

    @staticmethod
    def _summary(rows: list[dict], weights: np.ndarray) -> dict:
        def _arr(key: str) -> np.ndarray:
            return np.array([np.nan if r.get(key) is None else float(r[key])
                             for r in rows], dtype=float)

        total_w = float(weights.sum())
        obs = np.array([r["state"] == OBSERVED for r in rows])
        val, pred, margin = _arr("value"), _arr("pred"), _arr("margin")
        clim = _arr("climatology")
        ok = np.isfinite(val)

        def _wmean(x: np.ndarray, m: np.ndarray) -> float | None:
            if m.sum() == 0:
                return None
            return float(np.sum(weights[m] * x[m]) / np.sum(weights[m]))

        return {
            "n_cells": int(len(rows)),
            "total_weight": total_w,
            "observed_fraction": (float(weights[obs].sum() / total_w)
                                  if total_w > 0 else 0.0),
            "reconstructed_fraction": (float(weights[~obs].sum() / total_w)
                                       if total_w > 0 else 0.0),
            "weighted_mean": _wmean(val, ok),
            "weighted_mean_observed": _wmean(val, obs & ok),
            "weighted_mean_predicted": _wmean(pred, np.isfinite(pred)),
            "weighted_mean_climatology": _wmean(clim, np.isfinite(clim)),
            "weighted_mean_margin": _wmean(margin, np.isfinite(margin)),
        }

    @staticmethod
    def _rows(df: pd.DataFrame, overlaps: dict[str, float]) -> list[dict]:
        out = []
        for r in df.sort_values("cell_id").itertuples(index=False):
            row, col = parse_cell_id(r.cell_id)
            c = cell_from_row_col(row, col)
            margin = float(r.margin)
            pred = float(r.pred)
            clim = float(r.climatology) if pd.notna(r.climatology) else float("nan")
            val = r.value if r.value is not None else float("nan")
            out.append({
                "cell_id": r.cell_id,
                "lon": c.center_lon,
                "lat": c.center_lat,
                "weight": float(overlaps[r.cell_id]),
                "state": r.state,
                "reason_code": None if pd.isna(r.reason_code) else r.reason_code,
                "source": r.source,
                "valid_fraction": (round(float(r.n_valid_fraction), 4)
                                   if pd.notna(r.n_valid_fraction) else None),
                "value": None if not np.isfinite(val) else round(val, 4),
                "value_observed": (round(float(r.value_umol_m2), 4)
                                   if r.state == OBSERVED
                                   and pd.notna(r.value_umol_m2) else None),
                "pred": round(pred, 4) if np.isfinite(pred) else None,
                "climatology": round(clim, 4) if np.isfinite(clim) else None,
                "margin": round(margin, 4) if np.isfinite(margin) else None,
                "interval_lo": (round(pred - margin, 4)
                                if np.isfinite(pred) and np.isfinite(margin) else None),
                "interval_hi": (round(pred + margin, 4)
                                if np.isfinite(pred) and np.isfinite(margin) else None),
            })
        return out
