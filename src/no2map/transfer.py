"""Leave-one-city-out transfer and OOD evidence (Phase 5, gate G5).

Concept: the plan's Complete tier asks whether a model trained on some Indian
cities transfers to a city it has never seen, and whether the platform can say
*before* predicting that a cell is out of its training distribution. Neither
question may be tuned away: poor transfer is reported as poor.

Protocol implemented here:
  1. Fit everything (model, conformal margins, climatology/city-mean/Ridge
     references) on the training cities only, through the same blocked-CV
     machinery used for G3 - the held-out city contributes nothing.
  2. Predict the held-out city frozen, score every method with the shared
     two-level metric code, and report interval coverage on observed rows only
     (a row without a target is not an interval miss).
  3. Fit an OOD reference on the training cities' STATIC feature space
     (elevation mean/std, water fraction - no lat/lon, risk 11) and measure each
     held-out cell's distance to its k-th nearest training cell. Cells beyond
     the reference's own within-training quantile are flagged. The evidence
     block then compares held-out error for flagged vs unflagged cells: if the
     flag carries no error signal, that is reported as such rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from no2map.evaluation import full_scores, rmse
from no2map.features import TARGET
from no2map.model import PRED_COLS, XGBConfig
from no2map.splits import spatial_block_folds
from no2map.version_shift import fit_all, predict_frozen

# Static-only OOD space: deliberately no lat/lon (risk 11) and no target-
# derived quantity, so the flag is computable for any polygon before prediction.
OOD_FEATURES = ["elevation_mean_m", "elevation_std_m", "water_fraction"]


def _cell_matrix(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row per cell with its static features (statics are per cell)."""
    missing = [c for c in ["cell_id", *cols] if c not in panel.columns]
    if missing:
        raise KeyError(f"panel lacks OOD columns: {missing}")
    return panel.drop_duplicates("cell_id")[["cell_id", *cols]].reset_index(drop=True)


def _knn(z_ref: np.ndarray | None, z_new: np.ndarray, k: int) -> np.ndarray:
    """Euclidean distance to the k-th nearest reference cell (chunked).

    With `z_ref is None` the reference is the new points themselves and the
    self-distance is excluded, giving each training cell its leave-one-out
    distance. `k` is clamped to the available reference size.
    """
    ref = z_new if z_ref is None else z_ref
    if len(ref) == 0:
        return np.full(len(z_new), np.inf)
    kk = max(1, min(k + (1 if z_ref is None else 0), len(ref)))
    out = np.empty(len(z_new), dtype=float)
    for i in range(0, len(z_new), 512):
        block = z_new[i:i + 512]
        d = np.sqrt(((block[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1))
        if z_ref is None:
            # exclude self (distance 0); document order matches ref order
            for j in range(len(block)):
                d[j, i + j] = np.inf
        out[i:i + 512] = np.partition(d, kk - 1, axis=1)[:, kk - 1]
    return out


@dataclass
class OODReference:
    """k-NN distance gate in standardized static-feature space."""

    cols: list[str]
    mean: np.ndarray
    scale: np.ndarray
    k: int
    quantile: float
    threshold: float
    z: np.ndarray = field(repr=False)

    @classmethod
    def fit(cls, train_panel: pd.DataFrame, cols: list[str] | None = None,
            k: int = 5, quantile: float = 0.95) -> "OODReference":
        cols = list(cols or OOD_FEATURES)
        cells = _cell_matrix(train_panel, cols)
        x = cells[cols].to_numpy(dtype=float)
        ok = np.isfinite(x).all(axis=1)
        if ok.sum() < 2:
            raise ValueError("need >= 2 cells with complete static features "
                             "to fit the OOD reference")
        mean = x[ok].mean(axis=0)
        scale = x[ok].std(axis=0)
        scale[scale < 1e-12] = 1.0  # constant feature: contributes no distance
        z = (x[ok] - mean) / scale
        loo = _knn(None, z, k)
        return cls(cols=cols, mean=mean, scale=scale, k=k,
                   quantile=quantile,
                   threshold=float(np.quantile(loo, quantile)), z=z)

    def distances(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Per-cell OOD distance; a cell with incomplete statics is `inf`.

        Unknown static context is treated as out-of-distribution rather than
        assumed in-distribution - the flag must be conservative.
        """
        cells = _cell_matrix(panel, self.cols)
        x = cells[self.cols].to_numpy(dtype=float)
        ok = np.isfinite(x).all(axis=1)
        dist = np.full(len(cells), np.inf)
        if ok.any():
            z = (x[ok] - self.mean) / self.scale
            dist[ok] = _knn(self.z, z, self.k)
        return pd.DataFrame({"cell_id": cells["cell_id"].to_numpy(),
                             "ood_distance": dist})

    def to_dict(self) -> dict:
        return {"features": list(self.cols), "k": self.k,
                "quantile": self.quantile, "threshold": self.threshold,
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "n_reference_cells": int(len(self.z))}


def city_diagnostics(panel: pd.DataFrame) -> dict:
    """Cheap per-city data checks the plan asks for at Phase 5 start."""
    n = int(len(panel))
    states = (panel["state"].value_counts().to_dict()
              if "state" in panel.columns else {})
    observed = int(states.get("observed", 0))
    obs = panel.loc[panel["state"] == "observed"] if "state" in panel.columns else panel
    return {
        "n_rows": n,
        "n_cells": int(panel["cell_id"].nunique()),
        "date_start": str(panel["date"].min()),
        "date_end": str(panel["date"].max()),
        "state_counts": {str(k): int(v) for k, v in states.items()},
        "observed_fraction": (observed / n) if n else float("nan"),
        "target_median": (float(obs[TARGET].median())
                          if len(obs) and TARGET in obs.columns else float("nan")),
        "water_fraction_p95": (float(panel["water_fraction"].quantile(0.95))
                               if "water_fraction" in panel.columns else float("nan")),
    }


def ood_evidence(preds: pd.DataFrame, pred_col: str = "pred_model_a") -> dict:
    """Does the OOD flag actually mark higher held-out error?

    Scored on rows that have both an observation and a finite OOD distance.
    Returns `flagged_higher_error: None` when either group is empty - an empty
    group is not evidence of anything.
    """
    need = {TARGET, pred_col, "ood_distance", "ood"}
    if not need <= set(preds.columns):
        raise KeyError(f"OOD evidence needs columns {sorted(need)}")
    y = preds[TARGET].to_numpy(dtype=float)
    p = preds[pred_col].to_numpy(dtype=float)
    d = preds["ood_distance"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(d)
    dd = preds.loc[ok]
    flag = dd["ood"].to_numpy(dtype=bool)
    f = rmse(dd.loc[flag, TARGET], dd.loc[flag, pred_col]) if flag.any() else None
    u = (rmse(dd.loc[~flag, TARGET], dd.loc[~flag, pred_col])
         if (~flag).any() else None)
    return {
        "n_scored": int(ok.sum()),
        "n_flagged_cells": int(dd.loc[flag, "cell_id"].nunique()) if flag.any() else 0,
        "flagged_row_fraction": float(flag.mean()) if ok.any() else float("nan"),
        "rmse_flagged": f,
        "rmse_unflagged": u,
        "delta_flagged_minus_unflagged": (None if f is None or u is None else f - u),
        "flagged_higher_error": (None if f is None or u is None else bool(f > u)),
    }


def fold_feasibility_problems(
    panels: dict[str, pd.DataFrame],
    train_cities_for: dict[str, list[str]],
    block_m: float,
    buffer_m: float,
    verbose: bool = True,
) -> list[str]:
    """Per-fold block-feasibility check, made BEFORE any fitting starts.

    The blocked-CV protocol is part of the treatment, so this never looks at
    outcomes - only at whether each fold's training-pool extent can carry the
    requested block/gap at all (measured: a two-city LOSO leaves a fold whose
    pool is one small city, and Delhi's 89,278 m block tiles that city into a
    single degenerate block - the pilot run died at fold 1 after loading
    panels). Returns one sentence per infeasible fold, sorted by fold name;
    feasible folds are printed when `verbose` is set.
    """
    problems = []
    for test_city, train_cities in sorted(train_cities_for.items()):
        pool = pd.concat([panels[c] for c in train_cities], ignore_index=True)
        cells = sorted(pool["cell_id"].unique())
        try:
            folds = spatial_block_folds(cells, block_m=block_m,
                                        buffer_m=buffer_m)
        except ValueError as e:
            problems.append(
                f"fold test={test_city} (train: {', '.join(train_cities)}, "
                f"{len(cells)} cells): {e}")
            continue
        sizes = [len(t) for _, t, _ in folds]
        if verbose:
            print(f"[g5] fold test={test_city}: {len(folds)} blocks of "
                  f"{block_m:,.0f} m, train cells {min(sizes)}..{max(sizes)} "
                  f"per fold (pool {len(cells)})")
    return problems


def transfer_report(
    panels: dict[str, pd.DataFrame],
    groups: list[str],
    *,
    test_cities: list[str] | None = None,
    seed: int = 0,
    alpha: float = 0.1,
    block_m: float = 30_000.0,
    buffer_m: float = 0.0,
    xgb: XGBConfig | None = None,
    ood_quantile: float = 0.95,
    ood_k: int = 5,
) -> dict:
    """Leave-one-city-out (or train-on-all) transfer report.

    `test_cities` defaults to every city in `panels` (leave-one-out). To run
    the untouched case study, pass e.g. `--test-cities hyderabad` (the model
    still trains on all other cities); whichever city is tested never trains
    the model that scores it.
    """
    pool = sorted(panels)
    tests = sorted(test_cities or pool)
    unknown = sorted(set(tests) - set(pool))
    if unknown:
        raise KeyError(f"test cities not in panels: {unknown}")

    out: dict = {
        "protocol": {
            "kind": "leave_one_city_out",
            "seed": seed,
            "alpha": alpha,
            "block_m": block_m,
            "buffer_m": buffer_m,
            "groups": list(groups),
            "ood": {"features": list(OOD_FEATURES), "k": ood_k,
                    "quantile": ood_quantile},
        },
        "cities": {c: city_diagnostics(panels[c]) for c in pool},
        "folds": {},
    }

    for test_city in tests:
        train_cities = [c for c in pool if c != test_city]
        if not train_cities:
            raise ValueError(f"cannot train for {test_city!r}: no other city")
        train = pd.concat([panels[c] for c in train_cities], ignore_index=True)
        fitted = fit_all(train, groups, seed, alpha,
                         block_m=block_m, buffer_m=buffer_m, xgb=xgb)

        frame = panels[test_city]
        preds = predict_frozen(fitted, frame, groups)
        preds["conf_margin"] = fitted["conformal"].margin(preds)

        base_ok = (np.isfinite(preds[TARGET].to_numpy(dtype=float))
                   & np.isfinite(preds["conf_margin"].to_numpy(dtype=float)))
        scores: dict = {}
        for col in PRED_COLS:
            if col not in preds:
                continue
            s = {k: v for k, v in full_scores(preds, col).items()
                 if k != "conformal"}
            ok = base_ok & np.isfinite(preds[col].to_numpy(dtype=float))
            if ok.any():
                y = preds.loc[ok, TARGET].to_numpy(dtype=float)
                p = preds.loc[ok, col].to_numpy(dtype=float)
                m = preds.loc[ok, "conf_margin"].to_numpy(dtype=float)
                s["coverage"] = float((np.abs(y - p) <= m).mean())
                s["mean_width"] = float(m.mean())
            else:
                s["coverage"] = float("nan")
                s["mean_width"] = float("nan")
            s["n_coverage"] = int(ok.sum())
            scores[col] = s

        ref = OODReference.fit(train, k=ood_k, quantile=ood_quantile)
        dist = ref.distances(frame)
        preds = preds.merge(dist, on="cell_id", how="left")
        preds["ood"] = preds["ood_distance"] > ref.threshold

        out["folds"][test_city] = {
            "train_cities": train_cities,
            "n_train_rows": int(len(train)),
            "n_eval_rows": int(len(frame)),
            "scores": scores,
            "baseline_context": _baseline_context(
                fitted["final"]["pred_climatology"], preds),
            "ood": {"reference": ref.to_dict(),
                    **ood_evidence(preds, "pred_model_a")},
        }
    return out


def _baseline_context(climatology, preds: pd.DataFrame) -> dict:
    """What the reference methods actually had to work with in this fold.

    The held-out city's cells are unseen by the training fit, so
    `pred_climatology` falls back down its ladder (cell-month -> cell ->
    pooled month -> pooled city median) instead of using per-cell history.
    Reporting the fallback rate keeps the comparison honest: beating
    climatology in a transfer fold is not the same claim as beating the
    per-cell anchor inside one city (G3).
    """
    rate = None
    try:
        rate = float(climatology.fallback_rate(preds))
    except (RuntimeError, KeyError, ValueError):
        rate = None  # an unfitted/degenerate reference: report nothing
    return {
        "climatology_fallback_rate": rate,
        "climatology_has_cell_history": bool(rate is not None and rate == 0.0),
        "note": ("held-out cells have no (cell, month) history, so the "
                 "climatology reference falls back to the pooled month/city "
                 "median; treat it as a transfer baseline, not the per-cell "
                 "anchor scored in G3"),
    }
