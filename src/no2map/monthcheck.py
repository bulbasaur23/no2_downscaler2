"""Per-month export validation + Gate G1 (plan section on Phase-1 gate).

The export sweep is long (hours per city); human eyeballing of parquets is
how bad months slip into training undetected. This module turns the healthy-
month criteria into executable hard checks (a violation fails the month) and
warn-level checks (reported, judgement deferred):

HARD: cell set == the config-bbox grid (row/col tiling, D8); row count ==
cells x days; every calendar day present; state Domain (observed /
no_overpass / insufficient); water cells only land/mixed (D6); observed
values fully present with a plausible city median; weather columns
populated; precip non-negative with the lag invariant l48 >= l24 (the
de-accumulated hourly band fix, pinned 2026-10); manifest row count and
feature-schema parity with features.FEATURE_SCHEMA_VERSION.

WARN: retrieval-noise tails (negative observed values), monsoon precip
presence (would have caught the midnight-reset band bug only weakly, hence
a warning not a gate), t2m physical band.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass

import pandas as pd

from no2map.features import TARGET, FEATURE_SCHEMA_VERSION

STATE_OBSERVED = "observed"
VALID_STATES = {STATE_OBSERVED, "no_overpass", "insufficient"}
VALID_WATER_STATES = {"land", "mixed"}

WEATHER_COLS = [
    "wind_u10_inst", "wind_v10_inst", "wind_u100_inst", "wind_v100_inst",
    "blh_m_inst",
    "wind_u10_l24", "wind_v10_l24", "wind_u100_l24", "wind_v100_l24",
    "wind_u10_l48", "wind_v10_l48", "wind_u100_l48", "wind_v100_l48",
    "t2m_k_inst", "d2m_k_inst", "sp_pa_inst",
    "precip_l24_m", "precip_l48_m",
]


@dataclass
class Check:
    name: str
    passed: bool
    severity: str  # "hard" | "warn"
    detail: str


def _month_days(month: str) -> list[str]:
    y, m = int(month[:4]), int(month[5:7])
    nd = calendar.monthrange(y, m)[1]
    return [f"{month}-{d:02d}" for d in range(1, nd + 1)]


def check_month(
    df: pd.DataFrame,
    expected_cells: list[str],
    month: str,
    manifest_entry: dict | None = None,
) -> list[Check]:
    """All checks for one monthly export frame; hard failures fail the gate."""
    checks: list[Check] = []
    obs = df.loc[df["state"] == STATE_OBSERVED, TARGET].dropna()

    def add(name: str, ok: bool, detail: str, severity: str = "hard") -> None:
        checks.append(Check(name, ok, severity, detail))

    # 1. cell tiling (D8): exactly the bbox grid cells, no extras, no misses
    got_cells = set(df["cell_id"])
    missing = sorted(set(expected_cells) - got_cells)[:5]
    extra = sorted(got_cells - set(expected_cells))[:5]
    add("cells_match", got_cells == set(expected_cells),
        f"n_cells={df['cell_id'].nunique()} expected={len(expected_cells)}"
        + (f" missing={missing}" if missing else "")
        + (f" extra={extra}" if extra else ""))

    # 2. row grid: every cell x every day, exactly once
    days = _month_days(month)
    add("row_grid", len(df) == len(expected_cells) * len(days),
        f"rows={len(df)} expected={len(expected_cells) * len(days)}")

    # 3. calendar completeness
    vc = df["date"].value_counts()
    bad_days = sorted(d for d, n in vc.items()
                      if n != len(expected_cells) or d not in set(days))
    add("dates_complete", not bad_days and len(vc) == len(days),
        f"days_with_rows={len(vc)}/{len(days)}"
        + (f" bad={bad_days[:5]}" if bad_days else ""))

    # 4. state domain
    states = set(df["state"])
    add("states_valid", states <= VALID_STATES, f"states={sorted(states)}")

    # 5. water domain (D6): no pure-water grid cells
    wstates = set(df["water_state"])
    add("water_states", wstates <= VALID_WATER_STATES,
        f"water_state={sorted(wstates)} counts={df['water_state'].value_counts().to_dict()}")

    # 6. observed plausibility (city median band spans monsoon trough..winter peak)
    median_ok = 5.0 <= obs.median() <= 200.0 if len(obs) else False
    add("observed_median_band", median_ok,
        "median=" + (f"{obs.median():.1f}" if len(obs) else "NA") + " umol/m2")

    # 7. observed completeness: the TARGET present on every observed row
    n_obs_rows = int((df["state"] == STATE_OBSERVED).sum())
    add("observed_values_present", len(obs) == n_obs_rows,
        f"observed_rows={n_obs_rows} with_value={len(obs)}")

    # 8. weather columns populated (ERA5 stack is exported for every cell)
    frac = df[WEATHER_COLS].notna().mean()
    thin = [c for c in WEATHER_COLS if frac[c] < 0.9]
    add("weather_present", not thin,
        "min_nonnan_frac=" + (f"{frac.min():.3f}" if len(frac) else "NA")
        + (f" thin={thin}" if thin else ""))

    # 9. precip invariants: non-negative and l48 window >= l24 window.
    # Tolerance 0.1 mm: ECMWF accumulation packing quantizes hourly precip
    # at 1e-6 m, so shared-window sums can differ by a few micrometres in
    # EITHER direction (measured: 2025-02-20, 4/45080 rows, ~1-5 um). The
    # defect class this check exists for (raw total_precipitation summed
    # across lags, resetting at midnight) shows up at millimetre scale.
    PRECIP_LAG_TOL_M = 1e-4
    p24, p48 = df["precip_l24_m"], df["precip_l48_m"]
    neg = int(((p24 < -1e-12) | (p48 < -1e-12)).sum())
    lag = int((p48 < p24 - PRECIP_LAG_TOL_M).sum())
    worst = float((p24 - p48).max()) if len(df) else 0.0
    add("precip_invariants", neg == 0 and lag == 0,
        f"negative={neg} lag_violations={lag} worst_inversion_m={worst:.2e}")

    # 10. manifest parity (records what landed vs what the frame says)
    if manifest_entry is not None:
        rows_ok = manifest_entry.get("n_rows") == len(df)
        schema_ok = manifest_entry.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        versions_ok = bool(manifest_entry.get("processor_versions"))
        add("manifest_parity",
            rows_ok and schema_ok and versions_ok
            and manifest_entry.get("status") == "done",
            f"n_rows={manifest_entry.get('n_rows')} schema="
            f"{manifest_entry.get('feature_schema_version')} vs {FEATURE_SCHEMA_VERSION}"
            f" versions={manifest_entry.get('processor_versions')}")

    # ---- warn level ----
    neg_obs_frac = float((obs < 0).mean()) if len(obs) else 0.0
    add("observed_negative_tail",
        (obs.min() > -60.0 if len(obs) else False) and neg_obs_frac < 0.02,
        f"min={obs.min():.1f} neg_frac={neg_obs_frac:.4f}", severity="warn")

    mo = int(month[5:7])
    if 6 <= mo <= 9:
        frac_wet = float((df.loc[df["state"] == STATE_OBSERVED, "precip_l24_m"] > 1e-4).mean())
        add("monsoon_precip_present", frac_wet > 0.03,
            f"wet_obs_frac={frac_wet:.3f}", severity="warn")

    if len(df):
        t2m = df["t2m_k_inst"].dropna()
        add("t2m_physical_band",
            bool(len(t2m)) and 260.0 <= t2m.min() and t2m.max() <= 330.0,
            f"range={t2m.min():.1f}..{t2m.max():.1f}" if len(t2m) else "no t2m",
            severity="warn")

    obs_frac = float((df["state"] == STATE_OBSERVED).mean()) if len(df) else 0.0
    add("observed_fraction_extreme", 0.02 <= obs_frac <= 0.98,
        f"observed_frac={obs_frac:.3f}", severity="warn")

    return checks


def month_verdict(checks: list[Check]) -> tuple[bool, list[str]]:
    hard_failed = [c.name for c in checks if not c.passed and c.severity == "hard"]
    warns = [c.name for c in checks if not c.passed and c.severity == "warn"]
    return (not hard_failed), warns


def gate_g1(
    entries: dict[str, dict | None],
    frames: dict[str, pd.DataFrame],
    expected_cells: list[str],
    window_start: str,
    window_end: str,
    exclude_months: list[str],
) -> dict:
    """G1 decision over the whole window.

    entries: month -> manifest entry dict (or None if untracked);
    frames: month -> daily panel for months that produced a parquet.
    Every in-window month must be done and pass all hard checks; excluded
    (transition) months must NOT be done and are noted, not failed.
    """

    def _months(a: str, b: str) -> list[str]:
        out, ya, yb = [], int(a[:4]), int(b[:4])
        for y in range(ya, yb + 1):
            m0 = int(a[5:7]) if y == ya else 1
            m1 = int(b[5:7]) if y == yb else 12
            out.extend(f"{y}-{m:02d}" for m in range(m0, m1 + 1))
        return out

    per_month: dict[str, dict] = {}
    ok = True
    for month in _months(window_start[:7], window_end[:7]):
        entry = entries.get(month)
        if month in exclude_months:
            status = entry.get("status") if entry else None
            per_month[month] = {"role": "excluded_transition", "ok": True,
                                "status": status}
            if status == "done":
                ok = False
                per_month[month]["ok"] = False
                per_month[month]["detail"] = ("transition month exported; "
                                              "excluded from panel only")
            continue
        status = entry.get("status") if entry else None
        if status != "done":
            ok = False
            per_month[month] = {"role": "window", "ok": False,
                                "status": status, "detail": "not done"}
            continue
        if month not in frames:
            ok = False
            per_month[month] = {"role": "window", "ok": False,
                                "status": status, "detail": "no parquet"}
            continue
        checks = check_month(frames[month], expected_cells, month,
                             manifest_entry=entry)
        passed, warns = month_verdict(checks)
        per_month[month] = {
            "role": "window", "ok": passed, "status": status,
            "failed": [c.name for c in checks
                       if not c.passed and c.severity == "hard"],
            "warned": warns,
            "checks": {c.name: {"ok": c.passed, "detail": c.detail,
                                "severity": c.severity}
                       for c in checks},
        }
        ok = ok and passed
    return {"verdict": "PASS" if ok else "FAIL", "months": per_month}
