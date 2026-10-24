"""Tests for the per-month export checks and the G1 gate decision.

The checks were calibrated against the live 2023-04 export (median 51.6,
land+mixed water, 672% observed... no: 0.672). Each hard check is broken
deliberately here (missing day, out-of-grid cell, bad state, pure-water
row, dead median, NaN target on observed rows, thinned weather, precip
lag violation, manifest drift) and the gate must fail each one.
"""
from __future__ import annotations

from calendar import monthrange

import numpy as np
import pandas as pd
import pytest

from no2map.features import FEATURE_SCHEMA_VERSION, TARGET
from no2map.monthcheck import (
    VALID_STATES,
    WEATHER_COLS,
    check_month,
    gate_g1,
    month_verdict,
)

CELLS = [f"r{r:04d}c{c:03d}" for r in range(2) for c in range(3)]
MONTH = "2024-06"  # monsoon: exercises the monsoon-precip warn check
NDAYS = monthrange(2024, 6)[1]


def healthy_df() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for d in range(1, NDAYS + 1):
        for i, cid in enumerate(CELLS):
            rows.append({
                "cell_id": cid, "date": f"{MONTH}-{d:02d}",
                "city": "test", "reason_code": "ok",
                "state": "observed" if (i + d) % 4 else "no_overpass",
                "water_state": "land" if i else "mixed",
                TARGET: 40.0 + 5.0 * i + rng.normal(0, 2),
                "precip_l24_m": abs(rng.normal(0.002, 0.001)),
                "precip_l48_m": abs(rng.normal(0.003, 0.001)),
                "t2m_k_inst": 305.0 + rng.normal(0, 2),
            })
    df = pd.DataFrame(rows)
    # guarantee l48 >= l24 rowwise
    df["precip_l48_m"] = df[["precip_l24_m", "precip_l48_m"]].max(axis=1)
    for col in WEATHER_COLS:
        if col not in df:
            df[col] = 1.0
    return df


def manifest_entry(df: pd.DataFrame) -> dict:
    return {
        "city": "test", "month": MONTH, "status": "done",
        "n_rows": len(df),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "processor_versions": ["2.8.0"],
    }


class TestHealthy:
    def test_all_checks_pass(self):
        df = healthy_df()
        checks = check_month(df, CELLS, MONTH, manifest_entry(df))
        failed = [(c.name, c.detail) for c in checks if not c.passed and c.severity == "hard"]
        assert failed == [], failed
        passed, warns = month_verdict(checks)
        assert passed

    def test_warns_reported_not_fatal(self):
        df = healthy_df()
        df.loc[df.index[:5], "t2m_k_inst"] = 350.0  # physically silly
        checks = check_month(df, CELLS, MONTH, manifest_entry(df))
        passed, warns = month_verdict(checks)
        assert passed and "t2m_physical_band" in warns


class TestHardFailures:
    def run_check(self, df, entry=None):
        return check_month(df, CELLS, MONTH, entry or manifest_entry(df))

    def test_missing_day(self):
        df = healthy_df()
        broken = df[df["date"] != f"{MONTH}-{NDAYS:02d}"]
        v, _ = month_verdict(self.run_check(broken))
        assert not v and not all(c.passed for c in self.run_check(broken) if c.name == "dates_complete")

    def test_out_of_grid_cell(self):
        df = pd.concat([healthy_df(),
                        healthy_df().query(f"date == '{MONTH}-01'").assign(cell_id="ZZZ_boom")])
        checks = self.run_check(df)
        assert any(not c.passed and c.name in ("cells_match", "row_grid", "dates_complete")
                   for c in checks)

    def test_bad_state_value(self):
        df = healthy_df()
        df.loc[df.index[0], "state"] = "mystery"
        checks = self.run_check(df)
        assert any(not c.passed and c.name == "states_valid" for c in checks)

    def test_pure_water_row(self):
        df = healthy_df()
        df.loc[df.index[0], "water_state"] = "water"
        checks = self.run_check(df)
        assert any(not c.passed and c.name == "water_states" for c in checks)

    def test_implausible_median(self):
        df = healthy_df()
        df[TARGET] = 3.0  # too low for any real month (band is [5, 200])
        checks = self.run_check(df)
        assert any(not c.passed and c.name == "observed_median_band" for c in checks)

    def test_nan_target_on_observed(self):
        df = healthy_df()
        df.loc[df.index[0], TARGET] = np.nan
        checks = self.run_check(df)
        assert any(not c.passed and c.name == "observed_values_present" for c in checks)

    def test_thinned_weather(self):
        df = healthy_df()
        df.loc[df.index[: df.shape[0] // 3], "blh_m_inst"] = np.nan
        checks = self.run_check(df)
        assert any(not c.passed and c.name == "weather_present" for c in checks)

    def test_precip_lag_violation(self):
        df = healthy_df()
        df.loc[df.index[0], "precip_l48_m"] = 0.0
        df.loc[df.index[0], "precip_l24_m"] = 0.01
        checks = self.run_check(df)
        assert any(not c.passed and c.name == "precip_invariants" for c in checks)

    def test_manifest_drift(self):
        df = healthy_df()
        entry = manifest_entry(df)
        entry["n_rows"] = len(df) + 1
        entry["feature_schema_version"] = "feat-OLD"
        checks = self.run_check(df, entry)
        assert any(not c.passed and c.name == "manifest_parity" for c in checks)


class TestGateG1:
    def _gate(self, frames, entries):
        return gate_g1(entries, frames, CELLS,
                       "2024-06-01", "2024-07-31",
                       exclude_months=["2024-07"])

    def test_full_window_passes(self):
        june, july = healthy_df(), healthy_df()
        entries = {"2024-06": manifest_entry(june), "2024-07": None}
        rep = self._gate({MONTH: june}, entries)
        # July is excluded (transition) and untracked -> noted, not fatal
        assert rep["months"]["2024-07"]["role"] == "excluded_transition"
        assert rep["months"]["2024-06"]["ok"]

    def test_missing_month_fails(self):
        june = healthy_df()
        entries = {"2024-06": manifest_entry(june)}
        rep = self._gate({MONTH: june}, entries)
        # add July as a required window month by shrinking the exclusion
        from no2map.monthcheck import gate_g1 as g
        rep2 = g(entries, {MONTH: june}, CELLS, "2024-06-01", "2024-07-31", [])
        assert rep2["verdict"] == "FAIL"
        assert rep2["months"]["2024-07"]["status"] is None

    def test_excluded_month_done_flags(self):
        june = healthy_df()
        entries = {"2024-06": manifest_entry(june),
                   "2024-07": manifest_entry(healthy_df())}
        rep = self._gate({MONTH: june}, entries)
        assert rep["verdict"] == "FAIL"
        assert not rep["months"]["2024-07"]["ok"]

    def test_all_window_months_done_and_valid_passes(self):
        # window = 2024-06 only (end mid-month): months() inclusive on months
        june = healthy_df()
        entries = {"2024-06": manifest_entry(june)}
        rep = gate_g1(entries, {MONTH: june}, CELLS, "2024-06-01", "2024-06-30", [])
        assert rep["verdict"] == "PASS"
