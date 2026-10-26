"""Offline guards for the G4 parity tool (scripts/pipeline/parity_check.py).

The real parity run needs Earth Engine, but the part that decides PASS/FAIL is
pure comparison and must be right offline: a false PASS here would undermine
the whole G4 claim. Pinned:

- NaN patterns are compared as data (a cell where one path has a value and the
  other has NaN is a mismatch, not "no overlap"),
- exact equality is 0, real differences surface at their true magnitude,
- the CLI refuses loudly (exit 2) when the month export is missing instead of
  reporting parity against nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pipeline"))

import parity_check  # noqa: E402


class TestMaxAbsDiff:
    def test_identical_series_is_zero(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert parity_check._max_abs_diff(s, s.copy()) == 0.0

    def test_difference_reports_magnitude(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([1.0, 2.0, 3.5])
        assert parity_check._max_abs_diff(a, b) == pytest.approx(0.5)

    def test_nan_pattern_mismatch_is_infinite_not_ignored(self):
        a = pd.Series([1.0, np.nan, 3.0])
        b = pd.Series([1.0, 2.0, 3.0])
        assert parity_check._max_abs_diff(a, b) == float("inf")

    def test_shared_nans_are_equal(self):
        a = pd.Series([np.nan, 2.0])
        assert parity_check._max_abs_diff(a, a.copy()) == 0.0

    def test_all_nan_is_zero(self):
        a = pd.Series([np.nan, np.nan])
        assert parity_check._max_abs_diff(a, a.copy()) == 0.0


class TestMissingExportFailsLoudly:
    def test_exit_2_when_month_parquet_absent(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(parity_check, "EXPORTS", tmp_path)
        monkeypatch.setattr(sys, "argv",
                            ["parity_check", "--city", "delhi_ncr",
                             "--date", "2025-02-10"])
        assert parity_check.main() == 2
        assert "missing export" in capsys.readouterr().out
