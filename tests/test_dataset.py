"""Tests for the training-dataset assembly layer."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from no2map.dataset import (
    TrainingConfig,
    done_months,
    load_training_panel,
    missing_window_months,
    month_files,
)

TARGET = "value_umol_m2"


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    """Fake manifest + two done months + one outside-window month, written
    with the real manifest schema (cities/months keys with status)."""
    cfg = {
        "training": {
            "window_start": "2023-04-01",
            "window_end": "2025-09-30",
            "exclude_months": ["2024-09", "2024-11"],
            "version_shift_test": {"start": "2025-11-01"},
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    entries = {}
    exports = tmp_path / "exports" / "delhi_ncr"
    exports.mkdir(parents=True)
    for mo, n in (("2023-04", 40), ("2024-08", 20), ("2025-11", 10)):
        entries[f"delhi_ncr/{mo}"] = {"city": "delhi_ncr", "month": mo,
                                      "status": "done", "attempts": 1}
        df = pd.DataFrame([{"cell_id": f"c{i}", "date": f"{mo}-01",
                            TARGET: float(i)} for i in range(n)])
        df.to_parquet(exports / f"delhi_ncr_{mo}.parquet")
    entries["delhi_ncr/2024-10"] = {"city": "delhi_ncr", "month": "2024-10",
                                    "status": "skipped", "attempts": 2}
    manifest_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return dict(cfg=cfg_path, manifest=manifest_path, exports=tmp_path / "exports")


class TestWindow:
    def test_month_in_window_inclusive_and_exclusions(self):
        tc = TrainingConfig("2023-04-01", "2025-09-30", ["2024-09", "2024-11"])
        assert tc.month_in_window("2023-04")
        assert tc.month_in_window("2025-09")
        assert not tc.month_in_window("2023-03")
        assert not tc.month_in_window("2025-10")
        assert not tc.month_in_window("2024-09")   # transition month dropped
        assert not tc.month_in_window("2024-11")

    def test_version_shift_recorded(self):
        tc = TrainingConfig("2023-04-01", "2025-09-30", [], "2025-11-01")
        assert tc.version_shift_start == "2025-11-01"


class TestLoader:
    def test_done_months_reads_manifest(self, workspace):
        mos = done_months(workspace["manifest"], "delhi_ncr")
        assert mos == sorted({"2023-04", "2024-08", "2025-11"})
        assert "2024-10" not in mos  # skipped months are not done

    def test_missing_window_months_flags_partial_window(self, workspace):
        missing = missing_window_months(
            workspace["cfg"], workspace["manifest"], "delhi_ncr")
        assert "2023-04" not in missing        # done
        assert "2024-08" not in missing        # done
        assert "2023-05" in missing            # never exported
        assert "2024-10" in missing            # skipped, so still missing
        assert "2024-09" not in missing        # excluded transition month
        assert "2024-11" not in missing        # excluded transition month
        assert "2025-11" not in missing        # outside the training window

    def test_month_files_fails_loudly_on_missing_parquet(self, workspace):
        with pytest.raises(FileNotFoundError):
            month_files(workspace["exports"], "delhi_ncr", ["2023-04", "2024-07"])

    def test_panel_restricted_to_window(self, workspace):
        panel = load_training_panel(
            workspace["cfg"], workspace["manifest"], workspace["exports"], "delhi_ncr")
        # 2025-11 parquet exists and is done, but the config window ends 2025-09
        mos = panel.attrs["months"]
        assert "2024-08" in mos and "2025-11" not in mos
        assert sorted(panel["date"].str[:7].unique()) == ["2023-04", "2024-08"]

    def test_manifest_parquet_month_mismatch_raises(self, workspace):
        # corrupt one parquet's dates so the check trips
        p = workspace["exports"] / "delhi_ncr" / "delhi_ncr_2023-04.parquet"
        pd.DataFrame([{"cell_id": "x", "date": "2022-01-01", TARGET: 1.0}]).to_parquet(p)
        with pytest.raises(ValueError, match="mismatch"):
            load_training_panel(workspace["cfg"], workspace["manifest"],
                                workspace["exports"], "delhi_ncr")

    def _set_schema(self, workspace, months: list[str], value: str) -> None:
        data = json.loads(workspace["manifest"].read_text(encoding="utf-8"))
        for mo in months:
            data["entries"][f"delhi_ncr/{mo}"]["feature_schema_version"] = value
        workspace["manifest"].write_text(json.dumps(data), encoding="utf-8")

    def test_mixed_feature_schema_refused(self, workspace):
        from no2map.features import FEATURE_SCHEMA_VERSION
        self._set_schema(workspace, ["2023-04"], FEATURE_SCHEMA_VERSION)
        self._set_schema(workspace, ["2024-08"], "fm-old")
        with pytest.raises(ValueError, match="mixed feature schemas"):
            load_training_panel(workspace["cfg"], workspace["manifest"],
                                workspace["exports"], "delhi_ncr")

    def test_stale_feature_schema_refused(self, workspace):
        # a whole city exported under an older schema must not be trained on
        # and then served as if it carried the installed schema
        self._set_schema(workspace, ["2023-04", "2024-08"], "fm-old")
        with pytest.raises(ValueError, match="re-export"):
            load_training_panel(workspace["cfg"], workspace["manifest"],
                                workspace["exports"], "delhi_ncr")
