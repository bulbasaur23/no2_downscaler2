"""Manifest lifecycle tests: resume semantics, two-strike skip, atomicity."""
from pathlib import Path

import pytest

from no2map.manifest import (
    MAX_ATTEMPTS,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    ExportManifest,
)


@pytest.fixture()
def manifest(tmp_path: Path) -> ExportManifest:
    return ExportManifest.load(tmp_path / "manifest.json")


class TestMonths:
    def test_month_range_inclusive(self):
        assert ExportManifest.months_between("2022-11", "2023-02") == [
            "2022-11", "2022-12", "2023-01", "2023-02"
        ]

    def test_range_crosses_several_years(self):
        months = ExportManifest.months_between("2022-12", "2025-03")
        assert len(months) == 28 and months[-1] == "2025-03"


class TestLifecycle:
    def test_todo_then_done(self, manifest):
        months = ExportManifest.months_between("2026-07", "2026-09")
        manifest.ensure_pending("delhi_ncr", months)
        assert manifest.months_todo("delhi_ncr", months) == months
        manifest.record_success("delhi_ncr", "2026-07", "p.parquet", 100, "fv1", ["2023.001"])
        assert manifest.months_todo("delhi_ncr", months) == ["2026-08", "2026-09"]
        assert manifest.is_done("delhi_ncr", "2026-07")

    def test_two_failures_then_skip(self, manifest):
        months = ["2026-07"]
        manifest.ensure_pending("delhi_ncr", months)
        s1 = manifest.record_failure("delhi_ncr", "2026-07", "quota exhausted")
        assert s1 == STATUS_FAILED
        # retries remaining -> still in todo
        assert manifest.months_todo("delhi_ncr", months) == months
        s2 = manifest.record_failure("delhi_ncr", "2026-07", "quota exhausted again")
        assert s2 == STATUS_SKIPPED
        e = manifest.entries[manifest.key("delhi_ncr", "2026-07")]
        assert e.last_error == "quota exhausted again"  # recorded, never silent
        # skipped months are NOT retried by months_todo
        assert manifest.months_todo("delhi_ncr", months) == []

    def test_success_after_failure_resets_state(self, manifest):
        manifest.ensure_pending("delhi_ncr", ["2026-07"])
        manifest.record_failure("delhi_ncr", "2026-07", "err")
        manifest.record_success("delhi_ncr", "2026-07", "p.parquet", 10, "fv1", ["2023.001"])
        e = manifest.entries[manifest.key("delhi_ncr", "2026-07")]
        assert e.status == STATUS_DONE and e.last_error is None

    def test_success_after_skip_is_durably_done(self, manifest):
        # A recorded skip that later succeeds manually is DONE, not skipped.
        manifest.ensure_pending("delhi_ncr", ["2026-07"])
        manifest.record_failure("delhi_ncr", "2026-07", "a")
        manifest.record_failure("delhi_ncr", "2026-07", "b")
        manifest.record_success("delhi_ncr", "2026-07", "p.parquet", 10, "fv1", ["v"])
        assert manifest.entries[manifest.key("delhi_ncr", "2026-07")].status == STATUS_DONE

    def test_constant_is_plan_mandated(self):
        assert MAX_ATTEMPTS == 2  # plan Gate G1 fallback: fail twice -> skip


class TestPersistence:
    def test_save_and_load_round_trip(self, manifest):
        manifest.ensure_pending("delhi_ncr", ["2026-07"])
        manifest.record_success("delhi_ncr", "2026-07", "data/x.parquet", 1234, "fv1", ["2023.001"])
        manifest.save()
        m2 = ExportManifest.load(manifest.path)
        e = m2.entries[m2.key("delhi_ncr", "2026-07")]
        assert e.status == STATUS_DONE
        assert e.n_rows == 1234
        assert e.processor_versions == ["2023.001"]

    def test_load_missing_file_is_blank(self, tmp_path):
        m = ExportManifest.load(tmp_path / "nonexistent.json")
        assert m.entries == {}

    def test_schema_mismatches_detects_reexports(self, manifest):
        manifest.ensure_pending("delhi_ncr", ["2026-07", "2026-08"])
        manifest.record_success("delhi_ncr", "2026-07", "a.parquet", 10, "fv1", [])
        manifest.record_success("delhi_ncr", "2026-08", "b.parquet", 10, "fv2", [])
        mismatches = manifest.schema_mismatches("delhi_ncr", "fv2")
        assert mismatches == ["2026-07"]
