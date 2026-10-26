"""Tests for the per-cell result cache (the plan's five-part cache key).

The whole value of this cache is invalidation: if any of cell, date, feature
version, processor version or model version changes, the entry must miss.
These tests pin that per component, plus round-trip fidelity.
"""
from __future__ import annotations

import pytest

from no2map.cache import CellCache, cell_key

BASE = dict(cell_id="r02370c005132", date="2024-07-01",
            feature_version="fm-v1", processor_versions="2.8.0",
            model_version="run_a")


class TestKey:
    def test_all_five_components_change_the_key(self):
        k0 = cell_key(**BASE)
        variants = [
            {**BASE, "cell_id": "r02370c005133"},
            {**BASE, "date": "2024-07-02"},
            {**BASE, "feature_version": "fm-v2"},
            {**BASE, "processor_versions": "02.09.01"},
            {**BASE, "model_version": "run_b"},
        ]
        keys = {k0, *(cell_key(**v) for v in variants)}
        assert len(keys) == 6

    def test_processor_version_form_does_not_matter(self):
        a = cell_key(**{**BASE, "processor_versions": ["2.8.0", "2.7.1"]})
        b = cell_key(**{**BASE, "processor_versions": ["2.7.1", "2.8.0"]})
        c = cell_key(**{**BASE, "processor_versions": "2.7.1,2.8.0"})
        assert a == b == c

    def test_field_boundaries_are_unambiguous(self):
        # concatenation without a separator would collide on these two
        a = cell_key("ab", "c", "d", "e", "f")
        b = cell_key("a", "bc", "d", "e", "f")
        assert a != b


class TestStore:
    @pytest.fixture
    def cache(self, tmp_path) -> CellCache:
        return CellCache(tmp_path / "cache.sqlite")

    def _row(self, key: str, payload: dict):
        return (key, "r02370c005132", "2024-07-01", "fm-v1", "2.8.0", "run_a",
                payload)

    def test_roundtrip(self, cache):
        k = cell_key(**BASE)
        cache.put_many([self._row(k, {"value": 42.5, "state": "observed"})])
        got = cache.get_many([k])
        assert got[k] == {"value": 42.5, "state": "observed"}

    def test_partial_hits(self, cache):
        k1, k2 = cell_key(**BASE), cell_key(**{**BASE, "cell_id": "x"})
        cache.put_many([self._row(k1, {"v": 1})])
        got = cache.get_many([k1, k2])
        assert set(got) == {k1}

    def test_overwrite_same_key(self, cache):
        k = cell_key(**BASE)
        cache.put_many([self._row(k, {"v": 1})])
        cache.put_many([self._row(k, {"v": 2})])
        assert cache.get_many([k])[k] == {"v": 2}

    def test_persists_across_instances(self, tmp_path):
        k = cell_key(**BASE)
        c1 = CellCache(tmp_path / "c.sqlite")
        c1.put_many([self._row(k, {"v": 7})])
        c1.close()
        c2 = CellCache(tmp_path / "c.sqlite")
        assert c2.get_many([k])[k] == {"v": 7}
        assert c2.count() == 1

    def test_empty_lookups_are_noops(self, cache):
        assert cache.get_many([]) == {}
        assert cache.put_many([]) == 0
        assert cache.count() == 0
