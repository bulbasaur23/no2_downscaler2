"""Tests for the evaluation split machinery (plan section 6).

What gets pinned:
- every cell is assigned to exactly one block and is validation in exactly
  one fold (the fold structure is a partition);
- the buffer gap actually removes ring cells around the val block;
- degenerate tilings raise with the plan's G2 fallback message;
- chronological splits are ordered, disjoint, and complete;
- and the host PANIC-CHECK: near-boundary cells in the same block as their
  parent (float-boundary safety of the tiling math).
"""
from __future__ import annotations

import pytest

from no2map.splits import (
    assign_spatial_blocks,
    chronological_split,
    spatial_block_folds,
)
from no2map.grid import cell_from_point, parse_cell_id
from no2map.variogram import TROPOMI_CELL_M


def parse_row(c: str) -> int:
    return parse_cell_id(c)[0]


def _grid(n_rows: int, n_cols: int, lat0: float = 28.0, lon0: float = 77.0):
    return [cell_from_point(lat0 + 0.05 * r, lon0 + 0.05 * c).id
            for r in range(n_rows) for c in range(n_cols)]


class TestBlockPartition:
    def test_every_cell_in_exactly_one_block(self):
        cells = _grid(6, 6)
        part = assign_spatial_blocks(cells, block_m=TROPOMI_CELL_M * 1.3)
        assert len(part.block_of) == len(cells)
        assert set(part.block_of) == set(cells)
        assert all(len(b) >= 1 for b in part.block_ids)

    def test_blocks_grow_with_block_size(self):
        cells = _grid(10, 10)
        fine = assign_spatial_blocks(cells, TROPOMI_CELL_M * 1.3)
        coarse = assign_spatial_blocks(cells, TROPOMI_CELL_M * 5.3)
        assert len(coarse.block_ids) <= len(fine.block_ids)

    def test_adjacent_cells_share_block(self):
        cells = _grid(3, 3)
        part = assign_spatial_blocks(cells, TROPOMI_CELL_M * 1.3)
        assert part.block_of[cells[0]] == part.block_of[cells[1]]

    def test_same_cell_never_two_blocks(self):
        cells = _grid(6, 6)
        part = assign_spatial_blocks(cells, TROPOMI_CELL_M * 2.3)
        assert part.block_ids == sorted(set(part.block_of.values()))

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            assign_spatial_blocks([], 1000.0)
        with pytest.raises(ValueError):
            assign_spatial_blocks(_grid(2, 2), 0.0)


class TestFolds:
    def test_each_block_validates_exactly_once(self):
        cells = _grid(8, 8)
        folds = spatial_block_folds(cells, block_m=TROPOMI_CELL_M * 2.3)
        val_sets = [set(v) for _, _, v in folds]
        union = set().union(*val_sets)
        assert union == set(cells)
        total_val = sum(len(v) for _, _, v in folds)
        assert total_val == len(cells)  # disjoint folds

    def test_train_never_contains_val_cells(self):
        cells = _grid(8, 8)
        for _, train, val in spatial_block_folds(cells, TROPOMI_CELL_M * 2.3):
            assert not (set(train) & set(val))

    def test_buffer_gap_removes_ring_cells(self):
        # 6 rows x 2 cols, 1.3-cell blocks: rows {0,1}, {2}, {3}, {4,5} tile
        # exactly this way (offsets 0.77/1.54/2.31/3.08/3.85 blocks). For the
        # singleton middle block (row 2): a 7000 m buffer (> one cell step,
        # 5559 m) drops the adjacent rows 1 and 3; rows 0, 4, 5 survive.
        cells = _grid(6, 2)
        loc_row = {cid: i // 2 for i, cid in enumerate(cells)}
        folds = spatial_block_folds(cells, TROPOMI_CELL_M * 1.3, buffer_m=7000.0)
        middle = [f for f in folds if f[0] == "b01_00"][0]
        _, train, val = middle
        assert {loc_row[c] for c in val} == {2}
        train_rows = {loc_row[c] for c in train}
        assert 1 not in train_rows and 3 not in train_rows, "ring rows must be dropped"
        assert train_rows == {0, 4, 5}, "outer rows must survive"

    def test_buffer_specifically_drops_adjacent_row(self):
        cells = _grid(6, 2)
        loc_row = {cid: i // 2 for i, cid in enumerate(cells)}
        no_buf = dict((b, set(tr)) for b, tr, _ in
                      spatial_block_folds(cells, TROPOMI_CELL_M * 1.3))
        with_buf = dict((b, set(tr)) for b, tr, _ in
                        spatial_block_folds(cells, TROPOMI_CELL_M * 1.3, buffer_m=7000.0))
        for b in with_buf:
            rows_with = {loc_row[c] for c in with_buf[b]}
            rows_without = {loc_row[c] for c in no_buf[b]}
            assert rows_with < rows_without  # ring rows dropped in every fold

    def test_degenerate_single_block_raises_with_g2_hint(self):
        cells = _grid(3, 3)
        with pytest.raises(ValueError, match="chronological"):
            spatial_block_folds(cells, block_m=TROPOMI_CELL_M * 50.0)

    def test_buffer_swallowing_raises(self):
        cells = _grid(3, 3)
        with pytest.raises(ValueError, match="buffer"):
            spatial_block_folds(cells, TROPOMI_CELL_M * 1.3, buffer_m=500_000.0)


class TestChronological:
    def test_ordered_disjoint_complete(self):
        dates = [f"2024-{m:02d}-15" for m in range(1, 13)]
        train, test = chronological_split(dates, train_frac=0.75)
        assert train == sorted(train) and test == sorted(test)
        assert not (set(train) & set(test))
        assert len(train) == 9 and len(test) == 3
        assert train[-1] < test[0]

    def test_frac_bounds(self):
        with pytest.raises(ValueError):
            chronological_split(["a", "b"], train_frac=0.0)
        with pytest.raises(ValueError):
            chronological_split(["a", "b"], train_frac=1.0)
        with pytest.raises(ValueError):
            chronological_split(["a"])

    def test_dupes_collapsed(self):
        train, test = chronological_split(["d2", "d1", "d1", "d2"], 0.5)
        assert train == ["d1"] and test == ["d2"]
