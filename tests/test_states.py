"""Error taxonomy tests (D6): states are distinct and never conflated."""
import pytest

from no2map.states import CellState, CellStatus, InsufficientReason, SystemErrorSource


def test_states_are_distinct_members():
    assert CellState.INSUFFICIENT != CellState.SYSTEM_ERROR
    assert CellState.NO_OVERPASS != CellState.INSUFFICIENT


def test_status_ok_states():
    assert CellStatus(CellState.OBSERVED).is_ok()
    assert CellStatus(CellState.RECONSTRUCTED).is_ok()
    assert not CellStatus(CellState.INSUFFICIENT, reason=InsufficientReason.CLOUDS).is_ok()


def test_system_error_carries_source():
    s = CellStatus(CellState.SYSTEM_ERROR, error_source=SystemErrorSource.GEE)
    assert s.error_source is SystemErrorSource.GEE
    assert s.reason is None  # never both


def test_insufficient_carries_reason_not_source():
    s = CellStatus(CellState.INSUFFICIENT, reason=InsufficientReason.HAZE)
    assert s.reason is InsufficientReason.HAZE
    assert s.error_source is None


def test_no_overpass_is_its_own_state():
    # A date with no overpass data gets no_overpass, NOT insufficient (D6).
    s = CellStatus(CellState.NO_OVERPASS)
    assert s.state is CellState.NO_OVERPASS and s.reason is None
