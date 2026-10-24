"""Tests for the shared feature module (Phase 2).

This module IS the train/serve parity contract, so its math gets the full
treatment: each derivation checked against hand-computed values, invariants
(RH in [0,100], unit-vector norm <= 1), idempotence, and the group machine
(unknown group, unbuilt columns, no-overlap).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from no2map.features import (
    FEATURE_GROUPS,
    FEATURE_SCHEMA_VERSION,
    TARGET,
    build_features,
    clean_training_frame,
    feature_block,
)


def _row_frame(**overrides) -> pd.DataFrame:
    """Minimal raw frame with every column build_features requires.

    Static columns are included because the real pipeline merges the statics
    parquet BEFORE feature building; build_features passes them through.
    """
    base = dict(
        elevation_mean_m=200.0, elevation_std_m=10.0,
        elevation_min_m=180.0, elevation_max_m=230.0,
        water_fraction=0.0,
        wind_u10_inst=3.0, wind_v10_inst=4.0,
        wind_u10_l24=1.0, wind_v10_l24=0.0,
        wind_u10_l48=0.0, wind_v10_l48=-2.0,
        t2m_k_inst=300.0, d2m_k_inst=290.0, sp_pa_inst=100_000.0,
        blh_m_inst=1000.0, precip_l24_m=0.0, precip_l48_m=0.0,
        month=7, state="observed", value_umol_m2=50.0,
    )
    base.update(overrides)
    return pd.DataFrame([base])


class TestDerivedScalars:
    def test_wind_speed_is_hypot(self):
        out = build_features(_row_frame())  # u=3, v=4 -> 5 exactly
        assert out["wind_speed_inst"].iloc[0] == pytest.approx(5.0)

    def test_lagged_wind_speeds(self):
        out = build_features(_row_frame())
        assert out["wind_speed_l24"].iloc[0] == pytest.approx(1.0)
        assert out["wind_speed_l48"].iloc[0] == pytest.approx(2.0)

    def test_rh_magnus_known_values(self):
        # 300 K dew 290 K: hand-check via Magnus formula in the module.
        out = build_features(_row_frame())
        t_c, d_c = 300.0 - 273.15, 290.0 - 273.15
        es_t = 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))
        es_d = 6.112 * math.exp(17.67 * d_c / (d_c + 243.5))
        assert out["rh_pct"].iloc[0] == pytest.approx(100.0 * es_d / es_t, rel=1e-9)

    def test_rh_saturates_at_100(self):
        # dewpoint above air temperature is unphysical but must not explode
        out = build_features(_row_frame(t2m_k_inst=290.0, d2m_k_inst=300.0))
        assert 0.0 <= out["rh_pct"].iloc[0] <= 100.0
        assert out["rh_pct"].iloc[0] == pytest.approx(100.0)

    def test_rh_dry_air_low(self):
        # 30 C air, 0 C dew -> very dry
        out = build_features(_row_frame(t2m_k_inst=303.15, d2m_k_inst=273.15))
        assert out["rh_pct"].iloc[0] < 15.0

    def test_pressure_pa_to_hpa(self):
        out = build_features(_row_frame())
        assert out["sp_hpa_inst"].iloc[0] == pytest.approx(1000.0)

    def test_temp_k_to_c(self):
        out = build_features(_row_frame())
        assert out["t2m_deg_c"].iloc[0] == pytest.approx(26.85)


class TestCircularEncodings:
    def test_wind_dir_unit_vector_norm_le_1(self):
        out = build_features(_row_frame(wind_u10_inst=-0.3, wind_v10_inst=2.5))
        norm = math.hypot(out["wind_dir_cos_inst"].iloc[0],
                          out["wind_dir_sin_inst"].iloc[0])
        assert norm == pytest.approx(1.0)

    def test_wind_dir_matches_velocity_direction(self):
        # u>0, v=0 -> pointing +x -> cos=1, sin=0
        out = build_features(_row_frame(wind_u10_inst=2.0, wind_v10_inst=0.0))
        assert out["wind_dir_cos_inst"].iloc[0] == pytest.approx(1.0)
        assert out["wind_dir_sin_inst"].iloc[0] == pytest.approx(0.0)

    def test_zero_wind_gets_neutral_encoding_not_nan(self):
        out = build_features(_row_frame(wind_u10_inst=0.0, wind_v10_inst=0.0))
        assert out["wind_dir_cos_inst"].iloc[0] == 0.5
        assert out["wind_dir_sin_inst"].iloc[0] == 0.5
        assert out["wind_speed_inst"].iloc[0] == 0.0

    def test_month_encoding_period_12(self):
        # month 1 -> angle 0; month 7 -> angle pi (opposite phase)
        jan = build_features(_row_frame(month=1)).iloc[0]
        jul = build_features(_row_frame(month=7)).iloc[0]
        assert jan["month_sin"] == pytest.approx(0.0, abs=1e-12)
        assert jan["month_cos"] == pytest.approx(1.0)
        assert jul["month_sin"] == pytest.approx(0.0, abs=1e-12)
        assert jul["month_cos"] == pytest.approx(-1.0)

    def test_month_encoding_never_repeats_within_year(self):
        rows = [_row_frame(month=m).iloc[0].to_dict() for m in range(1, 13)]
        out = build_features(pd.DataFrame(rows))
        pairs = set(zip(out["month_sin"], out["month_cos"]))
        assert len(pairs) == 12


class TestProxies:
    def test_ventilation_proxy(self):
        out = build_features(_row_frame())  # 5 m/s * 1000 m / 1000
        assert out["ventilation_proxy"].iloc[0] == pytest.approx(5.0)

    def test_blh_x_precip_uses_filled_precip(self):
        out = build_features(_row_frame(precip_l24_m=np.nan, blh_m_inst=2000.0))
        assert out["blh_x_precip_proxy"].iloc[0] == 0.0  # NaN precip -> 0

    def test_ventilation_lags_use_inst_blh(self):
        out = build_features(_row_frame())
        assert out["ventilation_l24"].iloc[0] == pytest.approx(1.0)
        assert out["ventilation_l48"].iloc[0] == pytest.approx(2.0)


class TestContract:
    def test_idempotent(self):
        once = build_features(_row_frame())
        twice = build_features(once)
        pd.testing.assert_frame_equal(once, twice)

    def test_does_not_mutate_input(self):
        df = _row_frame()
        before = df.copy()
        build_features(df)
        pd.testing.assert_frame_equal(df, before)

    def test_no_no2_covariate_leaks_into_features(self):
        # parity guard: the target must never become a FEATURE. It rides in
        # the frame (supervised training needs it) but no feature group lists
        # it and build_features must not derive anything named like NO2.
        banned = TARGET
        for cols in FEATURE_GROUPS.values():
            assert banned not in cols
        df = _row_frame()
        built = build_features(df)
        derived = set(built.columns) - set(df.columns)
        assert not derived & {"value_umol_m2", "no2", "tropospheric_no2"}

    def test_missing_required_column_raises(self):
        df = _row_frame().drop(columns=["d2m_k_inst"])
        with pytest.raises(KeyError, match="d2m_k_inst"):
            build_features(df)

    def test_schema_version_is_pinned(self):
        assert FEATURE_SCHEMA_VERSION == "fm-v1-2026-10-24"


class TestFeatureGroups:
    def test_unknown_group_raises(self):
        with pytest.raises(KeyError, match="no_such_group"):
            feature_block(pd.DataFrame({"a": [1]}), ["no_such_group"])

    def test_unbuilt_columns_raise_with_hint(self):
        with pytest.raises(KeyError, match="build_features"):
            feature_block(_row_frame(), ["weather_inst"])

    def test_group_columns_present_after_build(self):
        built = build_features(_row_frame())
        for group, cols in FEATURE_GROUPS.items():
            block = feature_block(built, [group])
            assert block == cols

    def test_groups_do_not_overlap(self):
        seen: set[str] = set()
        for cols in FEATURE_GROUPS.values():
            assert not (seen & set(cols)), "feature listed in two groups"
            seen |= set(cols)


class TestCleanTrainingFrame:
    def test_keeps_only_observed_with_target(self):
        df = pd.concat([
            _row_frame(),
            _row_frame(state="no_overpass"),
            _row_frame(state="insufficient"),
            _row_frame(value_umol_m2=np.nan),
        ], ignore_index=True)
        cleaned = clean_training_frame(df)
        assert len(cleaned) == 1
        assert cleaned["state"].iloc[0] == "observed"

    def test_target_column_name_is_locked(self):
        assert TARGET == "value_umol_m2"
