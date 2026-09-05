"""
Tests for Module 4 — power audit back-calculations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline import module4_power_audit as m4
from src.pipeline.module4_power_audit import _back_calculate_hr, _write_power_audit_log


class TestBackCalculateHR:
    def test_large_trial_implies_modest_hr(self):
        # A large trial (n=5000, 15% event rate) should imply HR close to null
        hr = _back_calculate_hr(enrollment=5000, event_rate=0.15)
        assert 0.70 < hr < 1.0, f"Unexpected HR={hr}"

    def test_small_trial_implies_optimistic_hr(self):
        # A small trial needs a larger effect to be powered
        hr_small = _back_calculate_hr(enrollment=200, event_rate=0.15)
        hr_large = _back_calculate_hr(enrollment=5000, event_rate=0.15)
        assert hr_small < hr_large, "Smaller trial should require more optimistic HR"

    def test_hr_is_less_than_one(self):
        # Beneficial direction
        hr = _back_calculate_hr(enrollment=1000, event_rate=0.20)
        assert hr < 1.0

    def test_invalid_enrollment_raises(self):
        with pytest.raises(ValueError):
            _back_calculate_hr(enrollment=0, event_rate=0.15)

    def test_result_is_rounded_to_4dp(self):
        hr = _back_calculate_hr(enrollment=1000, event_rate=0.15)
        assert hr == round(hr, 4)


def test_power_audit_log_is_rewritten_one_row_per_nct(tmp_path, monkeypatch) -> None:
    path = tmp_path / "power_audit_log.csv"
    monkeypatch.setattr(m4, "POWER_AUDIT_LOG_PATH", path)

    _write_power_audit_log([{"nct_id": "NCT001", "optimism_bias": "-0.10"}])
    _write_power_audit_log(
        [
            {"nct_id": "NCT001", "optimism_bias": "-0.20"},  # newer run for the same trial
            {"nct_id": "NCT002", "optimism_bias": "-0.05"},
        ]
    )

    log = pd.read_csv(path, dtype=str)
    assert list(log["nct_id"]) == ["NCT001", "NCT002"]  # no append, no duplicate
    assert float(log.loc[log["nct_id"] == "NCT001", "optimism_bias"].iloc[0]) == -0.20
