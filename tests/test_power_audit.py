"""
Tests for Module 4 — power audit back-calculations.
"""

from __future__ import annotations

import pytest

from src.pipeline.module4_power_audit import _back_calculate_hr


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
