"""Tests for the ephemeris module."""

import pytest
from datetime import date

from src.ephemeris import (
    date_to_jd, birth_to_jd, calc_planet_positions, calc_houses,
    calc_full_chart, generate_candidate_grid, angle_diff,
    secondary_progressed_jd, solar_arc_for_age, profected_asc, sign_ruler,
)
from src.models import HouseSystem


class TestJulianDay:
    def test_known_date_jd(self):
        # J2000.0: 2000-01-01 12:00 UT = JD 2451545.0
        jd = date_to_jd(date(2000, 1, 1), hour_ut=12.0)
        assert abs(jd - 2451545.0) < 0.01

    def test_birth_to_jd_noon(self):
        # 12:00 local time with tz=0 should equal date_to_jd at noon
        jd_birth = birth_to_jd(date(2000, 1, 1), time_minutes=720, tz_offset=0.0)
        jd_direct = date_to_jd(date(2000, 1, 1), hour_ut=12.0)
        assert abs(jd_birth - jd_direct) < 1e-6

    def test_birth_to_jd_with_timezone(self):
        # 15:00 local (UTC+3) = 12:00 UT
        jd_local = birth_to_jd(date(2000, 1, 1), time_minutes=900, tz_offset=3.0)
        jd_ut = date_to_jd(date(2000, 1, 1), hour_ut=12.0)
        assert abs(jd_local - jd_ut) < 1e-6

    def test_time_minutes_range(self):
        jd_midnight = birth_to_jd(date(2000, 6, 15), time_minutes=0, tz_offset=0.0)
        jd_noon = birth_to_jd(date(2000, 6, 15), time_minutes=720, tz_offset=0.0)
        assert abs(jd_noon - jd_midnight - 0.5) < 1e-6


class TestPlanetPositions:
    def test_returns_ten_planets(self):
        jd = date_to_jd(date(2000, 1, 1), 12.0)
        positions = calc_planet_positions(jd)
        assert len(positions) == 10

    def test_all_positions_in_range(self):
        jd = date_to_jd(date(2000, 1, 1), 12.0)
        positions = calc_planet_positions(jd)
        for name, lon in positions.items():
            assert 0 <= lon < 360, f"{name} longitude {lon} out of range"

    def test_sun_in_capricorn_on_jan1(self):
        # Jan 1 2000: Sun should be in Capricorn (270–300°)
        jd = date_to_jd(date(2000, 1, 1), 12.0)
        positions = calc_planet_positions(jd)
        assert 260 <= positions["SUN"] <= 295, f"Sun at {positions['SUN']}"

    def test_sun_in_cancer_on_july4(self):
        # July 4 2000: Sun should be in Cancer (90–120°)
        jd = date_to_jd(date(2000, 7, 4), 12.0)
        positions = calc_planet_positions(jd)
        assert 100 <= positions["SUN"] <= 115, f"Sun at {positions['SUN']}"


class TestHouseCalculations:
    def test_houses_return_twelve_cusps(self):
        jd = date_to_jd(date(2000, 1, 1), 12.0)
        cusps, asc, mc = calc_houses(jd, 51.5, -0.12, HouseSystem.PLACIDUS)
        assert len(cusps) == 12

    def test_asc_in_range(self):
        jd = date_to_jd(date(2000, 1, 1), 12.0)
        _, asc, mc = calc_houses(jd, 51.5, -0.12, HouseSystem.PLACIDUS)
        assert 0 <= asc < 360
        assert 0 <= mc < 360

    def test_asc_differs_between_times(self):
        jd_morning = birth_to_jd(date(2000, 6, 15), 360, 1.0)   # 06:00
        jd_evening = birth_to_jd(date(2000, 6, 15), 1080, 1.0)  # 18:00
        _, asc_m, _ = calc_houses(jd_morning, 51.5, -0.12, HouseSystem.PLACIDUS)
        _, asc_e, _ = calc_houses(jd_evening, 51.5, -0.12, HouseSystem.PLACIDUS)
        assert abs(asc_m - asc_e) > 5.0

    def test_all_house_systems_produce_asc(self):
        jd = date_to_jd(date(2000, 1, 1), 12.0)
        for hs in HouseSystem:
            cusps, asc, mc = calc_houses(jd, 51.5, -0.12, hs)
            assert 0 <= asc < 360, f"ASC out of range for {hs}"
            assert len(cusps) == 12


class TestCandidateGrid:
    def test_fifteen_minute_grid_count(self):
        # 1440 / 15 = 96 candidates maximum
        candidates = generate_candidate_grid(
            date(1985, 3, 10), 48.85, 2.35, 1.0, interval_minutes=15
        )
        assert len(candidates) <= 96
        assert len(candidates) > 0

    def test_filtered_by_rising_sign(self):
        all_candidates = generate_candidate_grid(
            date(1985, 3, 10), 48.85, 2.35, 1.0, interval_minutes=15
        )
        # Get one rising sign from the full grid
        sign = all_candidates[0].rising_sign()
        filtered = generate_candidate_grid(
            date(1985, 3, 10), 48.85, 2.35, 1.0,
            interval_minutes=15, rising_signs=[sign]
        )
        assert all(c.rising_sign() == sign for c in filtered)
        assert len(filtered) <= len(all_candidates)

    def test_candidates_have_planets(self):
        candidates = generate_candidate_grid(
            date(1985, 3, 10), 48.85, 2.35, 1.0, interval_minutes=30
        )
        for c in candidates[:3]:
            assert len(c.planets) == 10
            assert "SUN" in c.planets

    def test_one_minute_resolution(self):
        candidates = generate_candidate_grid(
            date(1985, 3, 10), 48.85, 2.35, 1.0,
            interval_minutes=1, rising_signs=[1]  # only Aries rising
        )
        # Should be roughly 120 candidates (2 hours of Aries rising at 1-min)
        assert 50 <= len(candidates) <= 200


class TestAngleDiff:
    def test_zero_diff(self):
        assert angle_diff(180.0, 180.0) == 0.0

    def test_conjunction(self):
        assert angle_diff(10.0, 10.5) == pytest.approx(0.5)

    def test_opposition(self):
        assert angle_diff(0.0, 180.0) == pytest.approx(180.0)

    def test_wraparound(self):
        assert angle_diff(359.0, 1.0) == pytest.approx(2.0)

    def test_always_positive(self):
        assert angle_diff(350.0, 10.0) >= 0
        assert angle_diff(10.0, 350.0) >= 0


class TestProgressedAndArc:
    def test_secondary_progression(self):
        natal_jd = date_to_jd(date(1970, 1, 1), 12.0)
        prog_jd = secondary_progressed_jd(natal_jd, 30.0)
        assert abs(prog_jd - natal_jd - 30.0) < 1e-6

    def test_solar_arc_positive(self):
        natal_jd = date_to_jd(date(1970, 1, 1), 12.0)
        arc = solar_arc_for_age(natal_jd, 20.0)
        assert arc > 0
        assert arc < 360

    def test_solar_arc_naibod_approx(self):
        # Arc for 30 years should be ~29.5° (Naibod: 0.9856°/yr × 30 ≈ 29.57°)
        natal_jd = date_to_jd(date(1970, 1, 1), 12.0)
        arc = solar_arc_for_age(natal_jd, 30.0)
        assert 25 <= arc <= 35

    def test_profected_asc_one_year(self):
        # After 1 year, ASC advances 30°
        result = profected_asc(0.0, 1)
        assert result == pytest.approx(30.0)

    def test_profected_asc_twelve_years(self):
        # After 12 years, ASC should be back to natal (mod 360)
        result = profected_asc(15.0, 12)
        assert result == pytest.approx(15.0)

    def test_sign_ruler_returns_planet(self):
        known_rulers = {
            0.0: "MARS",     # Aries
            30.0: "VENUS",   # Taurus
            90.0: "MOON",    # Cancer
            120.0: "SUN",    # Leo
            270.0: "SATURN", # Capricorn
        }
        for lon, expected in known_rulers.items():
            assert sign_ruler(lon) == expected
