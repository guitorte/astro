"""Tests for the window comparison (recorded time vs its neighbours)."""

from datetime import date, timedelta

import pytest

import src.window as W
from src.ephemeris import generate_candidate_grid, solar_arc_for_age
from src.models import BirthData, EventType, EventWeight, LifeEvent

BIRTH = BirthData(name="t", birth_date=date(1970, 6, 15), birth_city="x",
                  latitude=40.0, longitude=-3.7, timezone_offset=1.0)
TRUE = 10 * 60 + 37


def planted(n=8):
    chart = generate_candidate_grid(BIRTH.birth_date, 40.0, -3.7, 1.0, interval_minutes=1,
                                    start_minute=TRUE, end_minute=TRUE)[0]
    ages = sorted(
        arc for p in chart.planets.values() for a in (chart.ascendant, chart.mc, chart.dsc, chart.ic)
        for arc in ((a - p) % 360, (p - a) % 360) if 8 < arc < 60
    )
    events, last = [], -9
    for arc in ages:
        age = arc
        for _ in range(4):
            age += (arc - solar_arc_for_age(chart.julian_day, age)) / 0.9856
        if age - last < 2:
            continue
        last = age
        events.append(LifeEvent(description=f"e{len(events)}", event_type=EventType.CAREER_PEAK,
                                weight=EventWeight.ANCHOR,
                                date=BIRTH.birth_date + timedelta(days=round(age * 365.25))))
        if len(events) == n:
            break
    return events


def test_grid_range_arguments():
    c = generate_candidate_grid(BIRTH.birth_date, 40.0, -3.7, 1.0, interval_minutes=1,
                                start_minute=600, end_minute=610)
    assert [x.time_minutes for x in c] == list(range(600, 611))


def test_slots_cover_the_window_and_mark_the_recorded_time():
    rep = W.compare_window(BIRTH, planted(), TRUE + 12, n_natives=3)
    assert rep.slots[0].start == "10:19" and rep.slots[-1].end == "11:19"
    assert sum(s.contains_recorded_time for s in rep.slots) == 1
    assert rep.recorded_time == "10:49" and rep.anchor_events == 8


def test_planted_time_is_found_inside_the_window():
    rep = W.compare_window(BIRTH, planted(), TRUE + 12, n_natives=3)
    assert abs((TRUE + 12 + rep.minutes_from_recorded) - TRUE) <= 3
    assert rep.best_score > rep.recorded_score


def test_no_preference_when_the_chance_test_fails(monkeypatch):
    monkeypatch.setattr(W, "window_chance_test", lambda *a, **k: (0.6, 1.0, [1.0]))
    rep = W.compare_window(BIRTH, planted(), TRUE + 12)
    assert rep.verdict == "no_preference"
    assert "Keep the recorded time" in rep.summary


def test_prefers_other_and_supports_recorded(monkeypatch):
    monkeypatch.setattr(W, "window_chance_test", lambda *a, **k: (0.03, 9.0, [1.0]))
    assert W.compare_window(BIRTH, planted(), TRUE + 20).verdict == "prefers_other"
    assert W.compare_window(BIRTH, planted(), TRUE).verdict == "supports_recorded"


def test_sign_changes_are_reported():
    # Synthetic chart: around 05:00 the MC enters Sagittarius (04:47) and the ASC Pisces (05:13).
    b = BirthData(name="s", birth_date=date(1980, 3, 3), birth_city="synthetic",
                  latitude=10.0, longitude=20.0, timezone_offset=1.0)
    ev = [LifeEvent(description="e", event_type=EventType.RELOCATION, date=date(2005, 5, 5))]
    rep = W.compare_window(b, ev, 5 * 60, n_natives=2)
    got = {(c.angle, c.to_sign, c.time) for c in rep.sign_changes}
    assert got == {("MC", "Sagittarius", "04:47"), ("ASC", "Pisces", "05:13")}


def test_window_chance_test_has_permutation_form():
    p, real, null = W.window_chance_test(BIRTH, planted(), TRUE - 30, TRUE + 30, n_natives=4, seed=2)
    assert len(null) == 4 and p == (1 + sum(x >= real for x in null)) / 5
