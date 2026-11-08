"""Tests for the chance-level gate (real birth date vs unrelated natives)."""

from datetime import date

import pytest

import src.rectifier as R
from src.models import BirthData, EventType, EventWeight, LifeEvent
from src.morin_filter import uniform_prior


@pytest.fixture
def birth():
    return BirthData(name="Test", birth_date=date(1975, 3, 10), birth_city="Lisbon",
                     latitude=38.72, longitude=-9.14, timezone_offset=0.0)


@pytest.fixture
def events():
    specs = [
        (EventType.MARRIAGE, date(2001, 6, 2)), (EventType.CAREER_PEAK, date(2004, 9, 15)),
        (EventType.DEATH_OF_PARENT, date(2008, 1, 20)), (EventType.RELOCATION, date(2012, 5, 3)),
    ]
    return [LifeEvent(description=f"e{i}", event_type=t, date=d, weight=EventWeight.ANCHOR)
            for i, (t, d) in enumerate(specs)]


def test_p_value_has_the_permutation_form(birth, events):
    p, real, null = R.chance_level_test(birth, events, n_natives=4, seed=1)
    assert len(null) == 4
    assert real > 0
    assert p == (1 + sum(x >= real for x in null)) / 5


def test_is_deterministic_for_a_seed(birth, events):
    a = R.chance_level_test(birth, events, n_natives=3, seed=7)
    b = R.chance_level_test(birth, events, n_natives=3, seed=7)
    assert a == b


def test_no_anchor_events_gives_p_one(birth, events):
    held = [e.model_copy(update={"held_out": True}) for e in events]
    assert R.chance_level_test(birth, held, n_natives=3)[0] == 1.0


def test_rectify_refuses_a_confident_time_at_chance_level(birth, events, monkeypatch):
    monkeypatch.setattr(R, "chance_level_test", lambda *a, **k: (0.5, 10.0, [10.0]))
    result = R.Rectifier(birth, events, morin_prior=uniform_prior()).rectify()
    assert result.chance_p_value == 0.5
    assert result.confidence_score == 0.0
    assert result.uncertainty_minutes == 720
    assert result.is_provisional
    assert "chance level" in result.notes


def test_significant_result_keeps_a_time(birth, events, monkeypatch):
    monkeypatch.setattr(R, "chance_level_test", lambda *a, **k: (0.01, 99.0, [1.0]))
    result = R.Rectifier(birth, events, morin_prior=uniform_prior()).rectify()
    assert result.chance_p_value == 0.01
    assert "chance level" not in result.notes
