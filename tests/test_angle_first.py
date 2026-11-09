"""Tests for the angle-first scorer (Tebbs-style: angles only, solar arc first)."""

from datetime import date, timedelta

import pytest

from src.ephemeris import birth_to_jd, generate_candidate_grid, solar_arc_for_age
from src.models import EventType, EventWeight, LifeEvent
from src.scoring.angle_first import AngleFirstConfig, AngleFirstScorer

BIRTH, LAT, LON, TZ = date(1970, 6, 15), 40.0, -3.7, 1.0
TRUE_MINUTE = 10 * 60 + 37


@pytest.fixture(scope="module")
def chart():
    return generate_candidate_grid(BIRTH, LAT, LON, TZ, interval_minutes=1)[TRUE_MINUTE]


@pytest.fixture(scope="module")
def natal_jd():
    return birth_to_jd(BIRTH, 720, TZ)


def event_when_sun_reaches_mc(chart, event_type=EventType.CAREER_PEAK):
    """An event dated when the solar-arc Sun is exactly on the MC (or IC/ASC/DSC)."""
    for angle in (chart.mc, chart.ic, chart.ascendant, chart.dsc):
        arc = (angle - chart.planets["SUN"]) % 360
        if 8 < arc < 70:
            age = arc
            for _ in range(5):
                age += (arc - solar_arc_for_age(chart.julian_day, age)) / 0.9856
            return LifeEvent(description="planted", event_type=event_type, weight=EventWeight.ANCHOR,
                             date=BIRTH + timedelta(days=round(age * 365.25)))
    pytest.skip("no usable arc for this chart")


def test_exact_solar_arc_contact_scores_near_one(chart, natal_jd):
    ev = event_when_sun_reaches_mc(chart)
    sa, _, _ = AngleFirstScorer().components(chart, ev, natal_jd)
    assert sa[0] > 0.95 and sa[2] < 0.05


def test_contact_fades_with_orb_and_stops_at_one_degree(chart, natal_jd):
    ev = event_when_sun_reaches_mc(chart)
    scorer = AngleFirstScorer(config=AngleFirstConfig(mode="sa_only"))
    half_year = ev.model_copy(update={"date": ev.date + timedelta(days=183)})
    two_years = ev.model_copy(update={"date": ev.date + timedelta(days=730)})
    full = scorer.components(chart, ev, natal_jd)[0][0]
    half = scorer.components(chart, half_year, natal_jd)[0]
    assert 0.3 < half[0] < 0.7 < full
    # two years on, that contact is 2° away: any score must come from another contact
    far = scorer.components(chart, two_years, natal_jd)[0]
    assert far[1] != half[1] or far[0] == 0.0


def test_one_hit_per_technique_per_event(chart, natal_jd):
    ev = event_when_sun_reaches_mc(chart)
    hits = AngleFirstScorer().score_event(chart, ev, natal_jd)
    techniques = [h.technique for h in hits]
    assert len(techniques) == len(set(techniques)) <= 3
    assert "solar_arc" in techniques


def test_sa_only_mode_ignores_confirmations(chart, natal_jd):
    ev = event_when_sun_reaches_mc(chart)
    hits = AngleFirstScorer(config=AngleFirstConfig(mode="sa_only")).score_event(chart, ev, natal_jd)
    assert [h.technique for h in hits] == ["solar_arc"]


def test_strict_mode_needs_all_three(chart, natal_jd):
    ev = event_when_sun_reaches_mc(chart)
    scorer = AngleFirstScorer(config=AngleFirstConfig(mode="strict"))
    comps = scorer.components(chart, ev, natal_jd)
    hits = scorer.score_event(chart, ev, natal_jd)
    assert (len(hits) == 3) == all(c[0] > 0 for c in comps)


def test_painful_events_weigh_four_times_more(chart, natal_jd):
    cfg = AngleFirstConfig(mode="sa_only")
    pleasant = event_when_sun_reaches_mc(chart, EventType.CAREER_PEAK)
    painful = pleasant.model_copy(update={"event_type": EventType.DEATH_OF_PARENT})
    s1 = sum(h.score for h in AngleFirstScorer(config=cfg).score_event(chart, pleasant, natal_jd))
    s2 = sum(h.score for h in AngleFirstScorer(config=cfg).score_event(chart, painful, natal_jd))
    assert s2 == pytest.approx(4 * s1)


def test_house_cusps_do_not_matter(chart, natal_jd):
    ev = event_when_sun_reaches_mc(chart)
    shuffled = chart.model_copy(update={"house_cusps": [(c + 11) % 360 for c in chart.house_cusps]})
    a = AngleFirstScorer().score_event(chart, ev, natal_jd)
    b = AngleFirstScorer().score_event(shuffled, ev, natal_jd)
    assert [h.score for h in a] == [h.score for h in b]


def test_events_before_birth_score_nothing(chart, natal_jd):
    ev = LifeEvent(description="x", event_type=EventType.OTHER, date=date(1960, 1, 1), weight=EventWeight.ANCHOR)
    assert AngleFirstScorer().score_event(chart, ev, natal_jd) == []
