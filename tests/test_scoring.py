"""Tests for the scoring techniques."""

import pytest
from datetime import date

from src.models import (
    CandidateChart, LifeEvent, EventType, EventWeight, HouseSystem,
)
from src.ephemeris import date_to_jd, calc_full_chart, birth_to_jd
from src.scoring.base import angle_diff, orb_score
from src.scoring.transits import TransitScorer
from src.scoring.progressions import ProgressionScorer
from src.scoring.solar_arc import SolarArcScorer
from src.scoring.profections import ProfectionScorer
from src.scoring.primary_directions import PrimaryDirectionScorer, ecliptic_to_equatorial


# --- Fixtures ---

@pytest.fixture
def natal_jd():
    return birth_to_jd(date(1964, 5, 23), 720, 1.0)  # noon BST


@pytest.fixture
def sample_candidate(natal_jd):
    jd = birth_to_jd(date(1964, 5, 23), 480, 1.0)  # 08:00
    chart = calc_full_chart(jd, 51.5074, -0.1278, HouseSystem.PLACIDUS)
    return CandidateChart(
        time_minutes=480,
        julian_day=jd,
        ascendant=chart["asc"],
        mc=chart["mc"],
        house_cusps=chart["cusps"],
        house_system=HouseSystem.PLACIDUS,
        planets=chart["planets"],
    )


@pytest.fixture
def marriage_event():
    return LifeEvent(
        description="First marriage",
        event_type=EventType.MARRIAGE,
        date=date(1988, 7, 14),
        date_certainty_days=1,
        weight=EventWeight.ANCHOR,
        held_out=False,
    )


@pytest.fixture
def accident_event():
    return LifeEvent(
        description="Motorcycle accident",
        event_type=EventType.ACCIDENT,
        date=date(1995, 9, 3),
        date_certainty_days=3,
        weight=EventWeight.ANCHOR,
        held_out=False,
    )


@pytest.fixture
def held_out_event():
    return LifeEvent(
        description="Second marriage (held out)",
        event_type=EventType.MARRIAGE,
        date=date(2001, 6, 10),
        date_certainty_days=1,
        weight=EventWeight.ANCHOR,
        held_out=True,
    )


# --- Base utilities ---

class TestOrbScore:
    def test_tight_orb_gives_max_score(self):
        assert orb_score(0.3) == 3.0

    def test_medium_orb(self):
        assert orb_score(0.7) == 2.0

    def test_loose_orb(self):
        assert orb_score(1.5) == 1.0

    def test_outside_orb_zero(self):
        assert orb_score(3.0) == 0.0

    def test_tight_mode(self):
        assert orb_score(0.2, tight=True) == 3.0
        assert orb_score(0.4, tight=True) == 2.0
        assert orb_score(0.8, tight=True) == 1.0
        assert orb_score(1.5, tight=True) == 0.0


# --- Transit scorer ---

class TestTransitScorer:
    def test_returns_list(self, sample_candidate, marriage_event, natal_jd):
        scorer = TransitScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        assert isinstance(hits, list)

    def test_all_scores_positive(self, sample_candidate, marriage_event, natal_jd):
        scorer = TransitScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.score > 0

    def test_technique_name(self, sample_candidate, marriage_event, natal_jd):
        scorer = TransitScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.technique == "transit"

    def test_held_out_flag_propagated(self, sample_candidate, held_out_event, natal_jd):
        scorer = TransitScorer()
        hits = scorer.score_event(sample_candidate, held_out_event, natal_jd)
        for hit in hits:
            assert hit.held_out is True

    def test_soft_event_lower_score(self, sample_candidate, natal_jd):
        scorer = TransitScorer()
        anchor = LifeEvent(
            description="Event A",
            event_type=EventType.CAREER_PEAK,
            date=date(1990, 3, 21),
            weight=EventWeight.ANCHOR,
        )
        soft = LifeEvent(
            description="Event A soft",
            event_type=EventType.CAREER_PEAK,
            date=date(1990, 3, 21),
            weight=EventWeight.SOFT,
        )
        hits_anchor = scorer.score_event(sample_candidate, anchor, natal_jd)
        hits_soft = scorer.score_event(sample_candidate, soft, natal_jd)

        if hits_anchor and hits_soft:
            total_anchor = sum(h.score for h in hits_anchor)
            total_soft = sum(h.score for h in hits_soft)
            assert total_soft < total_anchor

    def test_score_all_events(self, sample_candidate, natal_jd, marriage_event, accident_event):
        scorer = TransitScorer()
        hits = scorer.score_all_events(
            sample_candidate, [marriage_event, accident_event], natal_jd
        )
        assert isinstance(hits, list)

    def test_tight_mode_smaller_orbs(self, sample_candidate, marriage_event, natal_jd):
        standard = TransitScorer(tight=False)
        tight = TransitScorer(tight=True)
        hits_std = standard.score_event(sample_candidate, marriage_event, natal_jd)
        hits_tight = tight.score_event(sample_candidate, marriage_event, natal_jd)
        # Tight mode should produce fewer or equal hits
        assert len(hits_tight) <= len(hits_std)


# --- Progression scorer ---

class TestProgressionScorer:
    def test_returns_list(self, sample_candidate, marriage_event, natal_jd):
        scorer = ProgressionScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        assert isinstance(hits, list)

    def test_technique_prefix(self, sample_candidate, marriage_event, natal_jd):
        scorer = ProgressionScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.technique.startswith("progression")

    def test_held_out_propagated(self, sample_candidate, held_out_event, natal_jd):
        scorer = ProgressionScorer()
        hits = scorer.score_event(sample_candidate, held_out_event, natal_jd)
        for hit in hits:
            assert hit.held_out is True

    def test_future_event_before_birth_returns_empty(self, sample_candidate, natal_jd):
        before_birth = LifeEvent(
            description="Before birth",
            event_type=EventType.OTHER,
            date=date(1960, 1, 1),
            weight=EventWeight.SOFT,
        )
        scorer = ProgressionScorer()
        hits = scorer.score_event(sample_candidate, before_birth, natal_jd)
        assert hits == []


# --- Solar arc scorer ---

class TestSolarArcScorer:
    def test_returns_list(self, sample_candidate, marriage_event, natal_jd):
        scorer = SolarArcScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        assert isinstance(hits, list)

    def test_technique_name(self, sample_candidate, marriage_event, natal_jd):
        scorer = SolarArcScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.technique == "solar_arc"

    def test_naibod_mode(self, sample_candidate, marriage_event, natal_jd):
        scorer_actual = SolarArcScorer(use_naibod=False)
        scorer_naibod = SolarArcScorer(use_naibod=True)
        hits_actual = scorer_actual.score_event(sample_candidate, marriage_event, natal_jd)
        hits_naibod = scorer_naibod.score_event(sample_candidate, marriage_event, natal_jd)
        # Both modes should return lists (results may differ)
        assert isinstance(hits_actual, list)
        assert isinstance(hits_naibod, list)


# --- Profection scorer ---

class TestProfectionScorer:
    def test_returns_list(self, sample_candidate, marriage_event, natal_jd):
        scorer = ProfectionScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        assert isinstance(hits, list)

    def test_technique_prefix(self, sample_candidate, marriage_event, natal_jd):
        scorer = ProfectionScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.technique.startswith("profection")


# --- Primary direction scorer ---

class TestPrimaryDirectionScorer:
    def test_ecliptic_to_equatorial_range(self):
        ra, dec = ecliptic_to_equatorial(90.0, 0.0, 23.45)  # Cancer ingress
        assert 0 <= ra < 360
        assert -90 <= dec <= 90

    def test_ecliptic_to_equatorial_aries(self):
        # Aries 0° should convert to RA ~0° (along ecliptic)
        ra, dec = ecliptic_to_equatorial(0.0, 0.0, 23.45)
        assert abs(ra) < 5 or abs(ra - 360) < 5

    def test_returns_list(self, sample_candidate, marriage_event, natal_jd):
        scorer = PrimaryDirectionScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        assert isinstance(hits, list)

    def test_technique_name(self, sample_candidate, marriage_event, natal_jd):
        scorer = PrimaryDirectionScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.technique == "primary_direction"

    def test_scores_positive(self, sample_candidate, marriage_event, natal_jd):
        scorer = PrimaryDirectionScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.score > 0

    def test_tight_mode_fewer_hits(self, sample_candidate, marriage_event, natal_jd):
        standard = PrimaryDirectionScorer(tight=False)
        tight = PrimaryDirectionScorer(tight=True)
        hits_std = standard.score_event(sample_candidate, marriage_event, natal_jd)
        hits_tight = tight.score_event(sample_candidate, marriage_event, natal_jd)
        assert len(hits_tight) <= len(hits_std)


# --- Cross-scorer consistency tests ---

class TestScorerConsistency:
    """Verify all scorers handle edge cases consistently."""

    def test_all_scorers_same_candidate(self, sample_candidate, marriage_event, natal_jd):
        scorers = [
            TransitScorer(), ProgressionScorer(), SolarArcScorer(),
            ProfectionScorer(), PrimaryDirectionScorer(),
        ]
        for scorer in scorers:
            hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
            assert isinstance(hits, list), f"{scorer.name} did not return a list"
            for hit in hits:
                assert hit.score > 0, f"{scorer.name} returned non-positive score"
                assert 0 <= hit.orb <= 180, f"{scorer.name} returned invalid orb"

    def test_time_minutes_recorded_in_scores(self, sample_candidate, marriage_event, natal_jd):
        scorer = TransitScorer()
        hits = scorer.score_event(sample_candidate, marriage_event, natal_jd)
        for hit in hits:
            assert hit.time_minutes == sample_candidate.time_minutes
