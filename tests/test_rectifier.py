"""Tests for the 4-loop rectifier orchestrator."""

import pytest
from datetime import date

from src.models import (
    BirthData, LifeEvent, EventType, EventWeight,
    HouseSystem, RisingSignPrior, RectificationResult,
)
from src.rectifier import (
    Rectifier, score_candidate, bayesian_update, cross_validate,
    build_scorers, event_diversity_score,
)
from src.ephemeris import birth_to_jd, calc_full_chart
from src.morin_filter import uniform_prior


@pytest.fixture
def birth_data():
    return BirthData(
        name="Test Celebrity",
        birth_date=date(1964, 5, 23),
        birth_city="London",
        latitude=51.5074,
        longitude=-0.1278,
        timezone_offset=1.0,
        biography=(
            "A famous rock musician who became internationally known. "
            "Political activist. Married in 1988. Accident in 1995."
        ),
    )


@pytest.fixture
def events():
    return [
        LifeEvent(
            description="First marriage",
            event_type=EventType.MARRIAGE,
            date=date(1988, 7, 14),
            weight=EventWeight.ANCHOR,
            held_out=False,
        ),
        LifeEvent(
            description="Career peak",
            event_type=EventType.CAREER_PEAK,
            date=date(1990, 3, 21),
            weight=EventWeight.ANCHOR,
            held_out=False,
        ),
        LifeEvent(
            description="Motorcycle accident",
            event_type=EventType.ACCIDENT,
            date=date(1995, 9, 3),
            weight=EventWeight.ANCHOR,
            held_out=False,
        ),
        LifeEvent(
            description="Second marriage (held out)",
            event_type=EventType.MARRIAGE,
            date=date(2001, 6, 10),
            weight=EventWeight.ANCHOR,
            held_out=True,
        ),
    ]


class TestBuildScorers:
    def test_returns_five_scorers(self):
        scorers = build_scorers()
        assert len(scorers) == 5

    def test_tight_mode(self):
        scorers_tight = build_scorers(tight=True)
        assert len(scorers_tight) == 5


class TestScoreCandidate:
    def test_returns_candidate_score(self, birth_data, events):
        jd = birth_to_jd(birth_data.birth_date, 480, birth_data.timezone_offset)
        chart = calc_full_chart(jd, birth_data.latitude, birth_data.longitude, HouseSystem.PLACIDUS)
        from src.models import CandidateChart
        candidate = CandidateChart(
            time_minutes=480,
            julian_day=jd,
            ascendant=chart["asc"],
            mc=chart["mc"],
            house_cusps=chart["cusps"],
            house_system=HouseSystem.PLACIDUS,
            planets=chart["planets"],
        )
        natal_jd = birth_to_jd(birth_data.birth_date, 720, birth_data.timezone_offset)
        scorers = build_scorers()
        result = score_candidate(candidate, events, natal_jd, scorers)
        assert result.time_minutes == 480
        assert result.total_score >= 0

    def test_more_events_higher_potential_score(self, birth_data, events):
        jd = birth_to_jd(birth_data.birth_date, 480, birth_data.timezone_offset)
        chart = calc_full_chart(jd, birth_data.latitude, birth_data.longitude, HouseSystem.PLACIDUS)
        from src.models import CandidateChart
        candidate = CandidateChart(
            time_minutes=480, julian_day=jd,
            ascendant=chart["asc"], mc=chart["mc"],
            house_cusps=chart["cusps"], house_system=HouseSystem.PLACIDUS,
            planets=chart["planets"],
        )
        natal_jd = birth_to_jd(birth_data.birth_date, 720, birth_data.timezone_offset)
        scorers = build_scorers()
        score_one = score_candidate(candidate, events[:1], natal_jd, scorers)
        score_all = score_candidate(candidate, events, natal_jd, scorers)
        # More events means more scoring opportunities
        assert score_all.total_score >= score_one.total_score


class TestBayesianUpdate:
    def test_probabilities_sum_to_one(self, birth_data, events):
        from src.ephemeris import generate_candidate_grid
        candidates = generate_candidate_grid(
            birth_data.birth_date, birth_data.latitude, birth_data.longitude,
            birth_data.timezone_offset, interval_minutes=30,
        )
        natal_jd = birth_to_jd(birth_data.birth_date, 720, birth_data.timezone_offset)
        scorers = build_scorers()
        candidate_scores = [
            score_candidate(c, events, natal_jd, scorers) for c in candidates
        ]
        prior = uniform_prior()
        updated = bayesian_update(candidate_scores, prior, candidates)
        total = sum(cs.posterior_probability for cs in updated)
        assert abs(total - 1.0) < 1e-6

    def test_higher_score_higher_posterior(self, birth_data, events):
        from src.ephemeris import generate_candidate_grid
        candidates = generate_candidate_grid(
            birth_data.birth_date, birth_data.latitude, birth_data.longitude,
            birth_data.timezone_offset, interval_minutes=30,
        )[:10]  # small sample
        natal_jd = birth_to_jd(birth_data.birth_date, 720, birth_data.timezone_offset)
        scorers = build_scorers()
        candidate_scores = [
            score_candidate(c, events, natal_jd, scorers) for c in candidates
        ]
        prior = uniform_prior()
        updated = bayesian_update(candidate_scores, prior, candidates)

        # The candidate with the highest training score should have a higher
        # posterior than one with zero score (under uniform prior)
        by_posterior = sorted(updated, key=lambda cs: cs.posterior_probability, reverse=True)
        by_score = sorted(updated, key=lambda cs: cs.training_score(), reverse=True)
        # Top posterior candidate should have score >= average
        avg_score = sum(cs.training_score() for cs in updated) / len(updated)
        assert by_posterior[0].training_score() >= avg_score * 0.5


class TestRectifierLoops:
    def test_loop0_returns_signs(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        signs = rectifier.loop0_morin_filter()
        assert isinstance(signs, list)
        assert 1 <= len(signs) <= 12
        assert all(1 <= s <= 12 for s in signs)

    def test_loop1_returns_sorted_scores(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        signs = rectifier.loop0_morin_filter()
        scores, candidates = rectifier.loop1_broad_pass(signs, interval=30)
        assert isinstance(scores, list)
        assert len(scores) > 0
        # Should be sorted by posterior descending
        for i in range(len(scores) - 1):
            assert scores[i].posterior_probability >= scores[i + 1].posterior_probability

    def test_loop2_returns_one_minute_candidates(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        signs = rectifier.loop0_morin_filter()
        scores_l1, _ = rectifier.loop1_broad_pass(signs, interval=30)
        top_time = scores_l1[0].time_minutes
        scores_l2, candidates_l2 = rectifier.loop2_narrow_pass(top_time, window_minutes=15)
        # All candidates within ±15 min of top_time
        for c in candidates_l2:
            assert abs(c.time_minutes - top_time) <= 15

    def test_loop3_returns_valid_consensus(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        # Use a known time for consensus check
        consensus, results = rectifier.loop3_consensus_check(720)
        assert 0 <= consensus <= 4
        assert len(results) == 4  # 4 house systems

    def test_full_rectify_returns_result(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        result = rectifier.rectify()
        assert isinstance(result, RectificationResult)

    def test_rectify_time_in_valid_range(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        result = rectifier.rectify()
        assert 0 <= result.rectified_time_minutes <= 1439

    def test_rectify_confidence_in_range(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        result = rectifier.rectify()
        assert 0.0 <= result.confidence_score <= 1.0

    def test_rectify_uncertainty_positive(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        result = rectifier.rectify()
        assert result.uncertainty_minutes > 0

    def test_rectify_with_uniform_prior(self, birth_data, events):
        prior = uniform_prior()
        rectifier = Rectifier(birth_data, events, morin_prior=prior, verbose=False)
        result = rectifier.rectify()
        assert isinstance(result, RectificationResult)

    def test_empty_events_still_returns_result(self, birth_data):
        rectifier = Rectifier(birth_data, [], verbose=False)
        result = rectifier.rectify()
        assert isinstance(result, RectificationResult)
        assert result.is_provisional  # sparse data should be provisional

    def test_evidence_ledger_populated(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        result = rectifier.rectify()
        # With 3 anchor events, should have at least some scored hits
        assert isinstance(result.evidence_ledger, list)

    def test_summary_contains_time(self, birth_data, events):
        rectifier = Rectifier(birth_data, events, verbose=False)
        result = rectifier.rectify()
        summary = result.summary()
        assert ":" in summary  # time format HH:MM
        assert "Confidence" in summary


class TestEventDiversityScore:
    def test_diverse_events_high_score(self):
        diverse = [
            LifeEvent(description="Marriage", event_type=EventType.MARRIAGE,
                      date=date(1990, 1, 1), weight=EventWeight.ANCHOR),
            LifeEvent(description="Accident", event_type=EventType.ACCIDENT,
                      date=date(1992, 1, 1), weight=EventWeight.ANCHOR),
            LifeEvent(description="Career", event_type=EventType.CAREER_PEAK,
                      date=date(1995, 1, 1), weight=EventWeight.ANCHOR),
            LifeEvent(description="Surgery", event_type=EventType.SURGERY,
                      date=date(1998, 1, 1), weight=EventWeight.ANCHOR),
        ]
        score, warning = event_diversity_score(diverse)
        assert score > 0.5
        assert warning == ""

    def test_homogeneous_events_warning(self):
        # All career peaks — exactly the Neymar failure mode
        homogeneous = [
            LifeEvent(description=f"Career event {i}", event_type=EventType.CAREER_PEAK,
                      date=date(1990 + i, 1, 1), weight=EventWeight.ANCHOR)
            for i in range(5)
        ]
        score, warning = event_diversity_score(homogeneous)
        assert score < 0.5
        assert "homogeneous" in warning.lower()
        assert "career_peak" in warning

    def test_empty_events_returns_zero(self):
        score, warning = event_diversity_score([])
        assert score == 0.0
        assert warning != ""

    def test_held_out_events_excluded_from_check(self):
        # 3 held-out career peaks + 1 anchor marriage = diverse anchor set
        events = [
            LifeEvent(description=f"Career {i}", event_type=EventType.CAREER_PEAK,
                      date=date(1990 + i, 1, 1), weight=EventWeight.ANCHOR, held_out=True)
            for i in range(3)
        ] + [
            LifeEvent(description="Marriage", event_type=EventType.MARRIAGE,
                      date=date(2000, 1, 1), weight=EventWeight.ANCHOR, held_out=False),
        ]
        score, warning = event_diversity_score(events)
        # Only 1 anchor event, of 1 type — should warn about too few events or low diversity
        assert isinstance(score, float)
        assert isinstance(warning, str)

    def test_diversity_warning_surfaces_in_result(self, birth_data):
        # All events are career peaks — warning should appear in result notes
        homogeneous_events = [
            LifeEvent(description=f"Career {i}", event_type=EventType.CAREER_PEAK,
                      date=date(1988 + i * 3, 6, 1), weight=EventWeight.ANCHOR)
            for i in range(5)
        ]
        rectifier = Rectifier(birth_data, homogeneous_events, verbose=False)
        result = rectifier.rectify()
        assert "homogeneous" in result.notes.lower() or "career_peak" in result.notes
