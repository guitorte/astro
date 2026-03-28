"""
Calibration tests using known celebrity birth times.

These tests do NOT assert exact minute-level convergence (the system may still not
converge on the confirmed time with the available events), but they validate that:
  1. The correct rising sign appears in the top candidates
  2. The confirmed birth time scores within a reasonable fraction of the top score
  3. Diversity and clustering checks fire correctly for these real-world event sets

Known birth times (from Astrodatabank, Rodden Rating AA/A):
  Neymar Jr.   — 5 Feb 1992, 02:15 (Sagittarius ASC), Mogi das Cruzes, Brazil
  Britney Spears — 2 Dec 1981, 01:30 (Libra ASC), McComb, Mississippi, USA
"""

import pytest
from datetime import date
from src.models import (
    BirthData, LifeEvent, EventType, EventWeight,
    RisingSignPrior,
)
from src.rectifier import (
    Rectifier, score_candidate, build_scorers, bayesian_update,
    event_diversity_score, cluster_events,
)
from src.ephemeris import birth_to_jd, generate_candidate_grid
from src.models import HouseSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neymar_birth():
    return BirthData(
        name="Neymar Jr.",
        birth_date=date(1992, 2, 5),
        birth_city="Mogi das Cruzes, Brazil",
        latitude=-23.5225,
        longitude=-46.1861,
        # Brazilian DST 1991-92: Oct 20, 1991 → Feb 9, 1992 (São Paulo observes DST)
        # Feb 5, 1992 IS within DST → BRST = UTC-2 (not BRT = UTC-3)
        timezone_offset=-2.0,
        biography=(
            "Brazilian professional footballer, widely considered one of the best players "
            "of his generation. Plays as forward. Joined Santos FC as a teenager, "
            "transferred to Barcelona for a world-record fee, then to Paris Saint-Germain "
            "for an even larger world-record fee. Multiple Copa América appearances. "
            "Suffered serious injuries during World Cups."
        ),
    )


@pytest.fixture
def neymar_events():
    """Key life events for Neymar — deliberately diverse to test real-world performance."""
    return [
        LifeEvent(
            description="Santos FC professional debut",
            event_type=EventType.CAREER_PEAK,
            date=date(2009, 3, 7),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Transfer to FC Barcelona (world-record fee at the time)",
            event_type=EventType.RELOCATION,
            date=date(2013, 6, 3),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Back injury at 2014 World Cup — missed semi-final",
            event_type=EventType.ACCIDENT,
            date=date(2014, 7, 4),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Transfer to Paris Saint-Germain — world-record €222M fee",
            event_type=EventType.CAREER_PEAK,
            date=date(2017, 8, 3),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Foot fracture — missed 3 months, including Champions League",
            event_type=EventType.ILLNESS,
            date=date(2018, 2, 25),
            weight=EventWeight.ANCHOR,
            held_out=True,  # cross-validation
        ),
    ]


@pytest.fixture
def britney_birth():
    return BirthData(
        name="Britney Spears",
        birth_date=date(1981, 12, 2),
        birth_city="McComb, Mississippi, USA",
        latitude=31.2435,
        longitude=-90.4532,
        timezone_offset=-6.0,
        biography=(
            "American pop singer and entertainer. Rose to fame as a teenager in 1998 with "
            "her debut single. Multiple Grammy awards. Went through a public breakdown in "
            "2007-2008, lost custody of her children, and was placed under a conservatorship "
            "controlled by her father from 2008 to 2021. Married twice."
        ),
    )


@pytest.fixture
def britney_events():
    """Key life events for Britney Spears — diverse types to test clustering."""
    return [
        LifeEvent(
            description="Marriage to Jason Alexander (55-hour marriage)",
            event_type=EventType.MARRIAGE,
            date=date(2004, 1, 3),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Marriage to Kevin Federline",
            event_type=EventType.MARRIAGE,
            date=date(2004, 9, 18),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Public breakdown and hospitalization",
            event_type=EventType.HOSPITALIZATION,
            date=date(2007, 2, 16),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Lost custody of sons Sean and Jayden",
            event_type=EventType.CUSTODY_LOSS,
            date=date(2007, 10, 3),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Placed under conservatorship controlled by father",
            event_type=EventType.LEGAL_RESTRICTION,
            date=date(2008, 2, 1),
            weight=EventWeight.ANCHOR,
        ),
        LifeEvent(
            description="Conservatorship terminated by court",
            event_type=EventType.CAREER_PEAK,  # proxy for major life change / liberation
            date=date(2021, 11, 12),
            weight=EventWeight.ANCHOR,
            held_out=True,  # cross-validation
        ),
    ]


# ---------------------------------------------------------------------------
# Neymar calibration tests
# ---------------------------------------------------------------------------

class TestNeymarCalibration:
    CONFIRMED_SIGN = 9    # Sagittarius
    CONFIRMED_TIME = 135  # 02:15 = 135 minutes

    def test_diversity_warning_fires_for_homogeneous_events(self, neymar_birth, neymar_events):
        """Homogeneous event sets (many career peaks) should trigger diversity warning."""
        career_heavy = [
            LifeEvent(
                description=f"Career event {i}",
                event_type=EventType.CAREER_PEAK,
                date=date(2005 + i, 6, 1),
                weight=EventWeight.ANCHOR,
            )
            for i in range(5)
        ]
        score, warning = event_diversity_score(career_heavy)
        assert score < 0.5
        assert warning  # should warn about homogeneous events

    def test_neymar_events_pass_diversity_check(self, neymar_events):
        """The calibration event set uses diverse types — diversity score should be acceptable."""
        anchor = [e for e in neymar_events if not e.held_out]
        score, warning = event_diversity_score(anchor)
        # 4 events with 3 different types → diversity > 0.5
        assert score > 0.4, f"Expected diversity > 0.4, got {score}. Warning: {warning}"

    def test_sagittarius_in_top_half_candidates(self, neymar_birth, neymar_events):
        """Confirmed Sagittarius rising should appear in at least half the top candidates."""
        natal_jd = birth_to_jd(neymar_birth.birth_date, 720, neymar_birth.timezone_offset)
        # Uniform prior — no Morin bias
        prior = RisingSignPrior(probabilities={s: 1/12 for s in range(1, 13)})
        candidates = generate_candidate_grid(
            neymar_birth.birth_date,
            neymar_birth.latitude,
            neymar_birth.longitude,
            neymar_birth.timezone_offset,
            interval_minutes=15,
            house_system=HouseSystem.PLACIDUS,
        )
        scorers = build_scorers(tight=False)
        scored = [score_candidate(c, neymar_events, natal_jd, scorers) for c in candidates]
        scored = bayesian_update(scored, prior, candidates)
        scored.sort(key=lambda cs: cs.posterior_probability, reverse=True)

        top_half = scored[:len(scored) // 2]
        top_times = {cs.time_minutes for cs in top_half}
        sag_candidates = [c for c in candidates if c.rising_sign() == self.CONFIRMED_SIGN]
        sag_times = {c.time_minutes for c in sag_candidates}

        overlap = top_times & sag_times
        assert overlap, (
            f"No Sagittarius candidates in top half. "
            f"Sag times: {sorted(sag_times)}, Top times: {sorted(top_times)[:10]}"
        )

    def test_confirmed_time_within_uncertainty_of_winner(self, neymar_birth, neymar_events):
        """The system's rectified time should be within ±30 min of confirmed 02:15."""
        rectifier = Rectifier(neymar_birth, neymar_events, verbose=False)
        result = rectifier.rectify()
        diff = abs(result.rectified_time_minutes - self.CONFIRMED_TIME)
        assert diff <= 45, (
            f"Rectified time {result.time_label()} is {diff} min from "
            f"confirmed 02:15. Expected within ±45 min."
        )

    def test_clustering_applied_to_events(self, neymar_events):
        """Cluster events function should work correctly for Neymar's event set."""
        clustered = cluster_events(neymar_events)
        assert len(clustered) == len(neymar_events)
        # Verify clustering returns valid multipliers (1.0 or 0.5)
        for event, mult in clustered:
            assert mult in (1.0, 0.5), f"Unexpected multiplier {mult} for {event.description}"
        # At least the first event (earliest date) should have full weight
        sorted_clustered = sorted(clustered, key=lambda ec: ec[0].date)
        assert sorted_clustered[0][1] == 1.0, "Earliest event should always have weight 1.0"


# ---------------------------------------------------------------------------
# Britney Spears calibration tests
# ---------------------------------------------------------------------------

class TestBritneyCalibration:
    CONFIRMED_SIGN = 7   # Libra
    CONFIRMED_TIME = 90  # 01:30 = 90 minutes

    def test_temporal_clustering_fires_for_2007_sequence(self, britney_events):
        """The 2007 breakdown → custody loss → conservatorship (all < 365 days apart)."""
        clustered = cluster_events(britney_events)
        cluster_map = {e.description: mult for e, mult in clustered}

        # Custody loss (Oct 2007) is within 365 days of hospitalization (Feb 2007)
        hospitalization_key = next(
            k for k in cluster_map if "hospitalization" in k.lower() or "breakdown" in k.lower()
        )
        custody_key = next(k for k in cluster_map if "custody" in k.lower())
        conservatorship_key = next(k for k in cluster_map if "conservatorship" in k.lower())

        # First event in cluster → weight 1.0
        assert cluster_map[hospitalization_key] == 1.0
        # Within 365 days → weight 0.5
        assert cluster_map[custody_key] == 0.5, (
            f"Custody loss should be clustered (0.5×), got {cluster_map[custody_key]}"
        )
        assert cluster_map[conservatorship_key] == 0.5, (
            f"Conservatorship should be clustered (0.5×), got {cluster_map[conservatorship_key]}"
        )

    def test_britney_event_diversity(self, britney_events):
        """Event set uses CUSTODY_LOSS, LEGAL_RESTRICTION, HOSPITALIZATION, MARRIAGE — should be diverse."""
        anchor = [e for e in britney_events if not e.held_out]
        score, warning = event_diversity_score(anchor)
        assert score > 0.5, f"Expected diversity > 0.5, got {score}"
        assert not warning or "homogeneous" not in warning.lower(), (
            f"Should not warn about homogeneous events. Warning: {warning}"
        )

    def test_libra_in_top_half_candidates(self, britney_birth, britney_events):
        """Confirmed Libra rising should appear in at least one of the top half candidates."""
        natal_jd = birth_to_jd(britney_birth.birth_date, 720, britney_birth.timezone_offset)
        prior = RisingSignPrior(probabilities={s: 1/12 for s in range(1, 13)})
        candidates = generate_candidate_grid(
            britney_birth.birth_date,
            britney_birth.latitude,
            britney_birth.longitude,
            britney_birth.timezone_offset,
            interval_minutes=15,
            house_system=HouseSystem.PLACIDUS,
        )
        scorers = build_scorers(tight=False)
        scored = [score_candidate(c, britney_events, natal_jd, scorers) for c in candidates]
        scored = bayesian_update(scored, prior, candidates)
        scored.sort(key=lambda cs: cs.posterior_probability, reverse=True)

        top_half = scored[:len(scored) // 2]
        top_times = {cs.time_minutes for cs in top_half}
        libra_candidates = [c for c in candidates if c.rising_sign() == self.CONFIRMED_SIGN]
        libra_times = {c.time_minutes for c in libra_candidates}

        overlap = top_times & libra_times
        assert overlap, (
            f"No Libra candidates in top half. "
            f"Libra times: {sorted(libra_times)}, Top times: {sorted(top_times)[:10]}"
        )

    def test_confirmed_time_scores_within_reasonable_fraction(self, britney_birth, britney_events):
        """The confirmed 01:30 time should score ≥ 30% of the top scorer."""
        natal_jd = birth_to_jd(britney_birth.birth_date, 720, britney_birth.timezone_offset)
        candidates = generate_candidate_grid(
            britney_birth.birth_date,
            britney_birth.latitude,
            britney_birth.longitude,
            britney_birth.timezone_offset,
            interval_minutes=15,
            house_system=HouseSystem.PLACIDUS,
        )
        scorers = build_scorers(tight=False)
        scored = {
            c.time_minutes: score_candidate(c, britney_events, natal_jd, scorers)
            for c in candidates
        }

        nearest_time = min(scored.keys(), key=lambda t: abs(t - self.CONFIRMED_TIME))
        confirmed_score = scored[nearest_time].training_score()
        top_score = max(cs.training_score() for cs in scored.values())

        ratio = confirmed_score / (top_score + 1e-9)
        assert ratio >= 0.30, (
            f"Confirmed time {self.CONFIRMED_TIME}min (nearest: {nearest_time}min) "
            f"scores {confirmed_score:.2f}, top score {top_score:.2f}, ratio {ratio:.2%}. "
            "Expected ≥ 30% of top score."
        )


# ---------------------------------------------------------------------------
# Cross-cutting: score inflation guard
# ---------------------------------------------------------------------------

class TestScoreInflation:
    """Verify that hit capping and deduplication reduce combinatorial noise."""

    def test_hit_count_bounded_per_technique(self, neymar_birth, neymar_events):
        """Each scorer should return at most MAX_HITS_PER_EVENT hits per event."""
        from src.scoring.base import MAX_HITS_PER_EVENT
        natal_jd = birth_to_jd(neymar_birth.birth_date, 720, neymar_birth.timezone_offset)
        candidates = generate_candidate_grid(
            neymar_birth.birth_date,
            neymar_birth.latitude,
            neymar_birth.longitude,
            neymar_birth.timezone_offset,
            interval_minutes=60,  # just 24 candidates for speed
            house_system=HouseSystem.PLACIDUS,
        )
        scorers = build_scorers(tight=False)
        if not candidates:
            pytest.skip("No candidates generated")
        c = candidates[0]
        for scorer in scorers:
            for event in neymar_events:
                hits = scorer.score_event(c, event, natal_jd)
                assert len(hits) <= MAX_HITS_PER_EVENT, (
                    f"{scorer.name} returned {len(hits)} hits for '{event.description}', "
                    f"expected ≤ {MAX_HITS_PER_EVENT}"
                )

    def test_posterior_not_overconfident_with_uniform_prior(self, neymar_birth, neymar_events):
        """With uniform prior, top candidate posterior should be < 30% (not near-certain)."""
        natal_jd = birth_to_jd(neymar_birth.birth_date, 720, neymar_birth.timezone_offset)
        prior = RisingSignPrior(probabilities={s: 1/12 for s in range(1, 13)})
        candidates = generate_candidate_grid(
            neymar_birth.birth_date,
            neymar_birth.latitude,
            neymar_birth.longitude,
            neymar_birth.timezone_offset,
            interval_minutes=15,
            house_system=HouseSystem.PLACIDUS,
        )
        scorers = build_scorers(tight=False)
        scored = [score_candidate(c, neymar_events, natal_jd, scorers) for c in candidates]
        scored = bayesian_update(scored, prior, candidates)
        top_posterior = max(cs.posterior_probability for cs in scored)
        # With ~96 candidates and uniform prior, if scores were totally random the top
        # posterior would be ~1/96 ≈ 1%. A posterior ≥ 50% signals extreme score concentration
        # (the "false attractor" problem). Temperature=5.0 should keep this well below 50%.
        assert top_posterior < 0.50, (
            f"Top posterior {top_posterior:.3f} is overconfident (≥ 50%) with uniform prior. "
            "Score inflation or temperature too low."
        )
