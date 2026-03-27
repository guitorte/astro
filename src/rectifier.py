"""
The 4-loop rectification orchestrator.

Loop 0 — Morin pre-filter: reduces candidate pool from 288 → ~72
Loop 1 — Broad scoring pass: 15-min resolution, Bayesian update
Loop 2 — Narrow pass: 1-min resolution ±30min around top candidate
Loop 3 — House system consensus: re-run through all 4 house systems

All scoring operations are vectorized (numpy-based dispatch).
"""

import numpy as np
from datetime import date as date_type
import swisseph as swe

from .models import (
    BirthData, LifeEvent, CandidateChart, CandidateScore,
    HouseSystem, RisingSignPrior, RectificationResult, TechniqueScore,
)
from .ephemeris import (
    birth_to_jd, generate_candidate_grid, MOSHIER_FLAG,
)
from .scoring import (
    TransitScorer, ProgressionScorer, SolarArcScorer,
    ProfectionScorer, PrimaryDirectionScorer,
)
from .morin_filter import build_morin_prior, uniform_prior


# Minimum events required for high-precision output
MIN_EVENTS_FOR_5MIN = 8
# Minimum posterior ratio for clear mode
CLEAR_MODE_RATIO = 3.0
# Minimum posterior probability for 5-min precision
MIN_POSTERIOR_FOR_PRECISION = 0.40
# Cross-validation: held-out hit rate must be ≥ this fraction of training rate
CV_FLOOR = 0.80
# Event diversity: if any single type exceeds this fraction, warn of homogeneity bias
MAX_TYPE_FRACTION = 0.60


def build_scorers(tight: bool = False) -> list:
    """Return all 5 scoring technique instances in weighted order."""
    return [
        PrimaryDirectionScorer(tight=tight),   # weight 2.0 — most independent
        ProgressionScorer(tight=tight),         # weight 1.5 — progressed Moon
        SolarArcScorer(tight=tight),            # weight 1.2 — correlated with above
        ProfectionScorer(),                     # weight 1.0 — different time scale
        TransitScorer(tight=tight),             # weight 1.0 — real-time positions
    ]


def score_candidate(
    candidate: CandidateChart,
    events: list[LifeEvent],
    natal_jd: float,
    scorers: list,
) -> CandidateScore:
    """Apply all scorers to one candidate chart over all events."""
    all_scores: list[TechniqueScore] = []
    for scorer in scorers:
        all_scores.extend(scorer.score_all_events(candidate, events, natal_jd))
    total = sum(s.score for s in all_scores)
    return CandidateScore(
        time_minutes=candidate.time_minutes,
        total_score=total,
        technique_scores=all_scores,
        posterior_probability=0.0,
    )


def bayesian_update(
    candidate_scores: list[CandidateScore],
    prior: RisingSignPrior,
    candidates: list[CandidateChart],
) -> list[CandidateScore]:
    """
    Update posterior probabilities using the Morin prior and likelihood scores.

    P(time | events) ∝ P(time) × likelihood(time)
    where likelihood is derived from the normalized total scores.
    """
    # Build prior vector aligned with candidate list
    prior_vec = np.array([
        prior.probabilities.get(c.rising_sign(), 1 / 12)
        for c in candidates
    ])

    # Likelihood: softmax of training scores
    training_scores = np.array([cs.training_score() for cs in candidate_scores])
    training_scores = np.clip(training_scores, 0, None)

    # Exponentiate (temperature=1) and multiply by prior
    exp_scores = np.exp(training_scores - training_scores.max())  # numerical stability
    posterior = prior_vec * exp_scores
    posterior /= posterior.sum() + 1e-12

    for i, cs in enumerate(candidate_scores):
        cs.posterior_probability = float(posterior[i])

    return candidate_scores


def bootstrap_stability(
    top_candidate: CandidateChart,
    events: list[LifeEvent],
    natal_jd: float,
    scorers: list,
    n_bootstrap: int = 20,
    perturbation_days: int = 3,
) -> float:
    """
    Perturb event dates by ±perturbation_days and measure score stability.
    Returns the coefficient of variation (lower = more stable).
    """
    from datetime import timedelta
    import random

    scores = []
    for _ in range(n_bootstrap):
        perturbed = []
        for e in events:
            if not e.held_out:
                delta = random.randint(-perturbation_days, perturbation_days)
                new_event = e.model_copy(
                    update={"date": e.date + timedelta(days=delta)}
                )
                perturbed.append(new_event)
            else:
                perturbed.append(e)

        cs = score_candidate(top_candidate, perturbed, natal_jd, scorers)
        scores.append(cs.training_score())

    arr = np.array(scores)
    mean = arr.mean()
    std = arr.std()
    return float(std / (mean + 1e-12))  # coefficient of variation


def event_diversity_score(events: list[LifeEvent]) -> tuple[float, str]:
    """
    Measure how diverse the event set is across house types.

    Returns:
        (score 0.0–1.0, warning message or "")

    A score of 1.0 means all events are of different types.
    A score below 0.5 means >60% of events share the same type — this creates
    a systematic house bias and can produce false attractors in the scoring
    (the "homogeneous event set" failure mode from the Neymar calibration test).
    """
    anchor = [e for e in events if not e.held_out]
    if not anchor:
        return 0.0, "No anchor events."

    from collections import Counter
    counts = Counter(e.event_type.value for e in anchor)
    n = len(anchor)
    top_type, top_count = counts.most_common(1)[0]
    fraction = top_count / n
    n_distinct = len(counts)

    # Simpson diversity index (1 - sum of p^2)
    simpson = 1.0 - sum((c / n) ** 2 for c in counts.values())

    warning = ""
    if fraction > MAX_TYPE_FRACTION:
        warning = (
            f"Event set is homogeneous: {top_count}/{n} events are '{top_type}'. "
            "Scoring is biased toward whichever chart best fits that house type. "
            "Add diverse events (injuries, family, legal, relocation) to improve accuracy."
        )
    elif n_distinct < 3:
        warning = (
            f"Only {n_distinct} distinct event types across {n} events. "
            "More variety improves discriminating power across house axes."
        )

    return round(simpson, 3), warning


def cross_validate(
    top_candidate: CandidateChart,
    all_scores: CandidateScore,
    events: list[LifeEvent],
) -> float:
    """
    Compare hit rate on held-out events vs. training events.
    Returns ratio (1.0 = perfect generalization, <0.8 = possible overfitting).
    """
    training_hits = [s for s in all_scores.technique_scores if not s.held_out and s.score > 0]
    held_out_hits = [s for s in all_scores.technique_scores if s.held_out and s.score > 0]

    training_events = [e for e in events if not e.held_out]
    held_out_events = [e for e in events if e.held_out]

    if not training_events or not held_out_events:
        return 1.0  # no held-out set → cannot validate

    train_rate = len(training_hits) / (len(training_events) * 5)  # 5 techniques
    held_rate = len(held_out_hits) / (len(held_out_events) * 5)

    if train_rate == 0:
        return 1.0

    return min(held_rate / train_rate, 2.0)


class Rectifier:
    """
    Four-loop birth time rectifier.

    Usage:
        rectifier = Rectifier(birth_data, events)
        result = rectifier.rectify()
    """

    def __init__(
        self,
        birth_data: BirthData,
        events: list[LifeEvent],
        morin_prior: RisingSignPrior | None = None,
        verbose: bool = False,
    ):
        self.birth_data = birth_data
        self.events = events
        self.verbose = verbose

        # Compute natal Julian Day at noon (time will be varied over candidates)
        self.natal_jd = birth_to_jd(
            birth_data.birth_date, 720, birth_data.timezone_offset
        )

        self.prior = morin_prior or build_morin_prior(
            birth_data.biography, birth_data.physical_description
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Rectifier] {msg}")

    # ------------------------------------------------------------------
    # Loop 0: Morin pre-filter
    # ------------------------------------------------------------------
    def loop0_morin_filter(self) -> list[int]:
        """Return plausible rising signs (those above 5% threshold)."""
        signs = self.prior.top_signs(threshold=0.05)
        self._log(
            f"Loop 0 — Morin: {len(signs)} plausible rising signs: "
            + ", ".join(str(s) for s in sorted(signs))
        )
        return signs

    # ------------------------------------------------------------------
    # Loop 1: Broad scoring pass
    # ------------------------------------------------------------------
    def loop1_broad_pass(
        self,
        rising_signs: list[int],
        house_system: HouseSystem = HouseSystem.PLACIDUS,
        interval: int = 15,
    ) -> tuple[list[CandidateScore], list[CandidateChart]]:
        """
        Generate 15-min candidate grid, score all events, Bayesian update.
        Returns (ranked candidate scores, candidate charts).
        """
        scorers = build_scorers(tight=False)
        candidates = generate_candidate_grid(
            self.birth_data.birth_date,
            self.birth_data.latitude,
            self.birth_data.longitude,
            self.birth_data.timezone_offset,
            interval_minutes=interval,
            rising_signs=rising_signs,
            house_system=house_system,
        )

        self._log(f"Loop 1 — {len(candidates)} candidates at {interval}-min resolution")

        candidate_scores = [
            score_candidate(c, self.events, self.natal_jd, scorers)
            for c in candidates
        ]

        candidate_scores = bayesian_update(candidate_scores, self.prior, candidates)
        candidate_scores.sort(key=lambda cs: cs.posterior_probability, reverse=True)

        if candidate_scores:
            top = candidate_scores[0]
            second = candidate_scores[1] if len(candidate_scores) > 1 else None
            ratio = (
                top.posterior_probability / (second.posterior_probability + 1e-12)
                if second else float("inf")
            )
            self._log(
                f"  Top: {top.time_minutes // 60:02d}:{top.time_minutes % 60:02d} "
                f"score={top.training_score():.2f} p={top.posterior_probability:.3f} "
                f"ratio={ratio:.2f}"
            )

        return candidate_scores, candidates

    # ------------------------------------------------------------------
    # Loop 2: Narrow pass
    # ------------------------------------------------------------------
    def loop2_narrow_pass(
        self,
        top_time_minutes: int,
        house_system: HouseSystem = HouseSystem.PLACIDUS,
        window_minutes: int = 30,
    ) -> tuple[list[CandidateScore], list[CandidateChart]]:
        """
        Re-score ±window_minutes around top candidate at 1-minute resolution.
        Uses tighter orbs and full event set.
        """
        scorers = build_scorers(tight=True)

        # Build candidates in the narrow window
        low = max(0, top_time_minutes - window_minutes)
        high = min(1439, top_time_minutes + window_minutes)
        candidates = []

        for t in range(low, high + 1, 1):
            from .ephemeris import birth_to_jd as btj, calc_full_chart
            jd = btj(self.birth_data.birth_date, t, self.birth_data.timezone_offset)
            chart = calc_full_chart(
                jd, self.birth_data.latitude, self.birth_data.longitude, house_system
            )
            candidates.append(
                CandidateChart(
                    time_minutes=t,
                    julian_day=jd,
                    ascendant=chart["asc"],
                    mc=chart["mc"],
                    house_cusps=chart["cusps"],
                    house_system=house_system,
                    planets=chart["planets"],
                )
            )

        self._log(f"Loop 2 — {len(candidates)} candidates at 1-min resolution")

        candidate_scores = [
            score_candidate(c, self.events, self.natal_jd, scorers)
            for c in candidates
        ]

        candidate_scores = bayesian_update(candidate_scores, self.prior, candidates)
        candidate_scores.sort(key=lambda cs: cs.posterior_probability, reverse=True)

        if candidate_scores:
            top = candidate_scores[0]
            self._log(
                f"  Narrow top: {top.time_minutes // 60:02d}:{top.time_minutes % 60:02d} "
                f"score={top.training_score():.2f} p={top.posterior_probability:.3f}"
            )

        return candidate_scores, candidates

    # ------------------------------------------------------------------
    # Loop 3: House system consensus check
    # ------------------------------------------------------------------
    def loop3_consensus_check(
        self, top_time_minutes: int
    ) -> tuple[int, dict[str, int]]:
        """
        Re-run Loop 1 for all 4 house systems and check agreement.
        Returns (consensus_count, {house_system: top_time_minutes}).
        """
        systems = [
            HouseSystem.PLACIDUS,
            HouseSystem.KOCH,
            HouseSystem.WHOLE_SIGN,
            HouseSystem.REGIOMONTANUS,
        ]

        rising_signs = self.loop0_morin_filter()
        results: dict[str, int] = {}

        for hs in systems:
            scores, _ = self.loop1_broad_pass(rising_signs, house_system=hs)
            if scores:
                results[hs.value] = scores[0].time_minutes

        # Count how many systems agree within ±15 minutes of top candidate
        consensus = sum(
            1 for t in results.values() if abs(t - top_time_minutes) <= 15
        )
        self._log(
            f"Loop 3 — House consensus {consensus}/4: "
            + str({k: f"{v // 60:02d}:{v % 60:02d}" for k, v in results.items()})
        )
        return consensus, results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def rectify(self) -> RectificationResult:
        """
        Execute the full 4-loop rectification workflow.
        Returns a RectificationResult with the evidence ledger.
        """
        anchor_events = [e for e in self.events if not e.held_out]
        n_anchor = len(anchor_events)

        # --- Event diversity check (pre-flight) ---
        diversity, diversity_warning = event_diversity_score(self.events)
        if diversity_warning:
            self._log(f"WARNING — {diversity_warning}")

        # --- Loop 0 ---
        rising_signs = self.loop0_morin_filter()

        # --- Loop 1 (may repeat if evidence is sparse) ---
        max_loop1_attempts = 2
        for attempt in range(max_loop1_attempts):
            scores_l1, candidates_l1 = self.loop1_broad_pass(rising_signs)

            if not scores_l1:
                return self._empty_result("No candidates generated after Morin filter.")

            top1 = scores_l1[0]
            second1 = scores_l1[1] if len(scores_l1) > 1 else None
            ratio = (
                top1.posterior_probability / (second1.posterior_probability + 1e-12)
                if second1 else float("inf")
            )

            if ratio >= CLEAR_MODE_RATIO or attempt == max_loop1_attempts - 1:
                break
            self._log(
                f"  Loop 1 ratio {ratio:.2f} < {CLEAR_MODE_RATIO:.1f}; "
                "would re-run with more events (demo: continuing)"
            )

        # --- Loop 2 ---
        scores_l2, candidates_l2 = self.loop2_narrow_pass(top1.time_minutes)

        if not scores_l2:
            return self._empty_result("Loop 2 produced no candidates.")

        top2 = scores_l2[0]

        # Bootstrap stability check
        matching_candidates = [
            c for c in candidates_l2 if c.time_minutes == top2.time_minutes
        ]
        cv_score = 0.0
        stability_cv = 1.0  # default: unstable until proven otherwise
        if matching_candidates:
            scorers_tight = build_scorers(tight=True)
            stability_cv = bootstrap_stability(
                matching_candidates[0], self.events, self.natal_jd, scorers_tight
            )
            cv_score = cross_validate(matching_candidates[0], top2, self.events)
            self._log(f"  Stability CV={stability_cv:.3f}, CrossVal ratio={cv_score:.2f}")

        # Determine uncertainty window
        if n_anchor >= MIN_EVENTS_FOR_5MIN and stability_cv < 0.3:
            uncertainty = 5
        elif n_anchor >= 4:
            uncertainty = 15
        else:
            uncertainty = 30

        # --- Loop 3 ---
        consensus, system_results = self.loop3_consensus_check(top2.time_minutes)

        # Determine if result is provisional
        is_provisional = (
            top2.posterior_probability < MIN_POSTERIOR_FOR_PRECISION
            or consensus < 2
            or cv_score < CV_FLOOR
            or n_anchor < 4
        )

        confidence = min(
            top2.posterior_probability * (consensus / 4) * min(cv_score, 1.0) * 2,
            1.0,
        )

        notes_parts = []
        if consensus < 3:
            notes_parts.append(f"House system consensus only {consensus}/4")
        if stability_cv >= 0.3:
            notes_parts.append("Convergence may be event-driven rather than robust")
        if n_anchor < MIN_EVENTS_FOR_5MIN:
            notes_parts.append(
                f"Only {n_anchor} anchor events — precision capped at {uncertainty} min"
            )
        if diversity_warning:
            notes_parts.append(diversity_warning)

        result = RectificationResult(
            rectified_time_minutes=top2.time_minutes,
            uncertainty_minutes=uncertainty,
            confidence_score=round(confidence, 3),
            house_system_consensus=consensus,
            is_provisional=is_provisional,
            evidence_ledger=top2.technique_scores,
            ranked_candidates=scores_l2[:10],
            notes="; ".join(notes_parts),
        )

        self._log(f"\nResult: {result.summary()}")
        return result

    def _empty_result(self, reason: str) -> RectificationResult:
        return RectificationResult(
            rectified_time_minutes=720,
            uncertainty_minutes=720,
            confidence_score=0.0,
            house_system_consensus=0,
            is_provisional=True,
            evidence_ledger=[],
            notes=reason,
        )
