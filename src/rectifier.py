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
from .scoring.angle_first import AngleFirstScorer, AngleFirstConfig
from .morin_filter import build_morin_prior, uniform_prior


# Minimum events required for high-precision output
MIN_EVENTS_FOR_5MIN = 8
# Minimum posterior ratio for clear mode
CLEAR_MODE_RATIO = 3.0
# Minimum posterior probability for 5-min precision
MIN_POSTERIOR_FOR_PRECISION = 0.40
# Cross-validation: held-out hit rate must be ≥ this fraction of training rate
CV_FLOOR = 0.80
# Cross-validation: below this ratio, result is underdetermined — no specific time emitted
CV_UNDERDETERMINED = 0.20
# Event diversity: if any single type exceeds this fraction, warn of homogeneity bias
MAX_TYPE_FRACTION = 0.60
# Chance-level test: the real birth date competes against N_NULL_NATIVES other
# birth dates (same place, shifted by NULL_SHIFT_YEARS) on the same event list.
# The result must rank in the top CHANCE_ALPHA of that field.
N_NULL_NATIVES = 30
NULL_SHIFT_YEARS = (1.0, 15.0)
CHANCE_ALPHA = 0.05


# Scoring method. "angle_first" follows Tebbs (angles only, solar arc first, 1° orb);
# "classic" is the original five-technique sum, kept for comparison.
DEFAULT_METHOD = "angle_first"
# Solar arcs lead. Transits and progressions count only on top of a solar arc
# contact for the same event, at half weight ("confirm" mode). Measured on synthetic
# charts this was level with solar arc alone and better than letting them score
# independently (see README).
ANGLE_FIRST_CONFIG = AngleFirstConfig(mode="confirm", transit_weight=0.5, progression_weight=0.5)
# The chance-level test compares natives with each other, and transits are the same
# sky for every native: they lift everyone's best score and blur the comparison.
# With solar arc alone the real chart beat all controls in 11 of 16 planted cases,
# against 8 of 16 with confirmations included.
ANGLE_FIRST_GATE_CONFIG = AngleFirstConfig(mode="sa_only")
# Loop 1 grid per method. Two minutes of clock time is about 0.5° of MC, half the
# 1° orb of the angle-first scorer, so a true time between grid points keeps at least
# three quarters of its score. (At 4 minutes the chance-level test found only 6 of 20
# planted charts; the classic 15 minutes is 3.75°, wider than its own 2° orb.)
LOOP1_INTERVAL = {"angle_first": 2, "classic": 15}
# Softmax temperature per method. Angle-first scores are small (at most 2 per
# ordinary event), so the temperature is set where the largest Morin prior ratio
# (about 9 to 1) is worth roughly one exact contact: the prior can break ties but
# cannot overrule the events.
TEMPERATURE = {"angle_first": 0.5, "classic": 5.0}


def build_scorers(tight: bool = False, method: str | None = None) -> list:
    """Return the scorer instances for a method (default: DEFAULT_METHOD)."""
    method = method or DEFAULT_METHOD
    if method == "angle_first":
        return [AngleFirstScorer(tight=tight, config=ANGLE_FIRST_CONFIG)]
    if method != "classic":
        raise ValueError(f"unknown scoring method: {method}")
    return [
        PrimaryDirectionScorer(tight=tight),   # weight 2.0 — most independent
        ProgressionScorer(tight=tight),         # weight 1.5 — progressed Moon
        SolarArcScorer(tight=tight),            # weight 1.2 — correlated with above
        ProfectionScorer(),                     # weight 1.0 — different time scale
        TransitScorer(tight=tight),             # weight 1.0 — real-time positions
    ]


def cluster_events(
    events: list[LifeEvent], window_days: int = 365
) -> list[tuple[LifeEvent, float]]:
    """
    Group temporally-clustered events and apply a correlation penalty.

    Events within `window_days` of each other are partially correlated —
    they reflect the same underlying directional arc rather than independent
    witnesses. The first event in each cluster scores at full weight; each
    subsequent event in the same cluster scores at a reduced multiplier.

    Returns list of (event, cluster_multiplier) pairs.

    Calibration note (Britney Spears test): without clustering, the 2007-2008
    breakdown → custody → conservatorship sequence (3 events in 12 months)
    was triple-counted, creating a strong false attractor at Scorpio 04:47
    even though the confirmed birth time is Libra 01:30.
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.date)
    result: list[tuple[LifeEvent, float]] = []
    last_date: date_type | None = None

    for event in sorted_events:
        if last_date is None or (event.date - last_date).days > window_days:
            # First in a new cluster — full weight
            result.append((event, 1.0))
        else:
            # Within the rolling window of the previous event — half weight
            result.append((event, 0.5))
        last_date = event.date  # rolling: always advance to the most recent event

    return result


def score_candidate(
    candidate: CandidateChart,
    events: list[LifeEvent],
    natal_jd: float,
    scorers: list,
) -> CandidateScore:
    """Apply all scorers to one candidate chart over all events, with temporal clustering."""
    clustered = cluster_events(events)
    all_scores: list[TechniqueScore] = []
    for scorer in scorers:
        for event, cluster_mult in clustered:
            hits = scorer.score_event(candidate, event, natal_jd)
            # Apply cluster penalty to each hit's score
            for hit in hits:
                if cluster_mult < 1.0:
                    hit = hit.model_copy(update={"score": hit.score * cluster_mult})
                all_scores.append(hit)
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
    temperature: float = 5.0,
) -> list[CandidateScore]:
    """
    Update posterior probabilities using the Morin prior and likelihood scores.

    P(time | events) ∝ P(time) × likelihood(time)
    where likelihood is derived from the normalized total scores.

    temperature > 1.0 dampens the exponential sensitivity of the softmax so that
    a few coincidental tight-orb hits don't produce a near-certainty posterior.
    At temperature=1.0, one extra hit (score ≈ 4 pts) multiplies the likelihood
    by e^4 ≈ 55×. At temperature=5.0, the same hit multiplies by e^0.8 ≈ 2.2×.
    """
    # Build prior vector aligned with candidate list
    prior_vec = np.array([
        prior.probabilities.get(c.rising_sign(), 1 / 12)
        for c in candidates
    ])

    # Likelihood: temperature-scaled softmax of training scores
    training_scores = np.array([cs.training_score() for cs in candidate_scores])
    training_scores = np.clip(training_scores, 0, None)

    exp_scores = np.exp((training_scores - training_scores.max()) / temperature)
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


def chance_level_test(
    birth_data: BirthData,
    events: list[LifeEvent],
    house_system: HouseSystem = HouseSystem.PLACIDUS,
    interval_minutes: int | None = None,
    n_natives: int = N_NULL_NATIVES,
    seed: int = 0,
    method: str | None = None,
) -> tuple[float, float, list[float]]:
    """
    Does this event list fit this birth date better than it fits unrelated people?

    The statistic is the best candidate score over the whole day (every rising
    sign, no Morin filter, so the comparison is like for like). It is computed for
    the real birth date and for `n_natives` other birth dates at the same place,
    each shifted by a random 1–15 years either way. If the events carry no timing
    information about this native, the real birth date is just one more member of
    that field, so its rank works as a p-value. The controls are only approximately
    exchangeable with the real native (other eras have other outer-planet
    positions), so calibration is checked empirically: on 70 random event lists,
    3 reached p ≤ 0.05 (4%). With 30 controls the smallest possible p is 1/31, so
    passing the 0.05 gate means beating every control.

    Why not shift the event dates instead? Those controls all orbit the real
    dates, so the real set is not exchangeable with them; in a benchmark on
    random event lists that design flagged 35–40% of cases at p ≤ 0.05.

    Returns (p_value, real_best_score, null_best_scores).
    """
    from datetime import timedelta

    anchors = [e for e in events if not e.held_out]
    if not anchors:
        return 1.0, 0.0, []

    method = method or DEFAULT_METHOD
    if interval_minutes is None:
        interval_minutes = LOOP1_INTERVAL[method]
    if method == "angle_first":
        scorers = [AngleFirstScorer(config=ANGLE_FIRST_GATE_CONFIG)]
    else:
        scorers = build_scorers(tight=False, method=method)

    def best_score(bd) -> float:
        evs = [e for e in anchors if e.date > bd]
        if not evs:
            return 0.0
        jd = birth_to_jd(bd, 720, birth_data.timezone_offset)
        cands = generate_candidate_grid(
            bd, birth_data.latitude, birth_data.longitude,
            birth_data.timezone_offset, interval_minutes=interval_minutes,
            house_system=house_system,
        )
        return max(score_candidate(c, evs, jd, scorers).training_score() for c in cands)

    rng = np.random.default_rng(seed)
    lo, hi = NULL_SHIFT_YEARS
    real = best_score(birth_data.birth_date)
    first_event = min(e.date for e in anchors)
    null = []
    for _ in range(n_natives):
        days = int(rng.choice([-1, 1]) * rng.uniform(lo * 365.25, hi * 365.25))
        # A later-born control must still be alive for every event; otherwise it
        # is scored on fewer events than the real native and the comparison is
        # tilted in the real native's favour. Flip such shifts to an earlier birth.
        if birth_data.birth_date + timedelta(days=days) >= first_event - timedelta(days=365):
            days = -abs(days)
        null.append(best_score(birth_data.birth_date + timedelta(days=days)))
    p = (1 + sum(x >= real for x in null)) / (n_natives + 1)
    return float(p), float(real), null


def best_unfiltered_candidate(
    birth_data: BirthData,
    events: list[LifeEvent],
    house_system: HouseSystem = HouseSystem.PLACIDUS,
    interval_minutes: int | None = None,
    method: str | None = None,
) -> CandidateChart | None:
    """Best-scoring candidate over the whole day, ignoring the Morin filter."""
    anchors = [e for e in events if not e.held_out]
    if not anchors:
        return None
    method = method or DEFAULT_METHOD
    if interval_minutes is None:
        interval_minutes = LOOP1_INTERVAL[method]
    jd = birth_to_jd(birth_data.birth_date, 720, birth_data.timezone_offset)
    cands = generate_candidate_grid(
        birth_data.birth_date, birth_data.latitude, birth_data.longitude,
        birth_data.timezone_offset, interval_minutes=interval_minutes,
        house_system=house_system,
    )
    scorers = build_scorers(tight=False, method=method)
    return max(cands, key=lambda c: score_candidate(c, anchors, jd, scorers).training_score())


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
        method: str | None = None,
    ):
        self.birth_data = birth_data
        self.events = events
        self.verbose = verbose
        self.method = method or DEFAULT_METHOD
        if self.method not in LOOP1_INTERVAL:
            raise ValueError(f"unknown scoring method: {self.method}")

        # Compute natal Julian Day at noon (time will be varied over candidates)
        self.natal_jd = birth_to_jd(
            birth_data.birth_date, 720, birth_data.timezone_offset
        )

        self.chance_p: float | None = None

        self.prior = morin_prior or build_morin_prior(
            birth_data.biography, birth_data.physical_description
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Rectifier] {msg}")

    # ------------------------------------------------------------------
    # Loop 0: Morin pre-filter
    # ------------------------------------------------------------------
    def morin_exclusion_warning(self, rising_signs: list[int]) -> str:
        """
        Warn when the best-scoring time of the whole day has a rising sign that the
        Morin filter removed from the search. The filter is a hard cut: a sign below
        the threshold is never scored, so a wrong biography reading cannot be
        corrected by the event evidence.
        """
        best = best_unfiltered_candidate(self.birth_data, self.events, method=self.method)
        if best is None:
            return ""
        sign = int(best.ascendant // 30) + 1
        if sign in rising_signs:
            return ""
        h, m = divmod(best.time_minutes, 60)
        return (
            f"MORIN FILTER WARNING: the best-scoring time over the whole day is "
            f"{h:02d}:{m:02d} (rising sign {sign}), but the Morin filter excluded that "
            f"rising sign, so it was never considered. Re-run with a uniform prior to compare."
        )

    def loop0_morin_filter(self) -> list[int]:
        """
        Return the rising signs to search.

        classic: signs above the 5% prior threshold (a hard cut).
        angle_first: all twelve. Judging the Ascendant from appearance or character
        is one of the less reliable rectification methods (Tebbs, ch. 1), so the
        Morin prior only weights candidates (see TEMPERATURE) and never removes them.
        """
        if self.method == "angle_first":
            self._log("Loop 0 — Morin prior used as a weight only; all 12 rising signs searched")
            return list(range(1, 13))
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
        interval: int | None = None,
        run_chance_test: bool = False,
    ) -> tuple[list[CandidateScore], list[CandidateChart]]:
        """
        Generate the broad candidate grid (LOOP1_INTERVAL), score all events, Bayesian update.
        With run_chance_test, also sets self.chance_p (see chance_level_test).
        Returns (ranked candidate scores, candidate charts).
        """
        if interval is None:
            interval = LOOP1_INTERVAL[self.method]
        scorers = build_scorers(tight=False, method=self.method)
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

        candidate_scores = bayesian_update(
            candidate_scores, self.prior, candidates, temperature=TEMPERATURE[self.method]
        )
        candidate_scores.sort(key=lambda cs: cs.posterior_probability, reverse=True)

        if run_chance_test and candidates:
            self.chance_p, real_best, _ = chance_level_test(
                self.birth_data, self.events, house_system=house_system,
                interval_minutes=interval, method=self.method,
            )
            self._log(
                f"  Chance-level test: p = {self.chance_p:.3f} "
                f"(best score {real_best:.1f} vs {N_NULL_NATIVES} other natives)"
            )

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
        scorers = build_scorers(tight=True, method=self.method)

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
                    planet_latitudes=chart["planet_latitudes"],
                )
            )

        self._log(f"Loop 2 — {len(candidates)} candidates at 1-min resolution")

        candidate_scores = [
            score_candidate(c, self.events, self.natal_jd, scorers)
            for c in candidates
        ]

        candidate_scores = bayesian_update(
            candidate_scores, self.prior, candidates, temperature=TEMPERATURE[self.method]
        )
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

        The angle-first method scores only ASC and MC, which are the same in every
        house system, so the check cannot disagree and is skipped (reported as 4).
        """
        if self.method == "angle_first":
            return 4, {}
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
            scores_l1, candidates_l1 = self.loop1_broad_pass(
                rising_signs, run_chance_test=(attempt == 0)
            )

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
            scorers_tight = build_scorers(tight=True, method=self.method)
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

        # Underdetermined: cross-validation is too poor to trust the specific time at all
        if cv_score < CV_UNDERDETERMINED and len([e for e in self.events if e.held_out]) >= 1:
            return self._underdetermined_result(
                top_candidate_time=top2.time_minutes,
                cv_score=cv_score,
                consensus=consensus,
            )

        morin_warning = self.morin_exclusion_warning(rising_signs)
        if morin_warning:
            diversity_warning = (diversity_warning + " " + morin_warning).strip()

        chance_p = self.chance_p if self.chance_p is not None else 1.0
        if chance_p > CHANCE_ALPHA:
            return self._chance_level_result(
                top_candidate_time=top2.time_minutes,
                chance_p=chance_p,
                consensus=consensus,
                ranked=scores_l2[:10],
                diversity_warning=diversity_warning,
            )

        # Determine if result is provisional
        is_provisional = (
            top2.posterior_probability < MIN_POSTERIOR_FOR_PRECISION
            or consensus < 2
            or cv_score < CV_FLOOR
            or n_anchor < 4
        )

        # House consensus is only a necessary condition: the angles (ASC/MC), which
        # drive primary directions and solar arcs, are identical in every house
        # system, so agreement is close to automatic. It can lower confidence but is
        # never treated as independent evidence. The chance-level p-value scales it.
        confidence = min(
            top2.posterior_probability * (consensus / 4) * min(cv_score, 1.0) * 2
            * (1.0 - chance_p),
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
            chance_p_value=round(chance_p, 4),
        )

        self._log(f"\nResult: {result.summary()}")
        return result

    def _underdetermined_result(
        self, top_candidate_time: int, cv_score: float, consensus: int
    ) -> RectificationResult:
        """
        Emitted when cross-validation is too poor to trust the winning candidate.
        Returns a 720-minute (12-hour) uncertainty window rather than a specific time.
        Honest behavior: better to report uncertainty than to output a wrong time.
        """
        h, m = divmod(top_candidate_time, 60)
        return RectificationResult(
            rectified_time_minutes=top_candidate_time,
            uncertainty_minutes=720,
            confidence_score=0.0,
            house_system_consensus=consensus,
            is_provisional=True,
            evidence_ledger=[],
            chance_p_value=round(self.chance_p, 4) if self.chance_p is not None else None,
            notes=(
                f"UNDERDETERMINED: cross-validation ratio {cv_score:.2f} is below the "
                f"minimum threshold {CV_UNDERDETERMINED}. The scoring engine found a "
                f"candidate at {h:02d}:{m:02d} but held-out events do not confirm it. "
                "The result is statistically unreliable. Add more diverse anchor events "
                "or verify the held-out event dates."
            ),
        )

    def _chance_level_result(
        self, top_candidate_time: int, chance_p: float, consensus: int,
        ranked: list[CandidateScore], diversity_warning: str,
    ) -> RectificationResult:
        """
        Emitted when the event list fits this native's best candidate no better than
        it fits the best candidates of unrelated natives (see chance_level_test).
        The top-scoring time is kept for inspection but carries no confidence.
        """
        h, m = divmod(top_candidate_time, 60)
        notes = (
            f"UNDERDETERMINED (chance level): the best candidate ({h:02d}:{m:02d}) does not "
            f"fit these events better than the charts of unrelated people born at the "
            f"same place do (p = {chance_p:.3f}, needs ≤ {CHANCE_ALPHA}). Any time reported here "
            "could be produced by chance. Add more hard-dated anchor events of different "
            "types (aim for 8–15), or re-check the birth date, place and time zone."
        )
        if consensus == 4:
            notes += (
                " House-system consensus is 4/4, but this is expected even at chance level "
                "because the angles are the same in every house system."
            )
        if diversity_warning:
            notes += " " + diversity_warning
        return RectificationResult(
            rectified_time_minutes=top_candidate_time,
            uncertainty_minutes=720,
            confidence_score=0.0,
            house_system_consensus=consensus,
            is_provisional=True,
            evidence_ledger=[],
            ranked_candidates=ranked,
            notes=notes,
            chance_p_value=round(chance_p, 4),
        )

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
