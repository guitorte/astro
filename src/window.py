"""
Window comparison: the conservative use of rectification.

When an approximate birth time is known (a certificate, a hospital record, a parent's
memory), the question is not "what minute was it?" but "do the life events prefer
the recorded time or one of its neighbours?". Tebbs ("The Complete Book of Chart
Rectification", chs. 2–4) works this way: charts at ten-minute intervals across the
uncertain range, compared event by event.

compare_window() scores every minute in [approx - window, approx + window], groups
the minutes into slots, and reports for each slot its best minute, the Ascendant
and Midheaven there, and the contacts found. It also reports when the Ascendant or
Midheaven changes sign inside the window, and whether the events discriminate at
all: the best score in the window is ranked against the same statistic for
unrelated birth dates at the same clock times (see window_chance_test).
"""

from datetime import timedelta

import numpy as np
from pydantic import BaseModel, Field

from .ephemeris import birth_to_jd, generate_candidate_grid
from .models import BirthData, LifeEvent, SIGN_NAMES
from .rectifier import (
    ANGLE_FIRST_GATE_CONFIG, CHANCE_ALPHA, DEFAULT_METHOD, N_NULL_NATIVES,
    NULL_SHIFT_YEARS, build_scorers, score_candidate,
)
from .scoring.angle_first import AngleFirstScorer


def _label(minute: int) -> str:
    h, m = divmod(minute % 1440, 60)
    return f"{h:02d}:{m:02d}"


def _position(lon: float) -> str:
    deg = lon % 30
    return f"{int(deg)}°{int((deg % 1) * 60):02d}' {SIGN_NAMES[int(lon // 30) % 12]}"


class Contact(BaseModel):
    event: str
    technique: str
    contact: str
    orb: float
    score: float


class WindowSlot(BaseModel):
    start: str
    end: str
    best_time: str
    best_time_minutes: int
    score: float
    ascendant: str
    midheaven: str
    contains_recorded_time: bool
    contacts: list[Contact] = Field(default_factory=list)


class SignChange(BaseModel):
    time: str
    angle: str          # "ASC" or "MC"
    from_sign: str
    to_sign: str


class WindowReport(BaseModel):
    recorded_time: str
    window_minutes: int
    slot_minutes: int
    anchor_events: int
    recorded_ascendant: str
    recorded_midheaven: str
    recorded_score: float
    best_time: str
    best_score: float
    minutes_from_recorded: int
    chance_p_value: float
    verdict: str        # "no_preference" | "supports_recorded" | "prefers_other"
    summary: str
    sign_changes: list[SignChange] = Field(default_factory=list)
    slots: list[WindowSlot] = Field(default_factory=list)


def _window_candidates(birth_data: BirthData, birth_date, start: int, end: int):
    return generate_candidate_grid(
        birth_date, birth_data.latitude, birth_data.longitude,
        birth_data.timezone_offset, interval_minutes=1,
        start_minute=start, end_minute=end,
    )


def window_chance_test(
    birth_data: BirthData,
    events: list[LifeEvent],
    start: int,
    end: int,
    n_natives: int = N_NULL_NATIVES,
    seed: int = 0,
    method: str | None = None,
) -> tuple[float, float, list[float]]:
    """
    Same idea as rectifier.chance_level_test, restricted to the clock window: the
    best score between `start` and `end` for the real birth date is ranked against
    the best score in the same window for `n_natives` other birth dates at the same
    place. Returns (p_value, real_best, control_bests).
    """
    anchors = [e for e in events if not e.held_out]
    if not anchors:
        return 1.0, 0.0, []
    method = method or DEFAULT_METHOD
    scorers = (
        [AngleFirstScorer(config=ANGLE_FIRST_GATE_CONFIG)]
        if method == "angle_first" else build_scorers(method=method)
    )

    def best(bd) -> float:
        evs = [e for e in anchors if e.date > bd]
        if not evs:
            return 0.0
        jd = birth_to_jd(bd, 720, birth_data.timezone_offset)
        return max(
            score_candidate(c, evs, jd, scorers).training_score()
            for c in _window_candidates(birth_data, bd, start, end)
        )

    rng = np.random.default_rng(seed)
    lo, hi = NULL_SHIFT_YEARS
    first_event = min(e.date for e in anchors)
    real = best(birth_data.birth_date)
    null = []
    for _ in range(n_natives):
        days = int(rng.choice([-1, 1]) * rng.uniform(lo * 365.25, hi * 365.25))
        if birth_data.birth_date + timedelta(days=days) >= first_event - timedelta(days=365):
            days = -abs(days)
        null.append(best(birth_data.birth_date + timedelta(days=days)))
    p = (1 + sum(x >= real for x in null)) / (n_natives + 1)
    return float(p), float(real), null


def compare_window(
    birth_data: BirthData,
    events: list[LifeEvent],
    approx_time_minutes: int,
    window_minutes: int = 30,
    slot_minutes: int = 10,
    method: str | None = None,
    n_natives: int = N_NULL_NATIVES,
) -> WindowReport:
    """
    Compare the recorded time with its neighbours.

    approx_time_minutes: recorded or remembered time, minutes after local midnight.
    window_minutes: how far either side to look (Tebbs suggests about 30).
    slot_minutes: width of the slots reported (Tebbs uses 10).
    """
    method = method or DEFAULT_METHOD
    start, end = approx_time_minutes - window_minutes, approx_time_minutes + window_minutes
    anchors = [e for e in events if not e.held_out]
    cands = _window_candidates(birth_data, birth_data.birth_date, start, end)
    jd = birth_to_jd(birth_data.birth_date, 720, birth_data.timezone_offset)
    scorers = build_scorers(method=method)
    scored = [score_candidate(c, anchors, jd, scorers) for c in cands]
    totals = np.array([cs.training_score() for cs in scored])

    # sign changes of the angles inside the window
    changes = []
    for prev, cur in zip(cands, cands[1:]):
        for name, a, b in (("ASC", prev.ascendant, cur.ascendant), ("MC", prev.mc, cur.mc)):
            if int(a // 30) != int(b // 30):
                changes.append(SignChange(
                    time=_label(cur.time_minutes), angle=name,
                    from_sign=SIGN_NAMES[int(a // 30)], to_sign=SIGN_NAMES[int(b // 30)],
                ))

    # slots centred on the recorded time
    half = slot_minutes // 2
    slots = []
    n_side = int(np.ceil((window_minutes - half) / slot_minutes)) if window_minutes > half else 0
    for k in range(-n_side, n_side + 1):
        lo = max(start, approx_time_minutes + k * slot_minutes - half)
        hi = min(end, approx_time_minutes + k * slot_minutes + (slot_minutes - half - 1))
        idx = [i for i, c in enumerate(cands) if lo <= c.time_minutes <= hi]
        if not idx:
            continue
        # the closest minute to the slot centre among the top scorers
        centre = approx_time_minutes + k * slot_minutes
        top = max(totals[idx])
        i = min((j for j in idx if totals[j] >= top - 1e-9), key=lambda j: abs(cands[j].time_minutes - centre))
        slots.append(WindowSlot(
            start=_label(lo), end=_label(hi), best_time=_label(cands[i].time_minutes),
            best_time_minutes=cands[i].time_minutes, score=round(float(totals[i]), 3),
            ascendant=_position(cands[i].ascendant), midheaven=_position(cands[i].mc),
            contains_recorded_time=lo <= approx_time_minutes <= hi,
            contacts=[
                Contact(event=h.event_description, technique=h.technique,
                        contact=h.natal_point, orb=h.orb, score=round(h.score, 3))
                for h in sorted(scored[i].technique_scores, key=lambda h: -h.score)
            ],
        ))

    rec_i = approx_time_minutes - start
    top = float(totals.max())
    best_i = min((j for j in range(len(cands)) if totals[j] >= top - 1e-9),
                 key=lambda j: abs(cands[j].time_minutes - approx_time_minutes))
    p, _, _ = window_chance_test(birth_data, events, start, end, n_natives=n_natives, method=method)

    rec_slot = next(s for s in slots if s.contains_recorded_time)
    best_slot = max(slots, key=lambda s: s.score)
    if not anchors or p > CHANCE_ALPHA:
        verdict = "no_preference"
        summary = (
            f"The events do not discriminate between times in this window (p = {p:.2f}, needs "
            f"≤ {CHANCE_ALPHA}): unrelated birth dates fit them as well at these clock times. "
            f"Keep the recorded time {_label(approx_time_minutes)}; the differences between slots are not evidence."
        )
    elif best_slot is rec_slot or rec_slot.score >= best_slot.score - 1e-9:
        verdict = "supports_recorded"
        summary = (
            f"The events fit this window better than chance (p = {p:.2f}) and the slot around the "
            f"recorded time {_label(approx_time_minutes)} scores highest."
        )
    else:
        verdict = "prefers_other"
        summary = (
            f"The events fit this window better than chance (p = {p:.2f}) and prefer "
            f"{best_slot.start}–{best_slot.end} (best minute {best_slot.best_time}) over the slot around "
            f"the recorded time {_label(approx_time_minutes)}. Treat it as a working chart to confirm with later events."
        )
    if changes:
        summary += " Angle sign changes inside the window: " + "; ".join(
            f"{c.angle} {c.from_sign}→{c.to_sign} at {c.time}" for c in changes) + "."

    return WindowReport(
        recorded_time=_label(approx_time_minutes), window_minutes=window_minutes,
        slot_minutes=slot_minutes, anchor_events=len(anchors),
        recorded_ascendant=_position(cands[rec_i].ascendant),
        recorded_midheaven=_position(cands[rec_i].mc),
        recorded_score=round(float(totals[rec_i]), 3),
        best_time=_label(cands[best_i].time_minutes), best_score=round(top, 3),
        minutes_from_recorded=cands[best_i].time_minutes - approx_time_minutes,
        chance_p_value=round(p, 4), verdict=verdict, summary=summary,
        sign_changes=changes, slots=slots,
    )
