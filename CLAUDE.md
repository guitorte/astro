# Claude Code Instructions — Astrology Rectifier

This file is read by Claude at the start of every session. It encodes mistakes that were
made, investigated, and fixed — to prevent repeating them.

---

## Critical: Do Not Repeat These Mistakes

### MISTAKE 1 — natal_jd-at-noon in progressions
**What happened:** `score_candidate()` passed a noon reference JD to the progression
scorer instead of the candidate's actual birth JD. The progressed Moon was computed from
noon for every candidate, making it identical across all candidates (±5° error) and
destroying the Moon's discriminating power.

**Rule:** In `progressions.py` and `solar_arc.py`, always pass `candidate.julian_day`
(not `natal_jd`) to `secondary_progressed_jd()` and `solar_arc_for_age()`.

`natal_jd` is the noon reference used for planet lookups and Bayesian prior alignment.
`candidate.julian_day` is the actual birth moment for time-derivative calculations.
These are NOT interchangeable.

---

### MISTAKE 2 — Coverage ratio as a Bayesian multiplier
**What happened:** `hit_event_count / n_anchor_events` was used as a multiplicative
factor in `bayesian_update()`. This caused a Cancer 18:15 chart (weak hits on 4/4 events)
to beat a Sagittarius 02:45 chart (strong hits on 3/4 events), sending Loop 2 to zoom into
the wrong sign entirely.

**Rule:** `hit_event_count` is a diagnostic display metric only. Do NOT multiply or
otherwise reduce the Bayesian likelihood by coverage ratio. A chart with one tight hit
(0.04°) on 3 events is more meaningful than one with loose hits on 4 events.

If you are tempted to penalize candidates for low coverage, stop and think about whether
you are punishing precision.

---

### MISTAKE 3 — Adding techniques without sensitivity analysis
**What happened:** Solar arc (sensitivity ~0.04°/hour) and profections (sensitivity ~0°)
were given positive weights alongside the progressed Moon (sensitivity ~0.54°/hour). With
equal weighting, noise drowned signal. The system accumulated 500+ checks per event,
making coincidental hits indistinguishable from genuine ones.

**Rule:** Before assigning a non-zero weight to any technique, answer: "How many degrees
does the relevant point move per hour of birth-time shift?" If the answer is < 0.1°/hour,
weight must be 0.0.

Current sensitivity table:
- Progressed Moon: ~0.54°/hr → weight 3.0 ✓
- Transits to angles: ~4°/hr (via angles) → weight 2.0 ✓
- Primary directions (OA): moderate → weight 1.0 ✓
- Solar arc: ~0.04°/hr → weight 0.0 (disabled)
- Profections: ~0°/hr → weight 0.0 (disabled)

Do not re-enable solar arc or profections without a concrete plan to address their
near-zero birth-time sensitivity.

---

### MISTAKE 4 — Timezone assumed without DST check
**What happened:** Neymar's birth (Feb 5, 1992, Brazil) was initially given `timezone_offset=-3.0`
(BRT standard time). Feb 5, 1992 was in Brazilian DST season → correct offset is `-2.0`
(BRST). Using -3.0 shifted all charts by ~14° in ASC, placing the winning candidate in the
wrong sign.

**Rule:** Any time you see a Brazilian birth between October and February/March, the
correct offset is -2.0 (BRST), not -3.0 (BRT). When adding new test fixtures or
rectifying Brazilian births, verify the DST status for the specific date.

More generally: for any birth pre-1970 or in a politically complex region, call
`validate_timezone()` from `src/ephemeris.py` as a sanity check. A 1-hour timezone
error is equivalent to changing the rising sign.

---

### MISTAKE 5 — Interpreting PROVISIONAL as a bug
**What happened:** When Britney Spears' rectification returned PROVISIONAL with 1/4
house consensus, there was pressure to "fix" it by tweaking algorithm parameters.
Root-cause analysis revealed the failure is in the input event set (events clustered
in a 12-month window, Saturn dominating all clustered events), not the algorithm.

**Rule:** PROVISIONAL with low consensus is a CORRECT and HONEST output for ambiguous
input. Do not adjust algorithm parameters to make PROVISIONAL cases converge — that
produces confident wrong answers. Instead, improve the event set (see below).

When Britney returns PROVISIONAL, the fix is:
1. Add isolated events from other years (1998 debut, 2004 Federline marriage, 2005 first child)
2. Move conservatorship end (2021-11-12) from held-out to training

---

### MISTAKE 6 — `ecliptic_to_ra()` renamed without updating tests
**What happened:** `primary_directions.py` was refactored to replace `ecliptic_to_ra()`
with `ecliptic_to_equatorial()` (which returns both RA and declination). The import in
`tests/test_scoring.py` still referenced the old name, causing a collection error.

**Rule:** After renaming any exported function in a scorer, grep for all usages
before committing:
```bash
grep -r "ecliptic_to_ra" tests/ src/
```

---

## Architecture Invariants

These must not be violated without explicit justification:

1. **`candidate.julian_day` for time-derivative calculations.** `natal_jd` is the noon
   reference only.

2. **Transits target 4 angles only** (ASC/MC/DSC/IC). Not house cusps. House cusps are
   within a few degrees of angles in most house systems and adding them doubles the
   checks without adding discriminating power.

3. **Transits score conjunction and opposition only.** Not squares, trines, sextiles.
   These were removed because they created false attractors. If you want to re-add them,
   run a null-baseline test first and confirm they don't raise the random-chart score by more than 10%.

4. **The null baseline must be recomputed any time active scorers or orbs change.** It is
   not a fixed constant. It is re-computed at runtime in every Loop 1 and Loop 2 pass.

5. **Cap hits at 3 per (technique, event) pair.** `cap_hits()` in `base.py` enforces this.
   Do not remove or bypass this cap.

6. **Clustering penalty of 0.5× for events within 365 days of the previous event.**
   This is a minimum — do not reduce it. The Britney case shows that even 0.5× is
   insufficient when a slow-moving planet is near an angle throughout the cluster window.

---

## Event Set Requirements (for new test cases)

When writing a new calibration test, the event set must satisfy:

| Requirement | Minimum |
|---|---|
| Total anchor events | 5 (excluding held-out) |
| Non-clustered years | 3 distinct years with no adjacent event within 365 days |
| Distinct event types | 3 |
| No single type > 50% of anchor events | — |
| Held-out events | 1 (used for cross-validation only) |

If you cannot satisfy these requirements for a given celebrity, document it explicitly in
the test with `pytest.mark.xfail(reason="insufficient event diversity")` so the known
limitation is visible rather than hidden.

---

## Running the Test Suite

```bash
pytest tests/ -v            # full suite (120 tests, ~10s)
pytest tests/test_calibration.py -v   # calibration only
pytest tests/ -k "Neymar"  # single celebrity
```

All 120 tests must pass before committing. The calibration tests are the highest-value
signal — a regression in `test_confirmed_time_within_uncertainty_of_winner` or
`test_sagittarius_in_top_candidates` means a meaningful behavioral change.
