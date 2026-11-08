# Astrology Rectifier

Agentic birth time rectification system. Given a celebrity's date and place of birth
plus a set of hard-dated life events, the system narrows the unknown birth time to a
precise window using five independent astrological timing techniques and a four-loop
convergence algorithm.

---

## Architecture

```
src/
├── models.py               # Core data models (Pydantic)
├── ephemeris.py            # Swiss Ephemeris wrapper (pyswisseph, Moshier built-in)
├── morin_filter.py         # Loop 0: rising sign prior (rule-based + LLM blend)
├── rectifier.py            # 4-loop orchestrator + Bayesian update
├── llm_extractor.py        # Claude API event extraction (optional)
├── api.py                  # FastAPI service
└── scoring/
    ├── primary_directions.py   # Technique 1 — Ptolemaic primaries (Naibod arc)
    ├── progressions.py         # Technique 2 — Secondary progressions + progressed Moon
    ├── solar_arc.py            # Technique 3 — Solar arc directions
    ├── profections.py          # Technique 4 — Annual profections / lord of year
    └── transits.py             # Technique 5 — Outer-planet transits to angles
```

---

## The 4-Loop Algorithm

### Loop 0 — Morin Structural Pre-filter
Analyzes biography text with keyword matching (optionally blended with an LLM-generated
prior) to assign probability weights to each of the 12 rising signs. Signs below 5%
are excluded from candidate generation. Reduces the candidate pool from 288 → ~72.

Key design: sports/football biographies correctly activate 5th and 9th house keywords,
giving Sagittarius rising appropriate probability mass (calibrated via Neymar test).

### Loop 1 — Broad Scoring Pass (15-min resolution)
Generates the filtered candidate grid at 15-minute intervals, scores every candidate
against all anchor events using all five techniques, then applies a Bayesian update
combining the Morin prior with the likelihood scores. Repeats if the top candidate's
posterior ratio is below 3×. Also runs the chance-level test (see Quality Flags).

### Loop 2 — Narrow Pass (1-min resolution)
Re-scores ±30 minutes around the Loop 1 winner at 1-minute resolution with tighter
orbs. Runs a bootstrap stability analysis (perturb event dates ±3 days, measure score
variance) and cross-validates against held-out events.

### Loop 3 — House System Consensus Check
Re-runs Loop 1 for all four house systems (Placidus, Koch, Whole Sign, Regiomontanus).
Reports consensus count (0–4). Result is provisional if fewer than 2 systems agree.

---

## Scoring Techniques

Techniques are weighted by their mathematical independence:

| # | Technique | Weight | Basis |
|---|---|---|---|
| 1 | Primary directions (Naibod arc) | **2.0** | Right Ascension — fully independent |
| 2 | Secondary progressions / progressed Moon | **1.5** | 1 day = 1 year |
| 3 | Solar arc directions | **1.2** | Progressed Sun arc (correlated with #2) |
| 4 | Annual profections / lord of year | **1.0** | Different time scale — independent |
| 5 | Outer-planet transits to angles | **1.0** | Real-time positions |

Only hits to **time-sensitive points** score: angles (ASC/MC/DSC/IC) and house cusps
relevant to the event type. Transits to natal planets that are not angles are excluded —
they fire for everyone born in the same year and carry zero discriminating power.

---

## Event Taxonomy

Events are classified into types that map to relevant house axes:

| Type | Houses tested |
|---|---|
| `marriage` | 7, 5 |
| `divorce` | 7, 12 |
| `death_of_parent` | 4, 10 |
| `accident` / `illness` | 1, 8, 12 |
| `career_peak` | 10, 6 |
| `surgery` | 8, 12, 6 |
| `birth_of_child` | 5, 11 |
| `emigration` | 9, 4 |
| `publication` | 9, 3 |

---

## Quality Flags

Every result carries explicit quality indicators:

- **`is_provisional`**: true when posterior < 40%, house consensus < 2, cross-validation
  rate < 80%, or fewer than 4 anchor events.
- **`uncertainty_minutes`**: 5 min (≥8 events, stable bootstrap), 15 min (≥4 events),
  or 30 min (sparse data).
- **`chance_p_value`**: does the event list fit this native better than it fits
  unrelated people? The best score over the whole day for the real birth date is
  ranked against the same statistic for 30 other birth dates at the same place
  (shifted 1–15 years). If p > 0.05 the result is **UNDERDETERMINED**: the top time is
  kept for inspection, but `confidence_score` is 0 and `uncertainty_minutes` is 720.
- **`house_system_consensus`**: 0–4. Below 3 is a warning. It is only a necessary
  condition: ASC and MC, which drive primary directions and solar arcs, are identical
  in every house system, so 4/4 is close to automatic and is never treated as
  independent evidence.
- **Event diversity warning**: if >60% of events share the same type (e.g. all career
  peaks), the system warns of systematic house bias in the scoring output.

---

## Known Limitations & Calibration Notes

### Current measured performance (benchmark)
`python -m eval.benchmark --cases 12 --natives 30` plants up to 8 events on dates when
a solar-arc-directed natal planet exactly reaches a natal angle of a synthetic chart,
then asks the engine to find that chart. Result on the 15-minute grid:

| Measure | Result | Target |
|---|---|---|
| False-positive rate on random event lists | 1–2 of 12 | about 5% |
| Power on planted event lists (p ≤ 0.05) | 0 of 12 | high |
| True time ranked first | 0 of 12 (median rank 23/96) | most cases |

So today the engine cannot tell a chart from unrelated charts **even when the events
were built to fit it**. The cause is saturation: each candidate offers roughly 140
direction targets (10 planets × 4 angles × direct/converse, angles → planets, cusps),
spread over about 110° of arc, with orbs of 1–2 years. Nearly any date hits several of
them, for nearly any candidate. Each event therefore carries almost no information.
Any scoring change should be judged on this benchmark first; the two celebrity charts
are too few to separate a better model from a lucky one.


### Homogeneous event set bias (Neymar calibration test)
When all anchor events are of the same type (e.g. all `career_peak`), the scoring
systematically favors whichever birth time best fits that house axis — regardless of
the true birth time. This is the single most important failure mode.

**Neymar Jr. (5 Feb 1992, Mogi das Cruzes, 02:15 local summer time = UTC−2, ASC 13°53′ Sag):**
with 5 career peaks and 2 injury events the engine converges on a moment around 08:10 UT
(05:10 if the clock is read as UTC−3, 06:10 as UTC−2), not the real 04:15 UT.

> **Correction (timezone).** Earlier versions of the calibration fixture used UTC−3.
> Brazil was on summer time from 20 Oct 1991 to 9 Feb 1992, so the correct offset is
> UTC−2 (it reproduces the published ASC 13°53′ Sagittarius; UTC−3 gives ~28° Sag).
> With UTC−3 the test was scoring a moment one hour away from the real birth, and the
> "confirmed time scores ≥ 30% of the top" check passed by accident. With the correct
> offset the real moment ranks ~76/96 at 15-min resolution and scores ~16% of the top
> candidate. That test is now marked `xfail(strict=True)` until the scoring improves.
> The failure is in the scoring model, not the data entry.

**Rule of thumb:** include at least 3 distinct event types and keep no single type above
50% of the anchor set.

### Morin filter (corrected)
Initial implementation excluded Sagittarius from athlete/footballer biographies because
sports keywords were not mapped to 9th-house themes. Fixed: football/soccer/sport/transfer
now map to houses [5, 9]; `HOUSE_TO_RISING_SIGNS[9]` now includes Sagittarius (sign 9),
which naturally rules the 9th house. Uniform prior floor raised from 0.1 → 0.25 per
sign to prevent total exclusion on sparse keyword data.

### Historical timezones
The system trusts the `timezone_offset` value provided. For politically complex regions
or pre-1970 births, verify the historical timezone manually (wartime double summer time,
colonial zones, LMT). A wrong offset propagates as a systematic bias across all candidates.

### Physical appearance checks
Not implemented as a scoring factor (too noisy, too culturally specific). The cross-
validation pass and house consensus check serve as the primary sanity filters.

---

## Installation

```bash
pip install -r requirements.txt
```

No external Swiss Ephemeris data files required — the system uses the built-in Moshier
ephemeris (accurate to within ~1 arc-minute for 1800–2100 CE).

---

## Running the API

```bash
uvicorn src.api:app --reload
```

### POST /rectify

```json
{
  "birth_data": {
    "name": "Neymar Jr.",
    "birth_date": "1992-02-05",
    "birth_city": "Mogi das Cruzes",
    "latitude": -23.5228,
    "longitude": -46.1875,
    "timezone_offset": -3.0,
    "biography": "Famous football player..."
  },
  "events": [
    {
      "description": "Professional debut at Santos FC",
      "event_type": "career_peak",
      "date": "2009-03-07",
      "weight": "anchor",
      "held_out": false
    }
  ],
  "use_llm": false
}
```

### POST /morin-filter
Returns probability distribution over 12 rising signs from biography analysis.

### POST /extract-events
Extracts hard-dated life events from biography text via Claude API
(requires `ANTHROPIC_API_KEY`).

### GET /health

---

## Running Tests

```bash
pytest tests/ -v
```

135 tests (plus 1 documented expected failure) covering: ephemeris calculations, all 5
scoring techniques, Morin filter, 4-loop rectifier, FastAPI endpoints, and regression
tests for the timing fixes in `tests/test_timing_fixes.py` and the chance-level gate
in `tests/test_chance_level.py`.

---

## LLM Integration

Set `ANTHROPIC_API_KEY` to enable:
- Automatic event extraction from biography text (`/extract-events`)
- LLM-generated rising sign prior blended with rule-based prior (`use_llm: true`)

Without the key, the system falls back to rule-based keyword analysis and requires
events to be provided manually.
