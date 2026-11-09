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

The engine has two scoring methods. **`angle_first` is the default**; `classic` is the
original five-technique sum, kept for comparison (`Rectifier(..., method="classic")`).

### Loop 0 — Morin prior
Analyzes biography text with keyword matching (optionally blended with an LLM-generated
prior) to assign a probability to each of the 12 rising signs.

- `angle_first`: the prior only **weights** candidates. All twelve rising signs are
  searched, and the prior is scaled so that it can break a tie but cannot overrule the
  events. Reading the Ascendant from appearance or character is listed among the less
  reliable rectification methods by Tebbs (ch. 1).
- `classic`: signs below 5% are **excluded** before any event is scored. This is what
  sent the Neymar calibration four hours off (see Calibration Notes).

### Loop 1 — Broad pass
Scores every candidate against the anchor events, applies the Bayesian update with the
Morin prior, and runs the chance-level test (see Quality Flags). Grid: 2 minutes for
`angle_first` (about 0.5° of MC, half its 1° orb), 15 minutes for `classic`.

### Loop 2 — Narrow pass (1-minute resolution)
Re-scores ±30 minutes around the Loop 1 winner at 1-minute resolution with half the orb.
Runs a bootstrap stability analysis (perturb event dates ±3 days, measure score
variance) and cross-validates against held-out events.

### Loop 3 — House system consensus (`classic` only)
Re-runs Loop 1 for Placidus, Koch, Whole Sign and Regiomontanus and counts agreement.
`angle_first` scores only ASC and MC, which are identical in every house system, so
the check cannot disagree and is skipped (reported as 4).

---

## Scoring

### `angle_first` (default)
Follows the procedure in Carol Tebbs, *The Complete Book of Chart Rectification*
(Llewellyn), chapters 1, 2 and 8 and appendices B and C.

| Rule | Source |
|---|---|
| Only the four angles are scored; house cusps are ignored | Tebbs |
| Solar arc directions lead the search (planets to angles, angles to planets) | Tebbs |
| Orb of 1°, and a closer contact counts for more | Tebbs (orb); linear fade is our choice |
| Progressions (Sun–Mars, plus progressed Moon to natal planets) and Saturn–Pluto transits confirm | Tebbs |
| Painful events (deaths, divorce, accident, illness, arrest…) weigh 4× | Tebbs estimates four to five times |
| One contact per technique per event (the closest) | our choice |
| Confirmations count only on top of a solar arc contact, at half weight | our choice, measured below |
| Only conjunctions with the angles (this covers oppositions); no squares | our choice |

Not implemented from the book: starting from the natal 1–2° hard-aspect "hooks", and
Dobyns' rule that an event must show in all three systems. The strict version of that
rule (`mode="strict"`) found 0 of 4 planted charts in a pilot run and was not pursued.

### `classic`

| # | Technique | Weight | Basis |
|---|---|---|---|
| 1 | Primary directions (Naibod arc) | **2.0** | RA for MC/IC, oblique ascension for ASC/DSC |
| 2 | Secondary progressions / progressed Moon | **1.5** | 1 day = 1 year |
| 3 | Solar arc directions | **1.2** | Progressed Sun arc (correlated with #2) |
| 4 | Annual profections / lord of year | **1.0** | Different time scale |
| 5 | Outer-planet transits to angles | **1.0** | Real-time positions |

Hits to angles and to the house cusps relevant to the event type score, up to three
per technique per event, with orbs up to 2°. Progressed angles use the solar arc in
longitude on the MC and the tables of houses for the ASC (Tebbs, appendices B and C).

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
All figures are from synthetic charts (random date, place and birth minute). "Planted"
events are dated when a solar arc contact between a natal planet and an angle of the
true chart is exact. **This tests whether the engine can find the kind of signal it
assumes. It says nothing about whether real lives follow that signal.**

**Recovery: best-scoring time within 10 minutes of the true time** (20 charts per cell)

| Planted events | Error in event dates | `classic` | `angle_first` |
|---|---|---|---|
| 10 | none | 5 | 17 |
| 5 | none | 4 | 15 |
| 10 | ±6 months (std. dev.) | 2 | 17 |
| 5 | ±6 months (std. dev.) | 2 | 11 (5 with the earliest-contact events of `eval.benchmark`) |

`classic` was run on a 4-minute grid here (finer than its normal 15), `angle_first` on
its 2-minute grid. Other combinations tried for `angle_first`, same four rows: solar
arc alone 17, 14, 18, 12; transits and progressions scoring independently at half
weight 17, 14, 15, 9; confirmations at full weight 17, 11, 12, 8.

**Chance-level gate** (`chance_level_test`, 30 control natives)

| Measure | `angle_first` | `classic` |
|---|---|---|
| False positives on random event lists (p ≤ 0.05) | 0 of 40 | 3 of 70 |
| Power, 10 planted events with exact dates | 12 of 20 | 0 of 12 (8 events) |
| Power, 5 planted events, dates ±6 months | 2 of 20 | not run |

What this shows:

1. With ten well-dated events the new scoring usually finds a planted time; the old one
   rarely did.
2. **Few events and imprecise dates are the limit.** With five events dated to within
   about six months, recovery falls to between a quarter and a half depending on which
   contacts the events fall on, and the gate passes 2 times in 20.
3. The gate is conservative: it needs the real chart to beat all 30 controls, and
   passes only 12 of 20 charts that do carry a planted signal.

Reproduce the recovery table with `python -m eval.compare_variants 20 2` (arguments:
charts, grid minutes, variants, events, date error in years) and the gate figures with
`python -m eval.benchmark --cases 20 --events 10` (see `--help`).

### Homogeneous event set bias (Neymar calibration test)
When all anchor events are of the same type (e.g. all `career_peak`), the scoring
systematically favors whichever birth time best fits that house axis — regardless of
the true birth time. This is the single most important failure mode.

**Neymar Jr. (5 Feb 1992, Mogi das Cruzes, 02:15 local summer time = UTC−2, ASC 13°53′ Sag):**
the fixture has 4 anchor events (2 career peaks, 1 relocation, 1 accident) and 1
held-out event. Results with the correct time zone:

| Run | Result | Distance from 02:15 | Chance-level p |
|---|---|---|---|
| `classic`, Morin prior | 06:30 | 4 h 15 min | 0.23 |
| `classic`, uniform prior | 02:42 | 27 min | 0.23 |
| `angle_first` (Morin or uniform prior) | 02:46 | 31 min | 0.77 |

In `classic` the Morin prior gives Sagittarius, the true rising sign, 2.3%. That is
under the 5% cut, so Sagittarius is removed before any event is scored: **the 4-hour
miss comes from the Morin filter, not from the event scoring.** `angle_first` no longer
excludes signs. In every run the chance-level test fails, so the result is reported as
UNDERDETERMINED: four events are too few to tell this chart from unrelated ones, and
landing about half an hour from the recorded time may be luck.

**Britney Spears (2 Dec 1981, McComb MS, 01:30 CST):** 5 anchor events, three of them
inside twelve months. `classic` gives 04:47 (p = 0.32), `angle_first` gives 18:26
(p = 0.42). Both are far from 01:30 and both are reported as UNDERDETERMINED.

> **Correction (timezone).** Earlier versions of the calibration fixture used UTC−3.
> Summer time ran from 20 Oct 1991 to 9 Feb 1992 and included São Paulo state (Decreto
> de 25 de setembro de 1991, arts. 1–2), so the correct offset is UTC−2. It reproduces
> the ASC 13°53′ Sagittarius published by Astrotheme; UTC−3 gives 28°13′ Sagittarius.
> With UTC−3 the "confirmed time scores ≥ 30% of the top" check passed by accident.
> With the correct offset 02:15 scores about 16% of the top 15-minute candidate
> (03:00), so that test is marked `xfail(strict=True)`.

**Rule of thumb:** include at least 3 distinct event types and keep no single type above
50% of the anchor set.

### Morin filter (corrected)
Initial implementation excluded Sagittarius from athlete/footballer biographies because
sports keywords were not mapped to 9th-house themes. Fixed: football/soccer/sport/transfer
now map to houses [5, 9]; `HOUSE_TO_RISING_SIGNS[9]` now includes Sagittarius (sign 9),
which naturally rules the 9th house. Uniform prior floor raised from 0.1 → 0.25 per
sign to prevent total exclusion on sparse keyword data.

**Still open:** with the calibration biography the prior still gives Sagittarius only
2.3%, so the true rising sign is excluded (see the Neymar table above). Because the cut
is hard, event evidence can never bring an excluded sign back.

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

149 tests (plus 1 documented expected failure) covering: ephemeris calculations, all 5
scoring techniques, Morin filter, 4-loop rectifier, FastAPI endpoints, and regression
tests for the timing fixes in `tests/test_timing_fixes.py`, the chance-level gate in
`tests/test_chance_level.py` and the angle-first scorer in `tests/test_angle_first.py`.

---

## LLM Integration

Set `ANTHROPIC_API_KEY` to enable:
- Automatic event extraction from biography text (`/extract-events`)
- LLM-generated rising sign prior blended with rule-based prior (`use_llm: true`)

Without the key, the system falls back to rule-based keyword analysis and requires
events to be provided manually.
