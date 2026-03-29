# Astrology Rectifier

Agentic birth-time rectification system. Given a person's date and place of birth plus a
set of hard-dated life events, the system narrows the unknown birth time to a precise window
using three independent astrological timing techniques and a four-loop convergence algorithm.

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
    ├── progressions.py         # Technique 1 — Secondary progressions / progressed Moon  [ACTIVE, weight 3.0]
    ├── transits.py             # Technique 2 — Outer-planet transits to angles           [ACTIVE, weight 2.0]
    ├── primary_directions.py   # Technique 3 — Ptolemaic primary directions (OA/Naibod)  [ACTIVE, weight 1.0]
    ├── solar_arc.py            # Solar arc directions                                     [DISABLED, weight 0.0]
    └── profections.py          # Annual profections / lord of year                        [DISABLED, weight 0.0]
```

---

## The 4-Loop Algorithm

### Loop 0 — Morin Structural Pre-filter
Analyzes biography text with keyword matching (optionally blended with an LLM-generated
prior) to assign probability weights to each of the 12 rising signs. Signs below 5% are
excluded from candidate generation. Reduces the candidate pool from 288 → ~72.

Key design: sports/football biographies correctly activate 5th and 9th house keywords,
giving Sagittarius rising appropriate probability mass (calibrated via Neymar test).

### Loop 1 — Broad Scoring Pass (15-min resolution)
Generates the filtered candidate grid at 15-minute intervals, scores every candidate
against all anchor events using the three active techniques, then applies a Bayesian update
combining the Morin prior with the **null-baseline-adjusted** likelihood scores.

The null baseline (median score of 50 random birth times) is subtracted before
exponentiating — only candidates that beat random chance receive a posterior boost.

### Loop 2 — Narrow Pass (1-min resolution)
Re-scores ±30 minutes around the Loop 1 winner at 1-minute resolution with tighter orbs.
Runs a bootstrap stability analysis (perturb event dates ±3 days, measure score variance)
and cross-validates against held-out events.

### Loop 3 — House System Consensus Check
Re-runs Loop 1 for all four house systems (Placidus, Koch, Whole Sign, Regiomontanus).
Reports consensus count (0–4). Result is provisional if fewer than 2 systems agree.

---

## Scoring Techniques

The "Rifle, Not Shotgun" methodology (see `LESSONS.md`). Only three techniques are active;
each was selected based on its empirically-measured birth-time sensitivity:

| # | Technique | Weight | Birth-time sensitivity | Checks/event |
|---|---|---|---|---|
| 1 | Secondary progressions — progressed Moon | **3.0** | **HIGH** ~0.54°/hour | ~15 |
| 2 | Outer-planet transits to angles | **2.0** | Moderate ~4°/hour (via angles) | ~16 |
| 3 | Primary directions (Naibod arc + OA) | **1.0** | Moderate (via angle RA/OA) | ~40 |
| — | Solar arc directions | ~~1.2~~ → **0.0** | NEGLIGIBLE ~0.04°/hour | disabled |
| — | Annual profections | ~~1.0~~ → **0.0** | ZERO (2-hour blocks) | disabled |

**Why progressions score highest:** the progressed Moon moves ~13°/day in ephemeris time,
translating to ~0.54°/hour of birth-time shift. A 30-minute birth-time error moves the
progressed Moon ~0.27° — detectable within a 1.5° orb. No other technique has this
sensitivity.

**Why solar arc and profections are disabled:** Solar arc sensitivity is ~0.04°/hour
(all discrimination comes from natal angles, making it redundant with transits). Profections
change sign every ~2 hours, giving zero minute-level discrimination.

**Transit rules:** only the 4 natal angles are tested (not house cusps). Only conjunction
and opposition aspects score. Orb: 1.5° standard / 0.75° tight mode.

---

## Event Taxonomy

Events are classified into types that map to relevant house axes. The progressed Moon ingress
scorer awards bonus points when the Moon enters a house matching the event's house type:

| Type | Houses tested |
|---|---|
| `marriage` | 7, 5 |
| `divorce` | 7, 12 |
| `death_of_parent` | 4, 10 |
| `accident` / `illness` | 1, 8, 12 |
| `career_peak` | 10, 6 |
| `surgery` | 8, 12, 6 |
| `birth_of_child` | 5, 11 |
| `custody_loss` | 5, 4, 12 |
| `legal_restriction` | 4, 10, 12, 7 |
| `hospitalization` | 12, 6, 1 |
| `emigration` | 9, 4 |
| `publication` | 9, 3 |

---

## Quality Flags

Every result carries explicit quality indicators:

- **`is_provisional`**: true when posterior < 40%, house consensus < 2, cross-validation
  rate < 80%, or fewer than 4 anchor events.
- **`uncertainty_minutes`**: 5 min (≥8 events, stable bootstrap), 15 min (≥4 events),
  or 30 min (sparse data).
- **`house_system_consensus`**: 0–4. Below 3 is a warning; 0 means the problem is
  underdetermined and no reliable time can be output.
- **`hit_event_count`**: number of distinct anchor events with at least one technique hit.
  A good result has hits on ≥ 60% of anchor events.
- **Event diversity warning**: if >60% of events share the same type (e.g. all career
  peaks), the system warns of systematic house bias in the scoring output.

---

## Known Limitations & Calibration Notes

### 1. Event selection is the most critical input factor

No amount of algorithmic sophistication compensates for a bad event set. The system's
accuracy is bounded by how well the chosen events discriminate between birth times.

**Golden rules for event selection:**
- Minimum **6 training events**, spread across **at least 3 distinct years**
- Maximum **1 cluster** (events within 365 days of each other) — clustered events count as
  ~0.5 effective events after the temporal clustering penalty
- Minimum **3 distinct event types** — avoid all-career or all-health sets
- Include at least one event where the year is **uniquely isolated** (no other events within
  2 years), so a single strong transit or progression can discriminate without interference

### 2. Temporal clustering creates false attractors

Events within a 365-day window receive a 50% clustering penalty, but a planet transiting
slowly through one area of the sky can still generate multiple tight-orb hits across
all clustered events. This creates a false attractor at whichever chart has that planet
conjunct or opposite one of its angles.

**Britney Spears case study:** Events 2, 3, 4 (breakdown, custody loss, conservatorship)
all fall within a 12-month window in 2007-2008. Saturn was at ~20° Leo that entire year.
Any chart with ASC near 20° Leo captures Saturn conjunct ASC for all three events,
generating 3× the score boost despite the clustering penalty. The correct Libra 01:30
chart scores only 8 pts (primary directions only) vs 44.5 pts for Leo 22:13. The system
correctly returns PROVISIONAL with 1/4 house consensus — the honest answer.

**Fix:** add at least one event from a different year for each cluster you have.

### 3. Coincidental geometry cannot be disproved without a null-hypothesis model

Even with the null baseline, a chart that happens to have three independent tight-orb
coincidences (e.g. Saturn ⊙ ASC at 0.1°, Neptune ☍ ASC at 0.07°, ProgMoon → H5 at 0.2°)
will win over the correct chart that has no close aspects at the given events. The null
baseline filters out charts that score at random-chance levels, but cannot distinguish
genuine from coincidental tight geometry.

This is an irreducible limitation for cases with fewer than ~6 independent event clusters.

### 4. Historical timezone must be verified manually

The system trusts the `timezone_offset` value. For politically complex regions or pre-1970
births, verify manually:
- Brazilian DST (BRST = UTC-2 vs BRT = UTC-3): Feb 5, 1992 was in DST → correct offset is -2.0
- Wartime double summer time (UK 1940-45: UTC+2 instead of UTC+1)
- LMT (Local Mean Time) for births before standardized zones
- Colonial timezone changes

A wrong offset shifts all angles by ~15° per hour of error — equivalent to swapping the
entire rising sign.

### 5. Primary directions require geographic latitude for full accuracy

The current OA implementation uses `geo_lat=0` as a fallback when the geographic latitude
is not passed through to the scorer. For latitudes far from the equator, the ascensional
difference can be several degrees, materially affecting which direction perfects at what age.
Store `geo_lat` on `CandidateChart` to enable full accuracy.

### 6. Morin filter quality affects the candidate pool

The rule-based prior uses keyword matching on biography text. If the biography is sparse or
uses unusual vocabulary (e.g. "footy" instead of "football"), relevant keywords may not fire.
Use `use_llm: true` with an Anthropic API key to blend the rule-based prior with an LLM
judgment that understands context more robustly.

---

## Calibration Results

| Person | Confirmed time | System result | Status |
|---|---|---|---|
| Neymar Jr. (5 Feb 1992) | 02:15 Sagittarius (AA) | 02:42 Sagittarius (±27 min) | **PASS** — within 45 min |
| Britney Spears (2 Dec 1981) | 01:30 Libra (AA) | 22:13 Leo PROVISIONAL | **FAIL** — event clustering (see §2) |

Britney's failure is correctly diagnosed and flagged: the system returns PROVISIONAL with
house consensus 1/4, signaling the result is unreliable. Adding events from 1998
(debut single), 2004 (second marriage), and 2005 (first child) would provide three
additional independent time samples and likely converge on Libra.

---

## Roadmap

See `LESSONS.md` for the full methodology evolution log.

### Near-term

- **[ ] Pass `geo_lat` through to `PrimaryDirectionScorer`** — currently uses 0.0 fallback,
  losing the benefit of the oblique ascension implementation for non-equatorial charts
- **[ ] Event quality scorer** — before rectification, estimate how many independent
  time-samples the event set provides (count non-clustered events across distinct years);
  warn if < 4 independent samples
- **[ ] Expand Britney event set** — add 1998 debut, 2004 Federline marriage, 2005 first
  child. These three events span different years and houses, providing the missing signal

### Medium-term

- **[ ] Progressed angles via RAMC** — proper progressed ASC/MC computed from the
  progressed RAMC (not the crude `ASC + age * 1.0°` approximation that was removed).
  This would add a second progressed point beyond the Moon.
- **[ ] Return-chart scoring** — Solar Return and Lunar Return charts hitting natal angles
  are sensitive to birth time and mathematically independent from directions/progressions
- **[ ] Aspect deduplication across events** — if the same planet-angle pair hits in 3
  clustered events, count it once at full weight rather than 3× at half weight
- **[ ] Time-range events** — allow events with `date_certainty_days > 1` to score at
  the closest approach within the window rather than at noon on the given date

### Long-term

- **[ ] Multi-person cross-validation corpus** — calibrate on 10+ AA-rated birth times
  to tune weights and orbs statistically rather than by hand
- **[ ] LLM event extraction quality** — evaluate event dates extracted by LLM against
  manually-verified ground truth; add confidence scores to extracted events
- **[ ] Ayanamsa support** — for Vedic rectification (sidereal zodiac)

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
    "latitude": -23.5225,
    "longitude": -46.1861,
    "timezone_offset": -2.0,
    "biography": "Brazilian professional footballer. Plays as forward. Santos FC, Barcelona, PSG."
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

120 tests covering: ephemeris calculations, all 5 scoring techniques (3 active + 2
disabled), Morin filter, 4-loop rectifier, calibration against Neymar and Britney,
and FastAPI endpoints.

---

## LLM Integration

Set `ANTHROPIC_API_KEY` to enable:
- Automatic event extraction from biography text (`/extract-events`)
- LLM-generated rising sign prior blended with rule-based prior (`use_llm: true`)

Without the key, the system falls back to rule-based keyword analysis and requires
events to be provided manually.
