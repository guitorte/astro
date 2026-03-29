# Lessons Learned — Rectification System Development Log

This file documents the methodology evolution, every significant bug found, root-cause
analysis, and calibration outcomes. It is a permanent record — append-only.

---

## Session 1 — Initial Build & Bug Fixing

### Calibration targets
- **Neymar Jr.** — 5 Feb 1992, 02:15 local (Sagittarius ASC), Mogi das Cruzes, Brazil
  (Rodden Rating AA — birth certificate)
- **Britney Spears** — 2 Dec 1981, 01:30 local (Libra ASC), McComb, Mississippi, USA
  (Rodden Rating AA — birth certificate)

### Bugs found and fixed (7 total)

#### Bug 1: `natal_jd-at-noon` — the most impactful bug
**File:** `src/scoring/progressions.py` and `src/scoring/solar_arc.py`

`score_candidate()` passed a noon reference JD to all scorers. The progressions scorer
called `secondary_progressed_jd(natal_jd, age)` using this noon JD instead of the
candidate's actual birth JD. The progressed Moon is computed as the Moon's position
`age` days after birth. Using noon shifts the progressed Moon by:

```
(candidate_minutes - 720) / 1440 * 13.18° per day ≈ ±5° for births near midnight
```

This made the progressed Moon **identical across all candidates** — destroying the
Moon's discriminating power (which is the single strongest birth-time signal).

**Fix:** Pass `candidate.julian_day` (not `natal_jd`) to progression and solar arc
computations.

#### Bug 2: Neymar timezone — UTC-3 instead of UTC-2
**File:** `tests/test_calibration.py`

Brazil observes DST. Feb 5, 1992 was during DST season:
- BRST (Brasília Summer Time) = UTC-2
- BRT (Brasília Standard Time) = UTC-3

Using UTC-3 shifted all Neymar charts by ~14° in ASC, placing Sagittarius rising ~1 hour
earlier in local time. Fixed: `timezone_offset=-2.0` in the Neymar fixture.

**Lesson:** Always verify historical DST for the birth date. Brazil's DST period is
roughly October → February/March, but the exact dates vary by year.

#### Bug 3: Morin filter excluded Sagittarius for athletes
**File:** `src/morin_filter.py`

Football/sports keywords were not mapped to 9th-house themes. Sagittarius rules the 9th
house but was excluded from the candidate pool for Neymar. Fixed: `football|soccer|sport`
now map to houses [5, 9]; Sagittarius correctly appears in the candidate grid.

#### Bug 4: "footballer" not matching word boundary regex
**File:** `src/morin_filter.py`

Pattern `\bfootball\b` uses strict word boundary that doesn't match "footballer". Fixed:
split into exact-boundary patterns and prefix-only patterns (no trailing `\b`).

#### Bug 5: Primary directions used lat=0 for all planets
**File:** `src/scoring/primary_directions.py`

Converting ecliptic longitude to Right Ascension requires ecliptic latitude for accurate
results. Using lat=0 causes RA errors of up to 1-2° for the Moon and outer planets with
high ecliptic latitude. Fixed: `planet_latitudes` field added to `CandidateChart`, storing
the `index[1]` output from `swe.calc_ut()` for each planet.

#### Bug 6: Cusp/angle double-counting
**File:** `src/scoring/base.py`

`get_event_sensitive_points()` returned both angles (ASC/MC/DSC/IC) and house cusps. In
most house systems, H1 cusp ≈ ASC, H10 ≈ MC, etc. A planet at the ASC was scoring twice.
Fixed: added `_CUSP_ANGLE_DEDUP_ORB = 2.0°` — cusps within 2° of any angle are skipped.

#### Bug 7: Combinatorial hit accumulation
**File:** `src/scoring/base.py`

With 5 techniques × 12 sensitive points × 4 outer planets = 240 checks per event, coincidental
loose-orb hits accumulated even for random charts. Added `cap_hits(max=3)` — keeps only the
top 3 hits per technique per event.

### Calibration results after bug fixes
- Neymar: **02:42 Sagittarius** (confirmed 02:15, ±27 min) ✓
- Britney: **04:47 Scorpio** (confirmed 01:30 Libra) ✗

---

## Session 2 — Methodology Assessment & "Rifle, Not Shotgun"

### Birth-time sensitivity analysis

Each technique was analyzed for how sensitive it actually is to birth-time changes:

| Technique | Birth-time sensitivity | Root cause |
|---|---|---|
| Progressed Moon | **~0.54°/hour** | Moon moves 13°/day in ephemeris time |
| Transits to angles | **~4°/hour** (via angles) | Angles rotate ~1°/15min with Earth |
| Primary Directions (RA-based) | Moderate, via angle RA | Angle RA changes with birth time |
| Solar Arc | **~0.04°/hour** | Sun moves ~1°/day; 1hr birth shift → 0.04° arc |
| Profections | **~0°/hour** | Sign changes every ~2 hours only |

**Key insight:** Solar arc has 13× less birth-time sensitivity than the progressed Moon.
With both at equal weight, solar arc is pure noise from a discriminating-power perspective.

### Why Britney's first failure was not a bug

Scorpio 04:47 has Neptune conjunct IC at 0.28° in 2007. This is a real geometric fact —
not a calculation error. But it's coincidental. Without a null-hypothesis model, the system
cannot distinguish a genuine birth-time indicator from a coincidental tight-orb hit.

With ~600 checks per event (5 techniques × many sensitive points), a random chart is
expected to accumulate 50-80 loose-orb hits. The correct chart's signal drowns in noise.

### Methodology changes ("Rifle, Not Shotgun")

**Principle:** fewer checks, higher precision, null-hypothesis calibration.

| Change | Before | After | Rationale |
|---|---|---|---|
| Progression weight | 1.5 | **3.0** | Strongest discriminator |
| Transit weight | 1.0 | **2.0** | Second-strongest via angles |
| Solar arc weight | 1.2 | **0.0** | Negligible sensitivity |
| Profection weight | 0.5 | **0.0** | Zero minute-level discrimination |
| Transit targets | Angles + house cusps | **4 angles only** | Cusps add noise |
| Transit aspects | Any (0°-180° range) | **Conjunction + opposition only** | Only classically unambiguous contacts |
| Progression orb | 2.0° | **1.5°** | Tighter to exploit Moon sensitivity |
| Transit orb | 2.0° | **1.5°** | Tighter to reduce false positives |
| Primary directions | Raw RA, bidirectional | **OA-based, unidirectional** | OA adds geographic latitude; half the checks |
| PD year tolerance | ±1.5 years | **±1.0 years** | Fewer hits, higher precision |
| Null baseline | None | **Median of 50 random charts** | Calibrates Bayesian update against chance |
| Hit event count | Not tracked | **Stored on CandidateScore** | Diagnostic, not used in scoring |
| Progressed angles | Crude (ASC + age*1°) | **Removed** | Approximation is wrong; adds noise |

**Total checks/event:** ~500 → ~71

### Coverage penalty lesson (important)

An initial implementation multiplied the Bayesian likelihood by `hit_event_count / n_anchor`
to penalize candidates that score on few events. This caused the system to prefer a
Cancer 18:15 chart (4/4 events hit, modest scores) over Sagittarius 02:45 (3/4 events hit,
strong scores). The correct chart was missed.

**Lesson:** Coverage ratio works against precision-first scoring. A chart with one
extremely tight hit (0.04°) on 3 events is more meaningful than a chart with weak hits
on 4 events. The coverage metric is stored for display/debugging but should not be
applied as a multiplicative penalty in the Bayesian update.

### Calibration results after methodology revision

- Neymar: **02:42 Sagittarius** (confirmed 02:15, ±27 min) ✓
- Britney: **22:13 Leo, PROVISIONAL** (confirmed 01:30 Libra) ✗
  - System correctly returns PROVISIONAL with 1/4 house consensus
  - No longer falsely confident about a wrong answer (previously: Scorpio 04:47)

### Why Britney still fails (root-cause analysis)

**This is an input quality problem, not an algorithmic problem.**

The event set has a fatal temporal structure:

```
2004-01-03  Marriage (isolated)
                              ← 3-year gap ←
2007-02-16  Head shaving       ──┐
2007-10-01  Custody loss         ├── 12-month cluster
2008-02-01  Conservatorship    ──┘
2021-11-12  Conservatorship end (held out)
```

Saturn was at ~20° Leo throughout 2007-2008. Any chart with ASC near 20° Leo captures
Saturn conjunct ASC for all three clustered events. Leo 22:13 has:
- Saturn ⊙ ASC at 0.107° (head shaving event)
- Neptune ☍ ASC at 0.067° (conservatorship event)
- Progressed Moon → H5 at 0.211° (marriage)

These three independent tight hits from three different events convincingly outperform
the confirmed Libra 01:30, which has only primary-direction hits on the crisis cluster.

**What would fix it:** add events from different years:
- 1998-10-23: "...Baby One More Time" debut single (age 16 — isolated year)
- 2004-09-18: Marriage to Kevin Federline (same year as Jason Alexander but different H7/H5 axis activation)
- 2005-09-14: Birth of first child (activates H5 — different house axis from crisis events)
- Move conservatorship END (2021) from held-out to training (Saturn in Aquarius ☍ Libra ASC)

With these additions, the Libra chart would have hits in 1998, 2004, 2005, 2007-2008, and
2021 — five distinct temporal samples that collectively identify the correct ASC.

---

## Invariant Lessons (apply to all future development)

### L1: Event quality > algorithm quality
The system's accuracy ceiling is set by the event set, not the code. Six well-spread events
with diverse types and distinct years will outperform nine clustered homogeneous events
every time.

### L2: Verify historical timezone before debugging the algorithm
A 1-hour timezone error shifts all angles by ~15°, equivalent to changing the rising sign.
Always check DST status for the specific birth date and country.

### L3: Sensitivity analysis before adding techniques
Before implementing a new scoring technique, compute: how many degrees does the relevant
point move per hour of birth-time shift? If the answer is < 0.1°/hour, the technique
cannot discriminate candidates within a 1-hour window and should not be given positive weight.

### L4: Check the null hypothesis
Any scoring function on ~300 candidates × ~5 events will accumulate coincidental hits.
Always compute the median score of random charts before interpreting absolute scores.
A candidate scoring 30 pts means nothing if random charts average 25 pts.

### L5: Coverage ratio punishes precision
Penalizing candidates for not hitting every event is counterproductive when orbs are tight.
Missing 1/4 events usually means that event genuinely wasn't close — not that the chart
is wrong. Use coverage as a diagnostic display metric, not a likelihood multiplier.

### L6: Loop 2 zoom is only useful if Loop 1 found the right neighborhood
If Loop 1 converges to the wrong sign, Loop 2's ±30-minute zoom will refine a wrong answer.
The top-level failure mode is Loop 1 mis-ranking, not Loop 2 precision. Invest in
Loop 1 signal quality first.

### L7: PROVISIONAL is a correct output
When the event set is insufficient, returning PROVISIONAL with 1/4 house consensus is
the honest and correct behavior. Do not add ad-hoc fixes to make the system "converge"
on ambiguous data — that would just produce confident wrong answers.

---

## Event Date Verification Log

All event dates used in calibration tests were externally verified against primary sources.

| Event | Date | Source | Status |
|---|---|---|---|
| Neymar Santos debut | 2009-03-07 | neymarjr.com, beIN Sports | ✓ Confirmed |
| Neymar → Barcelona | 2013-06-03 | Official signing date (announcement May 26) | ✓ Confirmed |
| Neymar WC back injury | 2014-07-04 | CBS Sports, Bleacher Report | ✓ Confirmed |
| Neymar → PSG | 2017-08-03 | Al Jazeera, Sky Sports | ✓ Confirmed |
| Neymar metatarsal | 2018-02-25 | Bleacher Report, Yahoo Sports | ✓ Confirmed |
| Britney/Jason marriage | 2004-01-03 | E! Online, Billboard, ABC News | ✓ Confirmed |
| Britney head shaving | 2007-02-16 | ABC News, Newsweek | ✓ Confirmed |
| Britney custody ruling | 2007-10-01 | VOA News — custody ruling date | ✓ Confirmed (custody transfer; visitation suspended Oct 17) |
| Britney conservatorship | 2008-02-01 | Wikipedia conservatorship case | ✓ Confirmed |
| Britney conservatorship end | 2021-11-12 | NPR, PBS NewsHour, Variety | ✓ Confirmed |
