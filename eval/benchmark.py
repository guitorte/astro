"""
Benchmark for the scoring engine: can it find a signal we planted, and does it
stay quiet when there is none?

    python -m eval.benchmark --cases 12 --natives 30

For each synthetic native (random date, place and birth minute) it builds two
event lists:

  planted  up to 8 events dated exactly when a solar-arc-directed natal planet
           reaches a natal angle of the true chart — the cleanest timing signal
           the engine's own techniques can describe.
  random   8 events at random ages 8–65, unrelated to the chart.

and reports, using chance_level_test (the real birth date against unrelated
birth dates at the same place):

  false-positive rate   share of random lists with p <= alpha  (should be ~alpha)
  power                 share of planted lists with p <= alpha (should be high)
  recovery              rank of the true time on the candidate grid for the
                        planted lists (1 = found)

Any change to the scoring should be judged here before it is judged on a
celebrity chart: two calibration charts cannot separate a better model from a
lucky one.
"""

import argparse
from datetime import date, timedelta

import numpy as np

from src.ephemeris import birth_to_jd, generate_candidate_grid, solar_arc_for_age
from src.models import BirthData, EventType, EventWeight, LifeEvent
from src.rectifier import CHANCE_ALPHA, build_scorers, chance_level_test, score_candidate

TYPES = [
    EventType.MARRIAGE, EventType.CAREER_PEAK, EventType.DEATH_OF_PARENT,
    EventType.ACCIDENT, EventType.BIRTH_OF_CHILD, EventType.RELOCATION,
    EventType.DIVORCE, EventType.ILLNESS,
]


def planted_events(bd, lat, lon, tz, true_minute, n=8):
    chart = generate_candidate_grid(bd, lat, lon, tz, interval_minutes=1)[true_minute]
    angles = (chart.ascendant, chart.mc, chart.dsc, chart.ic)
    arcs = sorted(
        (a - p) % 360 for p in chart.planets.values() for a in angles
        if 5 < (a - p) % 360 < 70
    )
    events, last_age = [], -9.0
    for arc in arcs:
        age = arc
        for _ in range(4):  # solve true solar arc(age) == arc
            age += (arc - solar_arc_for_age(chart.julian_day, age)) / 0.9856
        if age - last_age < 1.0:
            continue
        last_age = age
        events.append(LifeEvent(
            description=f"planted {len(events)}", event_type=TYPES[len(events) % 8],
            date=bd + timedelta(days=round(age * 365.25)), weight=EventWeight.ANCHOR,
        ))
        if len(events) == n:
            break
    return events


def random_events(bd, rng, n=8):
    return [
        LifeEvent(description=f"random {j}", event_type=TYPES[j % 8], weight=EventWeight.ANCHOR,
                  date=bd + timedelta(days=int(rng.uniform(8, 65) * 365.25)))
        for j in range(n)
    ]


def true_time_rank(birth, events, true_minute, interval):
    jd = birth_to_jd(birth.birth_date, 720, birth.timezone_offset)
    cands = generate_candidate_grid(birth.birth_date, birth.latitude, birth.longitude,
                                    birth.timezone_offset, interval_minutes=interval)
    scorers = build_scorers()
    scores = np.array([score_candidate(c, events, jd, scorers).training_score() for c in cands])
    times = np.array([c.time_minutes for c in cands])
    i = int(np.argmin(np.abs(times - true_minute)))
    return int((scores > scores[i]).sum()) + 1, len(cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--natives", type=int, default=30, help="unrelated natives per test")
    ap.add_argument("--interval", type=int, default=15, help="candidate grid, minutes")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    null_p, alt_p, ranks = [], [], []
    for k in range(args.cases):
        bd = date(int(rng.integers(1940, 1985)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        lat, lon = float(rng.uniform(-40, 55)), float(rng.uniform(-100, 100))
        birth = BirthData(name=f"synthetic {k}", birth_date=bd, birth_city="synthetic",
                          latitude=lat, longitude=lon, timezone_offset=float(round(lon / 15)))
        true_minute = int(rng.integers(0, 1440))
        planted = planted_events(bd, lat, lon, birth.timezone_offset, true_minute)
        null_p.append(chance_level_test(birth, random_events(bd, rng), interval_minutes=args.interval,
                                        n_natives=args.natives, seed=k)[0])
        alt_p.append(chance_level_test(birth, planted, interval_minutes=args.interval,
                                       n_natives=args.natives, seed=k)[0])
        ranks.append(true_time_rank(birth, planted, true_minute, args.interval))
        print(f"case {k:2d}: random p={null_p[-1]:.3f}  planted p={alt_p[-1]:.3f}  "
              f"true-time rank {ranks[-1][0]}/{ranks[-1][1]}  ({len(planted)} planted events)")

    null_p, alt_p = np.array(null_p), np.array(alt_p)
    print()
    print(f"false-positive rate (random lists, p<={CHANCE_ALPHA}): {(null_p <= CHANCE_ALPHA).mean():.0%}")
    print(f"power (planted lists, p<={CHANCE_ALPHA}):             {(alt_p <= CHANCE_ALPHA).mean():.0%}")
    print(f"true time ranked first: {sum(r == 1 for r, _ in ranks)}/{len(ranks)}; "
          f"median rank {int(np.median([r for r, _ in ranks]))}/{ranks[0][1]}")


if __name__ == "__main__":
    main()
