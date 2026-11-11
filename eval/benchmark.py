"""
Benchmark for the scoring engine: can it find a signal we planted, and does it
stay quiet when there is none?

    python -m eval.benchmark --cases 20 --events 10
    python -m eval.benchmark --cases 20 --events 5 --jitter 0.5 --skip-random
    python -m eval.benchmark --method classic

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
  recovery              how far the best-scoring time is from the true time
                        for the planted lists

The true birth minute is random, so it usually falls between the points of the
candidate grid; with the classic 15-minute grid that alone hides much of the
signal (see the README). The planted events exercise solar arcs only, so this is a floor for
the engine, not a model of a real life.

Any change to the scoring should be judged here before it is judged on a
celebrity chart: two calibration charts cannot separate a better model from a
lucky one.
"""

import argparse
from datetime import date, timedelta

import numpy as np

from src.ephemeris import birth_to_jd, generate_candidate_grid, solar_arc_for_age
from src.models import BirthData, EventType, EventWeight, LifeEvent
from src.rectifier import (
    CHANCE_ALPHA, DEFAULT_METHOD, LOOP1_INTERVAL, build_scorers, chance_level_test, score_candidate,
)

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


def true_time_rank(birth, events, true_minute, interval, method):
    """(rank of the true time, grid size, distance in minutes from the winner to the truth)."""
    jd = birth_to_jd(birth.birth_date, 720, birth.timezone_offset)
    cands = generate_candidate_grid(birth.birth_date, birth.latitude, birth.longitude,
                                    birth.timezone_offset, interval_minutes=interval)
    scorers = build_scorers(method=method)
    scores = np.array([score_candidate(c, events, jd, scorers).training_score() for c in cands])
    times = np.array([c.time_minutes for c in cands])
    i = int(np.argmin(np.abs(times - true_minute)))
    miss = abs(int(times[int(np.argmax(scores))]) - true_minute)
    return int((scores > scores[i]).sum()) + 1, len(cands), min(miss, 1440 - miss)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--natives", type=int, default=30, help="unrelated natives per test")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=sorted(LOOP1_INTERVAL))
    ap.add_argument("--interval", type=int, default=None, help="candidate grid, minutes (default: the method's)")
    ap.add_argument("--events", type=int, default=8, help="planted events per chart")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="std. dev. in years of random error added to each planted date")
    ap.add_argument("--skip-random", action="store_true", help="skip the false-positive runs")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    interval = args.interval or LOOP1_INTERVAL[args.method]

    rng = np.random.default_rng(args.seed)
    null_p, alt_p, ranks = [], [], []
    for k in range(args.cases):
        bd = date(int(rng.integers(1940, 1985)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        lat, lon = float(rng.uniform(-40, 55)), float(rng.uniform(-100, 100))
        birth = BirthData(name=f"synthetic {k}", birth_date=bd, birth_city="synthetic",
                          latitude=lat, longitude=lon, timezone_offset=float(round(lon / 15)))
        true_minute = int(rng.integers(0, 1440))
        planted = planted_events(bd, lat, lon, birth.timezone_offset, true_minute, n=args.events)
        if args.jitter:
            planted = [e.model_copy(update={"date": e.date + timedelta(days=int(rng.normal(0, args.jitter * 365.25)))})
                       for e in planted]
        rnd = random_events(bd, rng, n=args.events)
        test = dict(interval_minutes=interval, n_natives=args.natives, seed=k, method=args.method)
        if not args.skip_random:
            null_p.append(chance_level_test(birth, rnd, **test)[0])
        alt_p.append(chance_level_test(birth, planted, **test)[0])
        ranks.append(true_time_rank(birth, planted, true_minute, interval, args.method))
        r = ranks[-1]
        print(f"case {k:2d}: " + (f"random p={null_p[-1]:.3f}  " if null_p else "")
              + f"planted p={alt_p[-1]:.3f}  true-time rank {r[0]}/{r[1]}  winner off by {r[2]} min", flush=True)

    alt_p = np.array(alt_p)
    n = len(ranks)
    print()
    print(f"method {args.method}, grid {interval} min, {args.events} events, date jitter {args.jitter} y")
    if null_p:
        print(f"false positives (random lists, p<={CHANCE_ALPHA}): {int((np.array(null_p) <= CHANCE_ALPHA).sum())}/{n}")
    print(f"power (planted lists, p<={CHANCE_ALPHA}):           {int((alt_p <= CHANCE_ALPHA).sum())}/{n}")
    print(f"winner within 10 min of the true time:        {sum(r[2] <= 10 for r in ranks)}/{n}")


if __name__ == "__main__":
    main()
