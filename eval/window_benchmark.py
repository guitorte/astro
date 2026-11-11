"""
Benchmark for src.window.compare_window on synthetic charts.

    python -m eval.window_benchmark --cases 20 --events 10
    python -m eval.window_benchmark --cases 20 --events 5 --jitter 0.5

Each synthetic native has a true birth minute and a "recorded" time up to 20 minutes
away from it. Planted events are a random choice among the dates when a solar arc
contact with an angle is exact for the true chart (optionally with a random error
in the date); random events are unrelated to the chart. Reported:

  planted  how often the window test passes (p <= alpha), and how often the best
           minute is within 5 minutes of the truth (overall, and when it passes)
  random   how often the window test passes although there is nothing to find

As with eval.benchmark, this shows whether the engine can find the signal it
assumes, not whether real lives follow that signal.
"""

import argparse
from datetime import date, timedelta

import numpy as np

from src.ephemeris import generate_candidate_grid, solar_arc_for_age
from src.models import BirthData, EventType, EventWeight, LifeEvent
from src.window import compare_window

TYPES = [EventType.MARRIAGE, EventType.CAREER_PEAK, EventType.RELOCATION,
         EventType.PUBLICATION, EventType.BIRTH_OF_CHILD, EventType.OTHER]


def planted_events(birth: BirthData, true_minute: int, n: int, rng) -> list[LifeEvent]:
    chart = generate_candidate_grid(
        birth.birth_date, birth.latitude, birth.longitude, birth.timezone_offset,
        interval_minutes=1, start_minute=true_minute, end_minute=true_minute)[0]
    ages = []
    for p in chart.planets.values():
        for a in (chart.ascendant, chart.mc, chart.dsc, chart.ic):
            for arc in ((a - p) % 360, (p - a) % 360):
                if 6 < arc < 70:
                    age = arc
                    for _ in range(4):
                        age += (arc - solar_arc_for_age(chart.julian_day, age)) / 0.9856
                    ages.append(age)
    events, last = [], -9.0
    for age in sorted(rng.choice(ages, size=min(len(ages), 3 * n), replace=False)):
        if abs(age - last) < 1.2:
            continue
        last = age
        events.append(LifeEvent(
            description=f"planted {len(events)}", event_type=TYPES[len(events) % len(TYPES)],
            date=birth.birth_date + timedelta(days=round(age * 365.25)), weight=EventWeight.ANCHOR))
        if len(events) == n:
            break
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=0.0, help="std. dev. in years of error in planted dates")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    passed = close = close_and_passed = false_pos = 0
    for k in range(args.cases):
        bd = date(int(rng.integers(1940, 1985)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        lat, lon = float(rng.uniform(-40, 55)), float(rng.uniform(-100, 100))
        birth = BirthData(name=f"synthetic {k}", birth_date=bd, birth_city="synthetic",
                          latitude=lat, longitude=lon, timezone_offset=float(round(lon / 15)))
        true_minute = int(rng.integers(60, 1380))
        recorded = true_minute + int(rng.integers(-20, 21))
        events = planted_events(birth, true_minute, args.events, rng)
        if args.jitter:
            events = [e.model_copy(update={"date": e.date + timedelta(days=int(rng.normal(0, args.jitter * 365.25)))})
                      for e in events]
        rnd = [LifeEvent(description=f"random {j}", event_type=TYPES[j % len(TYPES)], weight=EventWeight.ANCHOR,
                         date=bd + timedelta(days=int(rng.uniform(8, 65) * 365.25))) for j in range(args.events)]

        rep = compare_window(birth, events, recorded, window_minutes=args.window)
        miss = abs(recorded + rep.minutes_from_recorded - true_minute)
        ok = rep.verdict != "no_preference"
        passed += ok
        close += miss <= 5
        close_and_passed += ok and miss <= 5
        fp = compare_window(birth, rnd, recorded, window_minutes=args.window).verdict != "no_preference"
        false_pos += fp
        print(f"case {k:2d}: planted p={rep.chance_p_value:.3f} {rep.verdict:17s} best minute off by {miss:2d} | "
              f"random: {'PASSED (false positive)' if fp else 'no preference'}", flush=True)

    n = args.cases
    print()
    print(f"{args.events} events, date error {args.jitter} y, window ±{args.window} min, {n} charts")
    print(f"planted: window test passed                 {passed}/{n}")
    print(f"planted: best minute within 5 min of truth  {close}/{n}  (when the test passed: {close_and_passed}/{passed})")
    print(f"random:  window test passed (false positive) {false_pos}/{n}")


if __name__ == "__main__":
    main()
