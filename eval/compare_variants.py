"""
Compare scoring variants on synthetic charts with planted solar-arc events.

    python -m eval.compare_variants CHARTS GRID_MINUTES [VARIANTS] [EVENTS] [DATE_ERROR_YEARS]
    python -m eval.compare_variants 20 2 classic,sa_only,confirm_half 5 0.5

Two kinds of planted list are reported: "sa" (a random choice among the dates when a
solar arc contact with an angle is exact for the true chart) and "stacked" (the dates,
among those, where transits and progressions also touch an angle most closely).
Prints, per variant, how often the best-scoring time is within 10 minutes of the truth.
"""
import sys, time; sys.argv.insert(1, "."); N=int(sys.argv[2]); GRID=int(sys.argv[3])
from datetime import date, timedelta
import numpy as np
from src.ephemeris import birth_to_jd, generate_candidate_grid, solar_arc_for_age
from src.models import EventType, EventWeight, LifeEvent
from src.rectifier import build_scorers, score_candidate
from src.scoring.angle_first import AngleFirstScorer, AngleFirstConfig as C
TYPES=[EventType.MARRIAGE,EventType.CAREER_PEAK,EventType.RELOCATION,EventType.PUBLICATION,EventType.BIRTH_OF_CHILD,EventType.OTHER]  # all non-painful: weights equal
def sa_dates(chart,bd):
    """all dates (ages 6-70) where a solar-arc planet/angle contact is exact for the true chart"""
    out=[]; ang=(chart.ascendant,chart.mc,chart.dsc,chart.ic)
    for p in chart.planets.values():
        for a in ang:
            for arc in ((a-p)%360,(p-a)%360):
                if 6<arc<70:
                    age=arc
                    for _ in range(4): age+=(arc-solar_arc_for_age(chart.julian_day,age))/0.9856
                    out.append(age)
    return sorted(out)
def mk(bd,ages):
    ev=[];last=-9
    for age in ages:
        if abs(age-last)<1.2: continue
        last=age; ev.append(LifeEvent(description=f"p{len(ev)}",event_type=TYPES[len(ev)%len(TYPES)],date=bd+timedelta(days=round(age*365.25)),weight=EventWeight.ANCHOR))
    return ev
def plants(bd,lat,lon,tz,true,rng,kind,n=10):
    chart=generate_candidate_grid(bd,lat,lon,tz,interval_minutes=1)[true]
    ages=sa_dates(chart,bd); jd=birth_to_jd(bd,720,tz)
    if kind=="sa":
        pick=sorted(rng.choice(ages,size=min(len(ages),3*n),replace=False))
        return mk(bd,pick)[:n]
    sc=AngleFirstScorer(); scored=[]
    for e in mk(bd,ages):
        sa,tr,pr=sc.components(chart,e,jd); scored.append((tr[0]+pr[0],e))
    scored.sort(key=lambda x:-x[0])
    return sorted([e for _,e in scored[:n]],key=lambda e:e.date)
def evalv(scorers,bd,lat,lon,tz,ev,true):
    jd=birth_to_jd(bd,720,tz); cs=generate_candidate_grid(bd,lat,lon,tz,interval_minutes=GRID)
    S=np.array([score_candidate(c,ev,jd,scorers).training_score() for c in cs]); T=np.array([c.time_minutes for c in cs])
    i=int(np.argmin(np.abs(T-true))); best=T[int(np.argmax(S))]; err=min(abs(best-true),1440-abs(best-true))
    return (int((S>S[i]).sum())+1)/len(cs), err
V={"classic":lambda:build_scorers(),
   "sa_only":lambda:[AngleFirstScorer(config=C(mode="sa_only"))],
   "additive":lambda:[AngleFirstScorer(config=C())],
   "add_half":lambda:[AngleFirstScorer(config=C(transit_weight=.5,progression_weight=.5))],
   "add_stack":lambda:[AngleFirstScorer(config=C(stack_bonus=.5))],
   "confirm_half":lambda:[AngleFirstScorer(config=C(mode="confirm",transit_weight=.5,progression_weight=.5))],
   "confirm_full":lambda:[AngleFirstScorer(config=C(mode="confirm"))],
   "strict":lambda:[AngleFirstScorer(config=C(mode="strict"))]}
only=sys.argv[4].split(",") if len(sys.argv)>4 else list(V)
NEV=int(sys.argv[5]) if len(sys.argv)>5 else 10
JIT=float(sys.argv[6]) if len(sys.argv)>6 else 0.0
for kind in ("sa","stacked"):
    rng=np.random.default_rng(5); res={k:[] for k in only}; ne=[]
    t0=time.time()
    for k in range(N):
        bd=date(int(rng.integers(1940,1985)),int(rng.integers(1,13)),int(rng.integers(1,28)))
        lat,lon=float(rng.uniform(-40,55)),float(rng.uniform(-100,100)); tz=float(round(lon/15)); true=int(rng.integers(0,1440))
        ev=plants(bd,lat,lon,tz,true,rng,kind,NEV)
        if JIT: ev=[e.model_copy(update={'date':e.date+timedelta(days=int(rng.normal(0,JIT*365.25)))}) for e in ev]
        ne.append(len(ev))
        for name in only: res[name].append(evalv(V[name](),bd,lat,lon,tz,ev,true))
    print(f"--- plants: {kind} jitter {JIT}y (n={N}, grid {GRID} min, events/chart {np.mean(ne):.1f}, {time.time()-t0:.0f}s)")
    for name,v in res.items():
        fr=np.array([x[0] for x in v]); er=np.array([x[1] for x in v])
        print(f"{name:10s} winner within 10 min: {(er<=10).sum():2d}/{N} | true time top 1%: {(fr<=0.01).sum():2d}/{N} | median percentile {np.median(fr):.1%} | median winner error {np.median(er):.0f} min")
