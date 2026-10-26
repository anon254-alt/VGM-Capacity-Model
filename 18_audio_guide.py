"""
18_audio_guide.py - step 18: audio guide vs no audio guide in the simulation.

1. data/ag_share_hour.csv: share of visitors with an audio guide per day and arrival hour
   = audio-guide sessions (3+ rooms) starting in that hour / scanned visitors in that hour
   (from the crowding weights, 08_build_crowding.py). The same share the crowding measure uses.
2. stay_multiplier_non_ag (params/audio_guide.json): the stay model is estimated on audio-guide users; visitors
   without a guide stay this multiple of the predicted time in each room. Calibrated so that in the simulation
   their whole visit is non_ag_time_ratio x as long as an audio-guide visitor's (0.88 from the visit-duration
   file, 17_visit_duration_nps.py). Shorter stays also mean more rooms before the exit pull, so the
   multiplier per room ends up below the ratio for the whole visit.
Run from the model folder:  python 18_audio_guide.py   (~4 minutes). Then re-run 05_calibrate.py.
"""
import glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, PAR = os.path.join(HERE, "data"), os.path.join(HERE, "params")
DAYS = ["2026-03-11", "2026-04-15", "2026-06-24", "2026-07-22", "2026-08-12", "2026-09-02"]
REPS = 10


def build_share():
    w = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, "crowding", "weights_*.csv"))))
    w = w[(w.scans >= 20) & (w.sessions >= 5)]
    out = pd.DataFrame({"date": w.date, "hour": w.start_hour, "share": (w.sessions / w.scans).clip(0, 1).round(4)})
    out.to_csv(os.path.join(DATA, "ag_share_hour.csv"), index=False)
    print("audio-guide share by hour (median over days):", out.groupby("hour").share.median().round(2).to_dict())
    return out


def ratio_for(mult):
    from sim import Model, simulate_day
    m = Model()
    m.stay_mult_non_ag = mult
    # same 50% share in every hour, so guide and non-guide visitors arrive at the same times
    # (the 0.88 from the visit-duration file is adjusted for entry time)
    m.ag_share = pd.Series(dtype=float)
    m.ag_share_typ = pd.Series(0.5, index=range(24))
    ra, rn = [], []
    for d in DAYS:
        r = simulate_day(d, REPS, model=m, seed=3)
        ra.append(r["mean_visit_ag"]); rn.append(r["mean_visit_no_ag"])
    return float(np.mean(rn) / np.mean(ra))


def main():
    build_share()
    pj = os.path.join(PAR, "audio_guide.json")
    par = json.load(open(pj))
    target = par["non_ag_time_ratio"]
    lo, hi = 0.6, 1.0
    r_lo, r_hi = ratio_for(lo), ratio_for(hi)
    print(f"multiplier {lo}: visit ratio {r_lo:.3f}; {hi}: {r_hi:.3f}; target {target}")
    x = lo + (target - r_lo) * (hi - lo) / (r_hi - r_lo)
    for _ in range(2):                                   # two secant refinements
        r_x = ratio_for(x)
        print(f"multiplier {x:.3f}: visit ratio {r_x:.3f}")
        if abs(r_x - target) < 0.003:
            break
        if r_x < target:
            lo, r_lo = x, r_x
        else:
            hi, r_hi = x, r_x
        x = lo + (target - r_lo) * (hi - lo) / (r_hi - r_lo)
    par.update({"stay_multiplier_non_ag": round(float(x), 4),
                "source_stay_multiplier": f"18_audio_guide.py: simulated non-guide visit = {target} x guide visit"})
    json.dump(par, open(pj, "w"), indent=1)
    print(json.dumps(par, indent=1))


if __name__ == "__main__":
    main()
