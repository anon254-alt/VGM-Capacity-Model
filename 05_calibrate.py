"""
05_calibrate.py - step 5: calibrate the simulation so aggregate crowding matches the data.

1. continue_shift: one number added to the attractiveness of all rooms relative to EXIT, chosen so
   the simulated people in the 12 collection rooms (per 1,000 visitors, 10:00-17:00) match the measured
   value on CALIBRATION days (normal days, different from the validation days in 06_validate.py).
   This is the standard "calibrate the constant" step when a choice model is used for simulation.
2. exit_share_if_closed for the exhibition wing (q): share of visitors who would have gone to the
   closed wing and leave instead. Calibrated on the wing-closed days 1-12 Feb and 7-10 Sep 2026.
   The closure-test period (18 May - 11 Jun) is never used here.
Output: params/calibration.json
Run from the model folder:  python 05_calibrate.py   (about 6 minutes)
"""
import json, os, sys
import numpy as np
import pandas as pd
from sim import Model, simulate_day
import importlib
val = importlib.import_module("06_validate")

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 8
TEST = ("2026-05-18", "2026-06-11")


def per1000(occ_mean, date, scans):
    return val.window(occ_mean)[:12].sum() * 1000 / scans[date]


def fit(grid, days, key, v, scans, m, **fixed):
    meas = np.mean([per1000(val.measured(v, scans, d), d, scans) for d in days])
    res = {}
    for g in grid:
        res[g] = float(np.mean([per1000(simulate_day(d, REPS, model=m, seed=5, **{key: g}, **fixed)["occ"].mean(0), d, scans)
                                for d in days]))
        print(f"{key}={g}: simulated {res[g]:.1f} vs measured {meas:.1f}", file=sys.stderr)
    xs = np.array(list(res.keys()), dtype=float); ys = np.array(list(res.values()))
    o = np.argsort(ys)
    return float(np.interp(meas, ys[o], xs[o])), meas, res


def main():
    m = Model()
    v = val.load_visits_for_measurement()
    v = v[v.n_visits >= 3]
    scans = m.arr.groupby("date").visitors.sum()
    days = [d for d in sorted(set(v.date)) if d in scans.index and not (TEST[0] <= d <= TEST[1])]
    normal = [d for d in days if m.open.loc[d, "EXH"]]
    cal_days = normal[0::9][:12]                         # validation uses normal[3::7]
    shift, meas1, res1 = fit([0.0, 0.2, 0.4, 0.6], cal_days, "continue_shift", v, scans, m, exit_share={})
    m.continue_shift = shift
    closed_days = [d for d in days if not m.open.loc[d, "EXH"]]
    meas2 = np.mean([per1000(val.measured(v, scans, d), d, scans) for d in closed_days])
    res2 = {}
    for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
        res2[g] = float(np.mean([per1000(simulate_day(d, REPS, model=m, seed=5, exit_share={"EXH": g})["occ"].mean(0), d, scans)
                                 for d in closed_days]))
        print(f"q={g}: simulated {res2[g]:.1f} vs measured {meas2:.1f}", file=sys.stderr)
    xs = np.array(list(res2)); ys = np.array(list(res2.values())); o = np.argsort(ys)
    q = float(np.clip(np.interp(meas2, ys[o], xs[o]), 0, 1))
    out = {"continue_shift": round(shift, 4), "exit_share_if_closed": {"EXH": round(q, 3)},
           "calibration_days_normal": cal_days, "calibration_days_exh_closed": closed_days,
           "measured_collection_per1000_normal": meas1, "simulated_by_shift": res1,
           "measured_collection_per1000_exh_closed": meas2, "simulated_by_q": res2}
    json.dump(out, open(os.path.join(HERE, "params", "calibration.json"), "w"), indent=1)
    print("continue_shift:", round(shift, 3), " q(EXH):", round(q, 3))


if __name__ == "__main__":
    main()
