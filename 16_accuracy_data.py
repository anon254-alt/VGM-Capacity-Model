"""
16_accuracy_data.py - step 16: data for the app's "Accuracy vs history" page.

For every day with data (Feb-Sep 2026):
  measured  : people per room per 15 min = average of the three 5-min slots of the official crowding
              measure (08_build_crowding.py: all audio-guide sessions, scaled by start hour)
  simulated : the model run on that day as it happened (real entries, rooms open/closed as they were,
              that day's visitor mix and closing time), 20 runs, average
Outputs (small enough to deploy with the app):
  data/measured_15min.csv.gz   : date, slot, zone, measured
  data/simulated_15min.csv.gz  : date, slot, zone, simulated
  data/measured_days.csv       : date, sessions, scans, factor, reliable  (days with very few audio-guide
                                 sessions vs visitors are flagged: the measurement is unreliable there)
Re-run whenever the model parameters change.
Run from the model folder:  python 16_accuracy_data.py [--workers 2]   (~15 s per day per worker)
"""
import argparse, glob, os
from multiprocessing import Pool
import numpy as np
import pandas as pd

from layout import ZONES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPS = 20


def measured():
    c = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, "crowding", "crowding_*.csv"))))
    c["slot15"] = c.slot.str[:3] + (c.slot.str[3:5].astype(int) // 15 * 15).map("{:02d}".format)
    # 5-min slots with nobody are not stored -> divide the sum by 3 instead of taking the mean
    m = (c.groupby(["date", "slot15", "zone"]).est_people.sum() / 3).round(2).rename("measured").reset_index()
    return m.rename(columns={"slot15": "slot"})


def measured_days():
    w = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, "crowding", "weights_*.csv"))))
    g = w.groupby("date")[["sessions", "scans"]].sum()
    g["factor"] = (g.scans / g.sessions).round(3)
    # unreliable: fewer than 300 sessions, or scale-up factor > 1.5x the typical one (audio-guide data gaps)
    g["reliable"] = (g.sessions >= 300) & (g.factor <= 1.5 * g.factor.median())
    g.reset_index().to_csv(os.path.join(DATA, "measured_days.csv"), index=False)
    return g


def sim_day(date):
    from sim import Model, simulate_day
    global _M
    if "_M" not in globals():
        _M = Model()
    r = simulate_day(date, REPS, model=_M, seed=0)
    o = r["occ"].mean(0)
    rows = [(date, s, z, round(float(o[i, zi]), 2)) for i, s in enumerate(r["slots"]) for zi, z in enumerate(ZONES)]
    return pd.DataFrame(rows, columns=["date", "slot", "zone", "simulated"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--days-only", action="store_true", help="only rebuild the measured files, no simulation")
    a = ap.parse_args()
    meas = measured()
    meas.to_csv(os.path.join(DATA, "measured_15min.csv.gz"), index=False)
    measured_days()
    if a.days_only:
        return
    arr_days = set(pd.read_csv(os.path.join(DATA, "arrivals_5min.csv"), usecols=["date"]).date)
    days = sorted(set(meas.date) & arr_days)
    print(len(days), "days", flush=True)
    out = []
    with Pool(a.workers) as p:
        for i, df in enumerate(p.imap(sim_day, days)):
            out.append(df)
            if i % 20 == 0:
                print(i, days[i], flush=True)
    sim = pd.concat(out)
    sim = sim[(sim.slot >= "08:30") & (sim.slot <= "21:45")]
    sim.to_csv(os.path.join(DATA, "simulated_15min.csv.gz"), index=False)
    print("done", len(sim), "rows", flush=True)


if __name__ == "__main__":
    main()
