"""
run_scenario.py - compare a baseline day with a scenario (room closures, more/fewer visitors).

Examples (from the model folder):
  python run_scenario.py --date 2026-08-14 --close R06
  python run_scenario.py --date 2026-07-10 --close R05 R07 --reps 200
  python run_scenario.py --date 2026-08-14 --arrivals-factor 1.10          # 10% more visitors
  python run_scenario.py --date 2026-08-14 --close EXH --capacity 150      # flag slots above 150 people
  python run_scenario.py --date 2026-08-14 --tickets-factor 1.10           # sell 10% more tickets in every slot
  python run_scenario.py --date 2026-08-14 --tickets-per-slot 180          # sell 180 tickets in every slot 09:00-16:30
  python run_scenario.py --date 2026-12-04 --tickets-file my_slots.csv     # any date; csv with columns slot,tickets

Ticket options turn tickets sold per 15-min slot into people entering (arrivals.py: show-up rate and
arrival times vs the booked slot). Without them the scenario uses that day's real entries.

Baseline = the date as it was (rooms open that day, real arrivals). Both runs use the same random draws
(common random numbers). Output in output/scenarios/<name>/:
  occupancy.csv : slot x room - baseline/scenario mean, 10th and 90th percentile
  summary.csv   : per room - daytime average, peak, change, share of runs above --capacity
"""
import argparse, os
import numpy as np
import pandas as pd

from layout import ZONES, ROOM_LABEL
from sim import Model, simulate_day
import arrivals as A

HERE = os.path.dirname(os.path.abspath(__file__))


def day_tickets(date):
    t = pd.read_csv(os.path.join(HERE, "data", "tickets_slot.csv"))
    return t[t.date == date][["slot", "tickets", "channel"]]


def scenario_tickets(date, tickets_factor=None, tickets_per_slot=None, tickets_file=None):
    """Tickets per booked slot for the scenario, or None (= use the day's real entries)."""
    if tickets_file:
        return pd.read_csv(tickets_file)[["slot", "tickets"]]
    if tickets_per_slot is not None:
        return A.flat_tickets(tickets_per_slot)
    if tickets_factor is not None:
        t = day_tickets(date)
        if t.empty:
            raise ValueError(f"no ticket sales for {date}; use --tickets-per-slot or --tickets-file")
        return t.assign(tickets=t.tickets * tickets_factor)
    return None


def run(date, close=(), arrivals_factor=1.0, reps=100, capacity=None, seed=0, model=None, tickets=None):
    m = model or Model()
    scen_arr = A.to_counts(A.expected(tickets)) if tickets is not None else None
    has_real = not m.arr[m.arr.date == date].empty
    base = simulate_day(date, reps, model=m, seed=seed, arrivals=None if has_real else scen_arr)
    scen = simulate_day(date, reps, closed=close, arrivals_factor=arrivals_factor, model=m, seed=seed,
                        arrivals=scen_arr)
    rows = []
    for name, r in (("baseline", base), ("scenario", scen)):
        o = r["occ"]
        for si, s in enumerate(r["slots"]):
            for zi, z in enumerate(ZONES):
                x = o[:, si, zi]
                rows.append((name, s, z, x.mean(), np.percentile(x, 10), np.percentile(x, 90)))
    occ = pd.DataFrame(rows, columns=["run", "slot", "zone", "mean", "p10", "p90"])
    day = occ[(occ.slot >= "10:00") & (occ.slot < "17:00")]
    summ = day.groupby(["zone", "run"])["mean"].mean().unstack()
    peak = occ.groupby(["zone", "run"])["mean"].max().unstack()
    summary = pd.DataFrame({
        "room": [ROOM_LABEL[z] for z in ZONES],
        "baseline_avg": summ.reindex(ZONES).baseline, "scenario_avg": summ.reindex(ZONES).scenario,
        "baseline_peak": peak.reindex(ZONES).baseline, "scenario_peak": peak.reindex(ZONES).scenario,
    }, index=ZONES)
    summary["change_avg_pct"] = 100 * (summary.scenario_avg / summary.baseline_avg - 1)
    if capacity:
        for name, r in (("baseline", base), ("scenario", scen)):
            summary[f"{name}_share_runs_above_{capacity}"] = (r["occ"].max(1) > capacity).mean(0)
    return occ, summary, base, scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--close", nargs="*", default=[], help="zones to close, e.g. R06 EXH")
    ap.add_argument("--arrivals-factor", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--capacity", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tickets-factor", type=float, default=None, help="scale the day's real ticket sales")
    ap.add_argument("--tickets-per-slot", type=float, default=None, help="same tickets in every slot 09:00-16:30")
    ap.add_argument("--tickets-file", default=None, help="csv with columns slot (HH:MM), tickets")
    a = ap.parse_args()
    tickets = scenario_tickets(a.date, a.tickets_factor, a.tickets_per_slot, a.tickets_file)
    occ, summary, base, scen = run(a.date, a.close, a.arrivals_factor, a.reps, a.capacity, a.seed, tickets=tickets)
    tag = ("" if tickets is None else f"_tickets{int(tickets.tickets.sum())}")
    name = f"{a.date}_close-{'-'.join(a.close) or 'none'}_x{a.arrivals_factor:g}{tag}"
    out = os.path.join(HERE, "output", "scenarios", name)
    os.makedirs(out, exist_ok=True)
    occ.round(2).to_csv(os.path.join(out, "occupancy.csv"), index=False)
    summary.round(2).to_csv(os.path.join(out, "summary.csv"))
    print(f"visitors {base['n_visitors']:,} -> {scen['n_visitors']:,}; mean visit {base['mean_visit_min']:.0f} -> {scen['mean_visit_min']:.0f} min")
    print(summary.round(1).to_string())
    print("saved to", out)


if __name__ == "__main__":
    main()
