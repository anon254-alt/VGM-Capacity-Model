"""
capacity.py - "experience-neutral capacity": keep crowding no worse than on today's sold-out days.

Rule (agreed with the team, 12 Nov 2026):
  - VGM's current ticket cap is taken as its acceptable level of crowding ("today = at capacity").
  - Each room's limit = a high percentile (default P90) of its SIMULATED people per 15 minutes,
    10:00-17:00, on sold-out days (>= 5,000 tickets) when the room was open. Simulated, not measured,
    because the model under-estimates peaks (~18%): comparing model with model cancels that bias.
  - Crowding over the limit is measured as PERSON-HOURS over it: for every 15-minute slot, (people - limit)
    when above the limit, x 15 minutes, summed over the day. One person over for an hour = 1 person-hour, so being
    one person over barely counts and being 30 over counts a lot.
  - Margin ("how far from today's level we allow", chosen by the user): the scenario's crowding is measured
    against the limit raised by the margin (limit x (1 + margin)), today's against the plain limit. A scenario is
    acceptable if, in every open room, its person-hours over the raised limit are at most today's person-hours
    over the limit + a floor of 1 person-hour. Margin 0% = no worse than today; 10% = the busiest moments may be up
    to 10% busier than today's busy level. The floor stops a trivial overrun from failing a room. Rooms can't
    offset each other.
  - The solver scales tickets (all slots or a time window) to the highest level that is still acceptable.
Used by app.py (Capacity tab).
"""
import os
import numpy as np
import pandas as pd

from layout import ZONES

HERE = os.path.dirname(os.path.abspath(__file__))
SOLD_OUT = 5000                 # tickets per day
WINDOW = ("10:00", "17:00")
SLOT_MIN = 15


def limits(pct=0.90):
    """Per-room limit (people per 15 min) = pct percentile of simulated occupancy on sold-out days."""
    sim = pd.read_csv(os.path.join(HERE, "data", "simulated_15min.csv.gz"))
    t = pd.read_csv(os.path.join(HERE, "data", "tickets_slot.csv")).groupby("date").tickets.sum()
    op = pd.read_csv(os.path.join(HERE, "data", "open_rooms.csv"))
    open_ = set(zip(op.date[op.open.astype(bool)], op.zone[op.open.astype(bool)]))
    x = sim[sim.date.isin(t[t >= SOLD_OUT].index) & (sim.slot >= WINDOW[0]) & (sim.slot < WINDOW[1])]
    x = x[[(d, z) in open_ for d, z in zip(x.date, x.zone)]]
    lim = x.groupby("zone").simulated.quantile(pct).reindex(ZONES)
    days = x.date.nunique()
    return lim, days


FLOOR = 1.0                     # person-hours: always allowed (1 person over the limit for 1 hour)


def excess(occ, lim, margin=0.0):
    """occ: (slots, 13) mean people per 15-min slot -> person-hours over limit x (1 + margin), per room (13,)."""
    return (np.clip(occ - lim.to_numpy()[None, :] * (1 + margin), 0, None) * SLOT_MIN / 60).sum(0)


def allowance(base_ex):
    """Person-hours over the (raised) limit each room may have: today's over the limit + FLOOR."""
    return base_ex + FLOOR


def minutes_above(occ, lim):
    """occ: (slots, 13) mean people per 15-min slot -> minutes above the limit per room (array of 13)."""
    return (occ > lim.to_numpy()[None, :]).sum(0) * SLOT_MIN


def acceptable(scen_ex, base_ex, open_mask):
    """scen_ex: scenario person-hours over the raised limit; base_ex: today's over the limit.
    True if every open room stays within today's amount + FLOOR."""
    return bool(np.all((scen_ex <= allowance(base_ex)) | ~open_mask))


def scale_tickets(tickets, f, window=None):
    """Multiply tickets by f, in all slots or only in slots window=(first, last) (inclusive, 'HH:MM')."""
    t = tickets.copy()
    t["tickets"] = t["tickets"].astype(float)
    sel = np.ones(len(t), bool) if window is None else ((t.slot >= window[0]) & (t.slot <= window[1])).to_numpy()
    t.loc[sel, "tickets"] = t.loc[sel, "tickets"] * f
    return t


def solve(run, base_min, open_mask, lo=0.5, hi=1.3, steps=6):
    """Highest ticket factor f in [lo, hi] for which the scenario stays acceptable.
    run(f) -> scenario person-hours over the raised limit per room (13,). Returns (f, person-hours at f, trace)."""
    trace = []
    m1 = run(1.0); ok1 = acceptable(m1, base_min, open_mask); trace.append((1.0, ok1))
    if ok1:
        a, b, best, best_m = 1.0, hi, 1.0, m1                     # headroom: search upwards
        mh = run(hi); okh = acceptable(mh, base_min, open_mask); trace.append((hi, okh))
        if okh:
            return hi, mh, trace
    else:
        a, b, best, best_m = lo, 1.0, None, None                  # over the limit: search downwards
        ml = run(lo); okl = acceptable(ml, base_min, open_mask); trace.append((lo, okl))
        if not okl:
            return None, ml, trace
        best, best_m = lo, ml
    for _ in range(steps):
        mid = round((a + b) / 2, 4)
        mm = run(mid); ok = acceptable(mm, base_min, open_mask); trace.append((mid, ok))
        if ok:
            a, best, best_m = mid, mid, mm
        else:
            b = mid
    return best, best_m, trace
