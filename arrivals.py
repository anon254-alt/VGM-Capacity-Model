"""
arrivals.py - turns tickets sold per 15-minute slot into people entering per 5 minutes.

    arrivals(5-min bin) = sum over booked slots of
                          tickets x show-up rate (by channel) x P(arrive k minutes from slot start | slot hour)

Parameters (params/arrivals.json, fitted by 12_arrivals_model.py):
  show_up   : share of non-cancelled tickets that are scanned, by channel
  lag_pmf   : distribution of (scan time - booked slot start) in 5-minute steps, per slot hour
  open_min  : doors open (09:00); people arriving earlier queue and enter evenly over 09:00-09:15
Used by run_scenario.py (--tickets-per-slot, --tickets-factor, --tickets-file).
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAR = os.path.join(HERE, "params", "arrivals.json")
STEP = 5


def load():
    return json.load(open(PAR))


def _minutes(slot):
    s = pd.Series(slot).astype(str)
    return s.str[:2].astype(int).to_numpy() * 60 + s.str[3:5].astype(int).to_numpy()


def expected(tickets, params=None):
    """tickets: DataFrame with columns slot ('HH:MM' booked slot), tickets, and optionally channel.
    Returns DataFrame slot ('HH:MM', 5-min bins), visitors (expected, float)."""
    p = params or load()
    t = tickets.copy()
    if "channel" not in t:
        t["channel"] = "all"
    show = t.channel.map(p["show_up"]).fillna(p["show_up"]["all"]).to_numpy()
    sm = _minutes(t.slot)
    out = {}
    for m0, n in zip(sm, t.tickets.to_numpy() * show):
        h = str(min(max(m0 // 60, 9), 20))
        pmf = p["lag_pmf"].get(h, p["lag_pmf"]["all"])
        for lag, q in zip(p["lag_bins"], pmf):
            b = (m0 + lag) // STEP * STEP
            out[b] = out.get(b, 0.0) + n * q
    # doors open at 09:00: people who come early queue and enter in the first 15 minutes
    open_min = p.get("open_min", 540)
    early = sum(v for k, v in out.items() if k < open_min)
    out = {k: v for k, v in out.items() if k >= open_min}
    for k in range(open_min, open_min + 15, STEP):
        out[k] = out.get(k, 0.0) + early / 3
    b = np.array(sorted(out))
    b = b[(b >= 8 * 60) & (b < 23 * 60)]
    return pd.DataFrame({"slot": [f"{x // 60:02d}:{x % 60:02d}" for x in b], "visitors": [out[x] for x in b]})


def to_counts(exp):
    """Whole visitors per 5-min bin, keeping the day total (largest remainder rounding)."""
    v = exp.visitors.to_numpy()
    base = np.floor(v)
    k = int(round(v.sum() - base.sum()))
    if k > 0:
        base[np.argsort(-(v - base))[:k]] += 1
    return exp.assign(visitors=base.astype(int))


def flat_tickets(per_slot, first="09:00", last="16:30", extra=None):
    """Same number of tickets in every 15-min slot from first to last (inclusive). extra: dict slot->tickets."""
    m0, m1 = _minutes([first])[0], _minutes([last])[0]
    slots = [f"{m // 60:02d}:{m % 60:02d}" for m in range(m0, m1 + 1, 15)]
    t = pd.DataFrame({"slot": slots, "tickets": float(per_slot)})
    if extra:
        t = pd.concat([t, pd.DataFrame({"slot": list(extra), "tickets": list(extra.values())})])
    return t
