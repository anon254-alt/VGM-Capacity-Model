"""
sim.py - Monte Carlo simulation of visitors moving through the museum (step 5).

Each simulated visitor:
  1. arrives at a real scan time (ticket file, 5-min slots, spread uniformly inside the slot)
  2. waits a lead time (empirical), then picks a first room (entry shares)
  3. stays a time drawn from the stay-time model (prediction + real error of that room)
  4. picks the next room or EXIT with the logit (only open rooms are offered)
  5. repeats until EXIT or closing time
Language of each visitor is drawn from that day's audio-guide language mix.

Common random numbers: every random draw comes from a stream fixed by (seed, replication), drawn for
all visitors at every step, so a baseline and a scenario that share arrivals use identical luck.

Main function:
  simulate_day(date, n_reps=100, closed=(), extra_open=(), arrivals_factor=1.0, seed=0)
returns a dict with occupancy (reps x 15-min slots x 13 zones, average people in the slot),
slot labels, zone list and visitor-level summaries.
"""
import json, os
import numpy as np
import pandas as pd

from layout import ZONES, ZONE_IDX, N, FLOOR, FLOOR_IDX, FLOORS
from features import build_X, J, LANG_GROUPS, lang_group
import importlib
stay_mod = importlib.import_module("04_stay_model")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, PAR = os.path.join(HERE, "data"), os.path.join(HERE, "params")
SLOT = 15
MAX_STEPS = 60
DAY_START, DAY_END = 8 * 60, 23 * 60          # simulation clock window (minutes)


class Model:
    def __init__(self):
        lg = json.load(open(os.path.join(PAR, "logit.json")))
        self.beta = np.array(lg["beta"])
        st = json.load(open(os.path.join(PAR, "stay.json")))
        self.sbeta = np.array(st["beta"])
        self.smin, self.smax = st["min_stay_min"], st["max_stay_min"]
        r = np.load(os.path.join(PAR, "stay_residuals.npz"))
        self.resid = [r[z] for z in ZONES]
        en = json.load(open(os.path.join(PAR, "entry.json")))
        self.first = en["first_room_shares"]
        self.lead = np.array(en["lead_minutes_sample"])
        self.open = pd.read_csv(os.path.join(DATA, "open_rooms.csv")).pivot(index="date", columns="zone", values="open")
        self.arr = pd.read_csv(os.path.join(DATA, "arrivals_5min.csv"))
        lp = os.path.join(DATA, "lang_by_date.csv")
        if not os.path.exists(lp):
            v = pd.read_csv(os.path.join(DATA, "room_visits.csv"), usecols=["sid", "date", "lang"]).drop_duplicates("sid")
            v["g"] = v.lang.map(lang_group)
            v.groupby(["date", "g"]).size().rename("n").reset_index().to_csv(lp, index=False)
        cl = pd.read_csv(os.path.join(DATA, "closing_times.csv"))
        self.close = cl.set_index("date").close_h
        self.close_by_weekday = cl.groupby("weekday").close_h.median()
        ct = os.path.join(PAR, "counted_table.json")
        self.counted = np.array(json.load(open(ct))["counts"]) if os.path.exists(ct) else None
        cp = os.path.join(PAR, "calibration.json")
        cal = json.load(open(cp)) if os.path.exists(cp) else {}
        self.exit_share = cal.get("exit_share_if_closed", {})
        self.continue_shift = cal.get("continue_shift", 0.0)
        lb = pd.read_csv(lp)
        self.lang = lb.pivot(index="date", columns="g", values="n").fillna(0).reindex(columns=range(len(LANG_GROUPS)), fill_value=0)

    def open_zones(self, date, closed=(), extra_open=()):
        o = self.open.loc[date].reindex(ZONES).fillna(True).astype(bool) if date in self.open.index \
            else pd.Series(True, index=ZONES)
        for z in closed:
            o[z] = False
        for z in extra_open:
            o[z] = True
        return o.to_numpy()


def typical_mix(m, date):
    """Language mix for a day without data: average of the same month and weekday, else same month,
    else all days (13_mix_sensitivity.py: the typical mix is within ~0.5% of the real one for crowding)."""
    d = pd.to_datetime(pd.Series(m.lang.index, index=m.lang.index))
    t = pd.Timestamp(date)
    S = m.lang.div(m.lang.sum(axis=1), axis=0)
    for sel in (d.dt.month.eq(t.month) & d.dt.dayofweek.eq(t.dayofweek), d.dt.month.eq(t.month), d.notna()):
        if sel.any():
            return S[sel.to_numpy()].mean().to_numpy()


def _first_shares(m, exh_open, clock_h, open_mask):
    band = np.where(clock_h < 11, "before_11h", np.where(clock_h < 14, "11h_14h", "after_14h"))
    out = np.zeros((len(clock_h), N))
    for b in ("before_11h", "11h_14h", "after_14h"):
        key = f"exh_{'open' if exh_open else 'closed'}|{b}"
        p = np.array([m.first[key][z] for z in ZONES]) * open_mask
        p = p / p.sum()
        out[band == b] = p
    return out


def simulate_day(date, n_reps=100, closed=(), extra_open=(), arrivals_factor=1.0, seed=0, model=None,
                 arrivals=None, exit_share=None, continue_shift=None, router="logit", lang_mix=None):
    """exit_share: {zone: q} - when zone is closed, share q of the visitors who would have chosen it
    leave the museum instead; the rest spread over the open options as in the plain logit.
    router: "logit" (the model) or "counted" (benchmark: P(next room | current room) counted from
    the estimation sample, renormalised over open rooms; no other factors).
    continue_shift: added to the attractiveness of every room vs EXIT (calibrated so simulated
    crowding matches measured crowding on normal days). Defaults: params/calibration.json.
    lang_mix: shares of the language groups (features.LANG_GROUPS order); default = that day's mix."""
    m = model or Model()
    open_mask = m.open_zones(date, closed, extra_open)
    q = dict(m.exit_share) if exit_share is None else dict(exit_share)
    q_vec = np.array([q.get(z, 0.0) for z in ZONES])
    shift = m.continue_shift if continue_shift is None else continue_shift
    a = arrivals if arrivals is not None else m.arr[m.arr.date == date]
    if a.empty:
        raise ValueError(f"no arrivals for {date}")
    slot_min = a.slot.str[:2].astype(int).to_numpy() * 60 + a.slot.str[3:5].astype(int).to_numpy()
    counts = np.round(a.visitors.to_numpy() * arrivals_factor).astype(int)
    arr_slot = np.repeat(slot_min, counts).astype(float)
    n = len(arr_slot)
    friday = pd.Timestamp(date).dayofweek == 4
    # closing time: from the data for that day, else the typical closing time for that weekday
    close_h = m.close.get(date, m.close_by_weekday.get(pd.Timestamp(date).dayofweek, 17.75))
    close_t = close_h * 60
    lang_p = m.lang.loc[date].to_numpy() if date in m.lang.index else typical_mix(m, date)
    if lang_mix is not None:                             # scenario: a different visitor (language) mix
        lang_p = np.asarray(lang_mix, dtype=float)
    lang_p = lang_p / lang_p.sum()
    n_slots = (DAY_END - DAY_START) // SLOT
    occ = np.zeros((n_reps, n_slots, N))
    durations, n_rooms = [], []

    for rep in range(n_reps):
        rng = np.random.default_rng([seed, rep])
        t = arr_slot + rng.random(n) * 5                 # arrival inside the 5-min slot
        start = t.copy()
        lang = np.searchsorted(np.cumsum(lang_p), rng.random(n)).clip(0, len(LANG_GROUPS) - 1)
        t = t + m.lead[rng.integers(0, len(m.lead), n)]
        P0 = _first_shares(m, open_mask[ZONE_IDX["EXH"]], t / 60, open_mask)
        cur = (rng.random(n)[:, None] > np.cumsum(P0, 1)).sum(1).clip(0, N - 1)
        seen_z = np.zeros((n, N), bool)
        seen_f = np.zeros((n, len(FLOORS)), bool)
        active = np.ones(n, bool)
        visits = np.zeros(n, int)
        iv_z, iv_a, iv_b = [], [], []
        floor_of = np.array([FLOOR_IDX[FLOOR[z]] for z in ZONES])
        for step in range(MAX_STEPS):
            u_stay, u_choice, u_res = rng.random(n), rng.random(n), rng.integers(0, 1 << 30, n)
            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break
            c = cur[idx]
            revisit = seen_z[idx, c]
            clock = t[idx] / 60
            Xs = stay_mod.stay_X(c, lang[idx], clock, (t[idx] - start[idx]) / 60, revisit, np.full(len(idx), friday),
                                 visits[idx] == 0, close_h - clock)
            res = np.array([m.resid[z][r % len(m.resid[z])] for z, r in zip(c, u_res[idx])])
            stay = np.clip(np.exp(Xs @ m.sbeta + res), m.smin, m.smax)
            stay = np.minimum(stay, np.maximum(close_t - t[idx], 0.25))    # nobody stays past closing
            iv_z.append(c); iv_a.append(t[idx]); iv_b.append(t[idx] + stay)
            t[idx] += stay
            seen_z[idx, c] = True
            seen_f[idx, floor_of[c]] = True
            visits[idx] += 1
            # next room or exit
            if router == "counted":
                P = m.counted[c] * np.append(open_mask, True)[None, :]
                P[np.arange(len(idx)), c] = 0
                P /= P.sum(1, keepdims=True)
                choice = (u_choice[idx][:, None] > np.cumsum(P, 1)).sum(1).clip(0, J - 1)
                leave = (choice == N) | (t[idx] >= close_t)
                active[idx[leave]] = False
                cur[idx[~leave]] = choice[~leave]
                continue
            X = build_X(c, (t[idx] - start[idx]) / 60, seen_z[idx], seen_f[idx], t[idx] / 60,
                        np.full(len(idx), friday), lang[idx], open_mask=open_mask,
                        hours_to_close=close_h - t[idx] / 60)
            V = np.einsum("djk,k->dj", X, m.beta)
            V[:, :N] += shift                                       # calibration of "continue" vs exit
            notcur = np.ones((len(idx), J), bool)
            notcur[np.arange(len(idx)), c] = False
            V[~notcur] = -np.inf
            V -= V.max(1, keepdims=True)
            Pf = np.exp(V); Pf /= Pf.sum(1, keepdims=True)          # as if every room were open
            closed_m = Pf[:, :N] * (~open_mask)                     # mass of closed rooms
            to_exit = (closed_m * q_vec).sum(1)
            P = Pf.copy(); P[:, :N] *= open_mask
            P *= ((1 - to_exit) / P.sum(1))[:, None]                # rest spread as plain logit
            P[:, N] += to_exit
            choice = (u_choice[idx][:, None] > np.cumsum(P, 1)).sum(1).clip(0, J - 1)
            leave = (choice == N) | (t[idx] >= close_t)
            active[idx[leave]] = False
            stay_idx = idx[~leave]
            cur[stay_idx] = choice[~leave]
        active[:] = False
        z = np.concatenate(iv_z); a0 = np.concatenate(iv_a); b0 = np.concatenate(iv_b)
        occ[rep] = occupancy(z, a0, b0, n_slots)
        durations.append(float(np.mean(t - start)))
        n_rooms.append(float(seen_z.sum(1).mean()))

    slots = [f"{(DAY_START + i * SLOT) // 60:02d}:{(DAY_START + i * SLOT) % 60:02d}" for i in range(n_slots)]
    return {"occ": occ, "slots": slots, "zones": ZONES, "open": open_mask, "n_visitors": n,
            "mean_visit_min": float(np.mean(durations)), "mean_rooms": float(np.mean(n_rooms))}


def occupancy(zone, t_in, t_out, n_slots):
    """Average people per zone in each 15-min slot from visit intervals (minutes since midnight)."""
    a = np.clip(t_in, DAY_START, DAY_END - 1e-6) - DAY_START
    b = np.clip(t_out, DAY_START, DAY_END - 1e-6) - DAY_START
    keep = b > a
    zone, a, b = zone[keep], a[keep], b[keep]
    s0, s1 = (a // SLOT).astype(int), (b // SLOT).astype(int)
    k = s1 - s0 + 1
    rows = np.repeat(np.arange(len(a)), k)
    sl = s0[rows] + (np.arange(len(rows)) - np.repeat(np.cumsum(k) - k, k))
    lo = np.maximum(a[rows], sl * SLOT)
    hi = np.minimum(b[rows], (sl + 1) * SLOT)
    out = np.zeros((n_slots, N))
    np.add.at(out, (sl, zone[rows]), np.clip(hi - lo, 0, None) / SLOT)
    return out
