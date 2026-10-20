"""
09_crowding_descriptives.py - step 9: does crowding change behaviour? (descriptive checks before estimating)

Crowding = estimated people per room per 5-minute slot from ALL audio-guide sessions (08_build_crowding.py).
Relative crowding = people / that room's own 90th percentile (11:00-15:00 slots, normal days), so
"1.0" means "as full as this room gets on a busy 10% of slots".
To avoid measuring a person's own effect, crowding is always read in the 5-minute slot that ENDED
before the moment that matters (entry for stays, leaving for choices).

Hypotheses (behaviour sample = room_visits.csv, est+test, sessions >= 3 visits, normal days):
  H2  stay longer in a crowded room        : log stay ~ crowding of the room at entry
  H3  stay longer if the next room is full : log stay ~ crowding of the next room in sequence at entry
      both with the controls of the stay model (room x time band, first room, language, hours inside,
      revisit, closing) AND day fixed effects (compares slots within the same day)
  H1  avoid a crowded next room            : P(go to next room in sequence | leaving to another room)
                                             ~ crowding band of that next room
  skip  skip a crowded next room           : P(jump over it to the room after) ~ same bands
  exit  leave earlier when the building is full : P(exit) ~ building crowding band
      choice checks are linear probability models with controls (from-room x hour, hours inside,
      share of rooms seen, closing) and day fixed effects.
Outputs: output/crowding_descriptives.csv, output/crowding_descriptives.png
Run from the model folder:  python 09_crowding_descriptives.py   (about 2 minutes)
"""
import glob, os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layout import ZONES, ZONE_IDX, SEQ, N
from features import lang_group
import importlib
sm = importlib.import_module("04_stay_model")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
TEST = ("2026-05-18", "2026-06-11")
SLOT, DAY0 = 5, 8 * 60
BANDS = [0, 0.5, 0.75, 0.9, 1.1, np.inf]
BAND_LABELS = ["<50%", "50-75%", "75-90%", "90-110%", ">110%"]       # of the room's own p90; base = 50-75%
BASE = 1


def load_crowding():
    c = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, "crowding", "crowding_*.csv"))))
    hm = c.slot.str.split(":", expand=True).astype(int)
    c["k"] = (hm[0] * 60 + hm[1] - DAY0) // SLOT
    dates = sorted(c.date.unique())
    di = {d: i for i, d in enumerate(dates)}
    A = np.zeros((len(dates), 15 * 60 // SLOT, N))
    A[c.date.map(di).to_numpy(), c.k.to_numpy(), c.zone.map(ZONE_IDX).to_numpy()] = c.est_people.to_numpy()
    # each room's p90 on busy hours of normal days (EXH: only days it was open)
    mid = slice((11 * 60 - DAY0) // SLOT, (15 * 60 - DAY0) // SLOT)
    normal = np.array([not (TEST[0] <= d <= TEST[1]) for d in dates])
    p90 = np.array([np.percentile(A[normal, mid, z][A[normal, mid, z] > 1], 90) for z in range(N)])
    bld = A.sum(2)
    bld_p90 = np.percentile(bld[normal, mid], 90)
    return A, di, p90, bld, bld_p90


def lookup(A, di, date, t, zone=None):
    """people in the 5-min slot that ended before time t (zone may be -1 = none -> nan)."""
    k = ((t.dt.hour * 60 + t.dt.minute - DAY0) // SLOT - 1).clip(0, A.shape[1] - 1).to_numpy()
    d = date.map(di).to_numpy()
    if zone is None:
        return A[d, k]
    z = np.asarray(zone)
    out = np.full(len(z), np.nan)
    ok = z >= 0
    out[ok] = A[d[ok], k[ok], z[ok]]
    return out


def band(rel):
    b = np.digitize(rel, BANDS[1:-1])
    return np.where(np.isnan(rel), -1, b)


def band_dummies(b, prefix):
    return pd.DataFrame({f"{prefix}{BAND_LABELS[i]}": (b == i).astype(float)
                         for i in range(len(BAND_LABELS)) if i != BASE})


def demean_fit(y, X, groups):
    """OLS with group (day) fixed effects via within-transformation; returns beta, se (clustered by day)."""
    g = pd.factorize(groups)[0]
    G = g.max() + 1
    cnt = np.bincount(g, minlength=G)[:, None]
    Xd = np.asarray(X, dtype=np.float64)
    M = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))
    Xd -= (M @ Xd / cnt)[g]
    yd = y - (np.bincount(g, y, minlength=G) / cnt[:, 0])[g]
    keep = Xd.std(0) > 1e-9
    Xd = Xd[:, keep]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ yd)
    e = yd - Xd @ beta
    # cluster-robust by day
    Xd *= e[:, None]
    S = M @ Xd
    del Xd
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    b, s = np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    b[keep], s[keep] = beta, np.sqrt(np.diag(V))
    return b, s


def next_in_seq(cur):
    """index of the next numbered room (R01->R02 ... R11->R12); -1 for R12 and EXH."""
    nxt = np.full(N, -1)
    for z in range(N):
        if SEQ[z] > 0:
            m = np.where(SEQ == SEQ[z] + 1)[0]
            if len(m):
                nxt[z] = m[0]
    return nxt[cur]


def main():
    A, di, p90, bld, bld_p90 = load_crowding()
    v = pd.read_csv(os.path.join(DATA, "room_visits.csv"), parse_dates=["s_start", "t_in", "t_out"])
    v = v[(v.n_visits >= 3) & ~v.date.between(*TEST) & v.date.isin(di)].sort_values(["sid", "visit_no"])
    v["revisit"] = v.groupby(["sid", "zone"]).cumcount() > 0
    v["cur"] = v.zone.map(ZONE_IDX)
    v["nxt_seq"] = next_in_seq(v.cur.to_numpy())
    v["nxt2_seq"] = np.where(v.nxt_seq >= 0, next_in_seq(np.clip(v.nxt_seq, 0, None)), -1)
    v["chosen"] = v.groupby("sid").zone.shift(-1).map(ZONE_IDX).fillna(N).astype(int)
    close = pd.read_csv(os.path.join(DATA, "closing_times.csv")).set_index("date").close_h
    rows = []

    # ================= H2 / H3: stays
    s = v[(~v.is_last) & (v.stay_min <= sm.MAX_STAY)].copy()
    clock = s.t_in.dt.hour + s.t_in.dt.minute / 60
    Xc = sm.stay_X(s.cur.to_numpy(), s.lang.map(lang_group).to_numpy(), clock.to_numpy(),
                   (s.min_inside / 60).to_numpy(), s.revisit.to_numpy(), (s.t_in.dt.dayofweek == 4).to_numpy(),
                   (s.visit_no == 1).to_numpy(), (s.date.map(close) - clock).to_numpy())
    rel_cur = lookup(A, di, s.date, s.t_in, s.cur) / p90[s.cur]
    nz = s.nxt_seq.to_numpy()
    rel_nxt = lookup(A, di, s.date, s.t_in, nz) / np.where(nz >= 0, p90[np.clip(nz, 0, None)], np.nan)
    has_nxt = ~np.isnan(rel_nxt)
    y = np.log(np.clip(s.stay_min.to_numpy(), sm.MIN_STAY, None))
    for name, rel, mask in [("H2 stay ~ own room crowding at entry", rel_cur, np.ones(len(s), bool)),
                            ("H3 stay ~ next room crowding at entry", rel_nxt, has_nxt),
                            ("H2+H3 own room (joint)", rel_cur, has_nxt),
                            ("H2+H3 next room (joint)", rel_nxt, has_nxt)]:
        if "joint" in name:
            D = pd.concat([band_dummies(band(rel_cur[mask]), "own "), band_dummies(band(rel_nxt[mask]), "next ")], axis=1)
            pick = [c for c in D.columns if c.startswith("own" if "own" in name else "next")]
        else:
            D = band_dummies(band(rel[mask]), "")
            pick = list(D.columns)
        X = np.hstack([D.to_numpy(), Xc[mask]])
        b, se = demean_fit(y[mask], X, s.date.to_numpy()[mask])
        cnt = np.bincount(band(rel[mask])[band(rel[mask]) >= 0], minlength=5)
        for j, col in enumerate(D.columns):
            if col in pick:
                lab = col.split(" ")[-1]
                rows.append({"test": name, "band": lab, "effect": np.exp(b[j]) - 1, "se": se[j] * np.exp(b[j]),
                             "n": int(cnt[BAND_LABELS.index(lab)]), "unit": "% change in stay"})

    # ================= decisions: H1, skip, exit
    d = v.copy()
    d["clock_h"] = d.t_out.dt.hour + d.t_out.dt.minute / 60
    d["hours"] = (d.t_out - d.s_start).dt.total_seconds() / 3600
    d["to_close"] = d.date.map(close) - d.clock_h
    d["seen"] = 0
    for z in ZONES[:12]:
        d["seen"] += (d.zone == z).astype(int).groupby(d.sid).cummax()
    hb = pd.cut(d.clock_h, [0, 10, 11, 12, 13, 14, 15, 16, 17, 30], right=False, labels=False)
    ctrl = np.hstack([pd.get_dummies(d.zone.astype(str) + "_" + hb.astype(str), dtype=np.float32).to_numpy(),
                      np.column_stack([d.hours, d.hours ** 2, d.seen / 12, d.to_close < 1,
                                       d.to_close < 0.5]).astype(np.float32)])
    nz = d.nxt_seq.to_numpy()
    rel_n = lookup(A, di, d.date, d.t_out, nz) / np.where(nz >= 0, p90[np.clip(nz, 0, None)], np.nan)
    rel_b = lookup(A, di, d.date, d.t_out).sum(1) / bld_p90
    moved = (d.chosen < N).to_numpy()
    ok = ~np.isnan(rel_n) & moved & (d.cur != ZONE_IDX["R12"]).to_numpy()
    chose_next = (d.chosen.to_numpy() == nz).astype(float)
    chose_skip = (d.chosen.to_numpy() == d.nxt2_seq.to_numpy()).astype(float)
    tests = [("H1 go to next room ~ its crowding", chose_next, rel_n, ok, "pp change in probability"),
             ("Skip next room ~ its crowding", chose_skip, rel_n, ok, "pp change in probability"),
             ("Exit ~ building crowding", (~moved).astype(float), rel_b, np.ones(len(d), bool),
              "pp change in probability")]
    base_rates = {}
    for name, yv, rel, mask, unit in tests:
        D = band_dummies(band(rel[mask]), "")
        X = np.hstack([D.to_numpy(), ctrl[mask]])
        b, se = demean_fit(yv[mask], X, d.date.to_numpy()[mask])
        bb = band(rel[mask])
        base_rates[name] = yv[mask][bb == BASE].mean()
        cnt = np.bincount(bb[bb >= 0], minlength=5)
        for j, col in enumerate(D.columns):
            rows.append({"test": name, "band": col, "effect": b[j], "se": se[j], "n": int(cnt[BAND_LABELS.index(col)]),
                         "unit": unit})
    res = pd.DataFrame(rows)
    res["base_rate"] = res.test.map(base_rates)
    res["z"] = res.effect / res.se
    res.round(4).to_csv(os.path.join(OUT, "crowding_descriptives.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print("p90 per room:", dict(zip(ZONES, p90.round(0))), "building p90:", round(bld_p90))
    chart(res)


def chart(res):
    BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    show = [("H2 stay ~ own room crowding at entry", "H2: own room crowded → stay longer?", 100, "% change in stay"),
            ("H3 stay ~ next room crowding at entry", "H3: next room crowded → stay longer here?", 100, "% change in stay"),
            ("H1 go to next room ~ its crowding", "H1: next room crowded → go there less?", 100, "pp change"),
            ("Exit ~ building crowding", "Building crowded → leave sooner?", 100, "pp change")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.3), dpi=200, sharey=False)
    fig.patch.set_facecolor(BG)
    for ax, (t, title, mult, unit) in zip(axs, show):
        r = res[res.test == t].set_index("band").reindex(BAND_LABELS)
        r.loc[BAND_LABELS[BASE], ["effect", "se"]] = 0
        x = np.arange(len(BAND_LABELS))
        ax.set_facecolor(BG)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.errorbar(x, r.effect * mult, yerr=1.96 * r.se * mult, fmt="o", color="#2a78d6", ecolor="#9cc0ec",
                    capsize=3, ms=6, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(BAND_LABELS, fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(unit, color=INK2, fontsize=8.5)
        ax.set_xlabel("crowding vs the room's own busy level (p90)", color=INK2, fontsize=8)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8); ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    fig.suptitle("Does crowding change behaviour? (vs the 50-75% band; same-day comparisons; 95% intervals)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.8, bottom=0.17, wspace=0.32)
    fig.savefig(os.path.join(OUT, "crowding_descriptives.png"), facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
