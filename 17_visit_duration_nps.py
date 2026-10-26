"""
17_visit_duration_nps.py - step 17: the visit-duration file (entry + exit scans, 26 Sep 2025 - 6 Apr 2026).

Part 1 - audio guide vs no audio guide: total visit time (entry to exit scan) for orders with and without an
         audio guide. Raw averages, then a regression of log(duration) with controls (country, exhibition,
         youth in the order, group size, entry time, weekday, month); errors clustered by day.
Part 2 - NPS vs congestion, per respondent (the NPS answer is linked to a known visit window):
  building congestion (all days): people inside the building estimated from ticket scans
     inside(t) = sum over earlier 5-min slots k of entries(t - k) x P(visit lasts longer than k)
     with P from this file's visit durations. Measures: at entry, average and peak during the visit,
     the day's total entries and the day's peak.
  room congestion (only visits from 1 Feb 2026, when the audio-guide crowding measure exists):
     the most crowded room during the visit (relative to that room's busy level, p90) and Room 1 at entry.
  For each: Spearman correlation with the score, NPS by thirds of congestion, and a regression of the
  score on the (standardised) measure with the same controls.
Note: entry and exit times in the file are rounded to 30 minutes; the entry is taken as the middle of the
half hour, the exit as entry + exact duration.
Outputs: output/visit_duration_audioguide.csv, output/nps_vs_congestion_visits.csv,
         output/nps_vs_congestion_visits.png
Run from the model folder:  python 17_visit_duration_nps.py
"""
import glob, json, os
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
FILE = os.path.join(HERE, "..", "..", "Raw Data", "20261001 - Visit duration - IESE.xlsx")
STEP = 5


def load_visits():
    x = pd.read_excel(FILE)
    x = x[x.Visitduration_in_min.between(15, 360)].copy()          # drop scan errors
    x["date"] = x.Scan_date.dt.strftime("%Y-%m-%d")
    x["entry"] = x.Scan_time + pd.Timedelta(minutes=15)
    x["exit"] = x.entry + pd.to_timedelta(x.Visitduration_in_min, unit="m")
    x["ag"] = (x.Order_contains_audioguide == "Yes").astype(float)
    x["youth"] = (x.Order_contains_youth == "Yes").astype(float)
    x["grp"] = pd.cut(x.Amount_of_tickets_in_order, [0, 1, 2, 4, 1000], labels=["1", "2", "3-4", "5+"])
    x["hour"] = x.Scan_time.dt.hour.clip(9, 17)
    x["wd"] = x.Scan_date.dt.dayofweek
    x["month"] = x.Scan_date.dt.strftime("%Y-%m")
    return x


def controls(x, cols=("Country", "Exhibition", "grp", "hour", "wd", "month"), extra=()):
    D = pd.get_dummies(x[list(cols)].astype(str), drop_first=True).astype(float)
    for c in extra:
        D[c] = x[c].to_numpy(float)
    return D


def ols(y, X, groups=None):
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    if groups is None:
        V = XtXi @ (X.T * e ** 2) @ X @ XtXi                       # robust (HC0)
    else:
        S = pd.DataFrame(X * e[:, None]).groupby(np.asarray(groups)).sum().to_numpy()
        V = XtXi @ S.T @ S @ XtXi
    return b[1:], np.sqrt(np.diag(V))[1:]


def nps(scores):
    s = np.asarray(scores)
    return ((s >= 9).mean() - (s <= 6).mean()) * 100


# ------------------------------------------------------------------ Part 1
def audio_guide(x):
    rows = []
    for name, g in [("all", x)] + [(f"exhibition: {e}", g) for e, g in x.groupby("Exhibition")] + \
                   [(f"youth in order: {y}", g) for y, g in x.groupby("Order_contains_youth")]:
        a, n = g[g.ag == 1].Visitduration_in_min, g[g.ag == 0].Visitduration_in_min
        C = controls(g, cols=[c for c in ("Country", "Exhibition", "grp", "hour", "wd", "month") if g[c].nunique() > 1],
                     extra=["youth"] if g.youth.nunique() > 1 else [])
        b, se = ols(np.log(g.Visitduration_in_min.to_numpy(float)), np.column_stack([g.ag, C]), g.date)
        rows.append({"group": name, "n_audio": len(a), "n_no_audio": len(n),
                     "mean_audio": a.mean(), "mean_no_audio": n.mean(), "median_audio": a.median(),
                     "median_no_audio": n.median(), "raw_diff_min": a.mean() - n.mean(),
                     "adj_effect_pct": (np.exp(b[0]) - 1) * 100,
                     "ci_low_pct": (np.exp(b[0] - 1.96 * se[0]) - 1) * 100, "ci_high_pct": (np.exp(b[0] + 1.96 * se[0]) - 1) * 100,
                     "adj_effect_min": (np.exp(b[0]) - 1) * n.mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Part 2
def building_inside(x):
    """People inside the building per 5 min, all days, from scans x survival of visit duration."""
    a = pd.read_csv(os.path.join(DATA, "arrivals_5min_from2025.csv"))
    dur = x.Visitduration_in_min.to_numpy()
    lags = np.arange(0, 361, STEP)
    surv = np.array([(dur > k).mean() for k in lags])                 # P(still inside k minutes after entering)
    grid = [f"{h:02d}:{m:02d}" for h in range(8, 23) for m in range(0, 60, STEP)]
    out = {}
    for d, g in a.groupby("date"):
        e = g.set_index("slot").visitors.reindex(grid, fill_value=0).to_numpy(float)
        out[d] = np.convolve(e, surv)[:len(grid)]
    return pd.DataFrame(out, index=grid).T, grid


def window(series_row, grid, t0, t1):
    m0, m1 = t0.hour * 60 + t0.minute, t1.hour * 60 + t1.minute
    mins = np.array([int(s[:2]) * 60 + int(s[3:]) for s in grid])
    sel = (mins >= m0 - STEP) & (mins <= m1)
    return series_row[sel]


def congestion_measures(x):
    inside, grid = building_inside(x)
    rows = []
    # room crowding (audio-guide measure) from Feb 2026
    files = sorted(glob.glob(os.path.join(DATA, "crowding", "crowding_*.csv")))
    rc = pd.concat(pd.read_csv(f) for f in files) if files else pd.DataFrame(columns=["date", "slot", "zone", "est_people"])
    p90 = rc[(rc.slot >= "11:00") & (rc.slot < "15:00") & (rc.est_people > 1)].groupby("zone").est_people.quantile(0.9)
    rc["rel"] = rc.est_people / rc.zone.map(p90)
    rc_piv = {d: g.pivot(index="slot", columns="zone", values="rel") for d, g in rc.groupby("date")}
    r1 = {d: g[g.zone == "R01"].set_index("slot").est_people for d, g in rc.groupby("date")}
    for i, r in x.iterrows():
        row = {"idx": i}
        if r.date in inside.index:
            s = inside.loc[r.date].to_numpy()
            w = window(s, grid, r.entry, r.exit)
            row.update(b_entry=window(s, grid, r.entry, r.entry)[-1] if len(w) else np.nan,
                       b_avg=w.mean() if len(w) else np.nan, b_peak=w.max() if len(w) else np.nan,
                       d_entries=None, d_peak=s.max())
        if r.date in rc_piv:
            pv = rc_piv[r.date]
            m0 = (r.entry - pd.Timedelta(minutes=5)).strftime("%H:%M")
            m1 = r.exit.strftime("%H:%M")
            sub = pv[(pv.index >= m0) & (pv.index <= m1)]
            row["room_peak_rel"] = sub.max().max() if len(sub) else np.nan
            row["room_avg_rel"] = sub.mean().mean() if len(sub) else np.nan
            e0, e1 = r.entry.strftime("%H:%M"), (r.entry + pd.Timedelta(minutes=30)).strftime("%H:%M")
            q = r1[r.date]
            row["room1_entry"] = q[(q.index >= e0) & (q.index < e1)].mean()
        rows.append(row)
    m = pd.DataFrame(rows).set_index("idx")
    daily = pd.read_csv(os.path.join(DATA, "arrivals_5min_from2025.csv")).groupby("date").visitors.sum()
    m["d_entries"] = x.date.map(daily)
    return m, inside


MEASURES = [("b_entry", "Building: people inside at entry"), ("b_avg", "Building: average during the visit"),
            ("b_peak", "Building: peak during the visit"), ("d_entries", "Day: total entries"),
            ("d_peak", "Day: peak people inside"), ("room_avg_rel", "Rooms: average crowding during visit*"),
            ("room_peak_rel", "Rooms: most crowded room during visit*"), ("room1_entry", "Room 1: people at entry*")]


def main():
    np.seterr(invalid="ignore")
    x = load_visits()
    ag = audio_guide(x)
    ag.round(2).to_csv(os.path.join(OUT, "visit_duration_audioguide.csv"), index=False)
    # time ratio non-audio-guide / audio-guide visitors (adjusted, all visits) -> used by the crowding measure
    # (08_build_crowding.py, 06_validate.py) and the simulation (18_audio_guide.py calibrates the stay multiplier)
    pj = os.path.join(HERE, "params", "audio_guide.json")
    par = json.load(open(pj)) if os.path.exists(pj) else {}
    eff = float(ag.loc[ag.group == "all", "adj_effect_pct"].iloc[0])
    par.update({"non_ag_time_ratio": round(1 / (1 + eff / 100), 4),
                "source_time_ratio": "17_visit_duration_nps.py: audio-guide orders stay +%.1f%% longer (adjusted), "
                                     "visit-duration file Sep 2025-Apr 2026" % eff})
    json.dump(par, open(pj, "w"), indent=1)
    pd.set_option("display.width", 220)
    print(ag.round(1).to_string(index=False))

    m, inside = congestion_measures(x)
    v = x.join(m)
    n = v[v.Nps_score.notna()].copy()
    print("NPS answers:", len(n), "| from Feb 2026 (room measures):", n.room_peak_rel.notna().sum())
    rows, terc = [], {}
    C = controls(n, extra=["ag", "youth"])
    for col, label in MEASURES:
        ok = n[col].notna().to_numpy()
        if ok.sum() < 30:
            continue
        g = n[ok]
        rho, p = ss.spearmanr(g[col], g.Nps_score)
        z = (g[col] - g[col].mean()) / g[col].std()
        Cg = C[ok].loc[:, C[ok].std() > 0]
        b, se = ols(g.Nps_score.to_numpy(float), np.column_stack([z, Cg]))
        t = pd.qcut(g[col].rank(method="first"), 3, labels=["low", "middle", "high"])
        nps_t = g.groupby(t, observed=True).Nps_score.apply(nps)
        boot = []
        rng = np.random.default_rng(1)
        for _ in range(1000):
            s = g.sample(len(g), replace=True, random_state=rng.integers(1e9))
            ts = pd.qcut(s[col].rank(method="first"), 3, labels=["low", "middle", "high"])
            ns = s.groupby(ts, observed=True).Nps_score.apply(nps)
            boot.append(ns.get("high", np.nan) - ns.get("low", np.nan))
        lo_, hi_ = np.nanpercentile(boot, [2.5, 97.5])
        terc[label] = (nps_t, g.groupby(t, observed=True)[col].mean())
        rows.append({"measure": label, "n": int(ok.sum()), "spearman": rho, "p_value": p,
                     "score_change_per_sd": b[0], "score_change_se": se[0],
                     "nps_low_third": nps_t["low"], "nps_mid_third": nps_t["middle"], "nps_high_third": nps_t["high"],
                     "nps_high_minus_low": nps_t["high"] - nps_t["low"], "ci_low": lo_, "ci_high": hi_,
                     "low_third_avg": g.groupby(t, observed=True)[col].mean()["low"],
                     "high_third_avg": g.groupby(t, observed=True)[col].mean()["high"]})
    res = pd.DataFrame(rows)
    res.round(3).to_csv(os.path.join(OUT, "nps_vs_congestion_visits.csv"), index=False)
    print(res.round(2).to_string(index=False))
    print("overall NPS:", round(nps(n.Nps_score), 1), "| mean score", round(n.Nps_score.mean(), 2))
    # sanity: building estimate vs audio-guide measured building total (Feb-Apr 2026)
    files = sorted(glob.glob(os.path.join(DATA, "crowding", "crowding_2026-0[234].csv")))
    if files:
        rc = pd.concat(pd.read_csv(f) for f in files)
        tot = rc.groupby(["date", "slot"]).est_people.sum()
        est = inside.stack().rename("est")
        j = pd.concat([tot, est], axis=1, join="inner")
        j = j[(j.index.get_level_values(1) >= "10:00") & (j.index.get_level_values(1) < "17:00")]
        print("check: building estimate from scans vs audio-guide total, corr", round(j.corr().iloc[0, 1], 3),
              "ratio", round((j.est.sum() / j.est_people.sum()), 3))
    chart(res, terc)


def chart(res, terc):
    BG, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(res) + 1.6), dpi=200)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    y = np.arange(len(res))[::-1]
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.errorbar(res.nps_high_minus_low, y, xerr=[res.nps_high_minus_low - res.ci_low, res.ci_high - res.nps_high_minus_low],
                fmt="o", color=BLUE, ecolor="#9cc0ec", capsize=3, ms=7)
    for yi, (_, r) in zip(y, res.iterrows()):
        ax.text(ax.get_xlim()[1] if False else 1.0, yi, f"  n={r.n}, Spearman {r.spearman:+.2f} (p={r.p_value:.2f})",
                transform=ax.get_yaxis_transform(), fontsize=8, color=INK2, va="center")
    ax.set_yticks(y); ax.set_yticklabels(res.measure, fontsize=9)
    ax.set_xlabel("NPS of the most crowded third minus the least crowded third (points, 95% bootstrap interval)",
                  color=INK2, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, axis="x", color=GRID, lw=0.8)
    fig.suptitle("NPS vs congestion during the visit (visit-duration file, Sep 2025 – Apr 2026)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.set_title("* room measures only for visits from 1 Feb 2026 (audio-guide crowding data)", loc="left", fontsize=8.5,
                 color=INK2)
    fig.subplots_adjust(left=0.28, right=0.72, top=0.84, bottom=0.12)
    fig.savefig(os.path.join(OUT, "nps_vs_congestion_visits.png"), facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
