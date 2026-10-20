"""
10_weather_vs_duration.py - step 10: are visits longer when it rains?

Weather: KNMI hourly data, station 240 (Schiphol), External Data/Amsterdam_weather_data.txt.
  KNMI hour HH (UT) covers HH-1 to HH UT; converted to Amsterdam local time (CET/CEST).
  A "wet hour" = at least 0.1 mm of rain in that hour.
Visit duration = from the start of the audio-guide session to leaving the last room (sessions with
>= 3 room visits, 15% sample in data/room_visits.csv, all days Feb-Sep 2026, capped at 6 hours).

Three ways of defining "raining" (each answers a slightly different question):
  A. rainy DAY      : >= 1 mm during opening hours (09:00-18:00) vs dry day (no wet hour 09:00-18:00)
  B. rain AT ARRIVAL: the hour the visitor starts is wet
  C. rain WHILE INSIDE: number of wet hours in the 2 hours after arrival (a fixed window, so long
                     visits are not more likely to "see" rain just because they are long)
For each: raw averages, then a regression of log(duration) with controls (start hour, weekday, month,
language, Friday evening, closing time, daily mean temperature), standard errors clustered by day.
Outputs: output/weather_vs_duration.csv, output/weather_vs_duration.png, data/weather_hourly.csv
Run from the model folder:  python 10_weather_vs_duration.py
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import lang_group, LANG_GROUPS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
WEATHER = os.path.join(HERE, "..", "..", "External Data", "Amsterdam_weather_data.txt")
MAX_DUR = 360


def load_weather():
    rows = []
    with open(WEATHER, encoding="latin-1") as f:
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) == 25 and p[0] == "240" and p[1][:4] == "2026":
                rows.append(p)
    cols = ["STN", "YYYYMMDD", "HH", "DD", "FH", "FF", "FX", "T", "T10N", "TD", "SQ", "Q", "DR", "RH", "P",
            "VV", "N", "U", "WW", "IX", "M", "R", "S", "O", "Y"]
    w = pd.DataFrame(rows, columns=cols)
    for c in ["HH", "T", "RH", "DR", "SQ"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    end_ut = pd.to_datetime(w.YYYYMMDD, format="%Y%m%d") + pd.to_timedelta(w.HH, unit="h")
    start_local = (end_ut - pd.Timedelta(hours=1)).dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
    out = pd.DataFrame({"hour_local": start_local,                      # hour starting at this local time
                        "rain_mm": np.where(w.RH < 0, 0.02, w.RH / 10),  # -1 = < 0.05 mm
                        "rain_minutes": w.DR * 6, "temp_c": w["T"] / 10, "sun_minutes": w.SQ.clip(lower=0) * 6})
    out["wet"] = out.rain_mm >= 0.1
    out.to_csv(os.path.join(DATA, "weather_hourly.csv"), index=False)
    return out.set_index("hour_local")


def cluster_ols(y, X, groups):
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    g = pd.factorize(groups)[0]
    M = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))))
    S = M @ (X * e[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    return b[1:], np.sqrt(np.diag(V))[1:]


def main():
    w = load_weather()
    v = pd.read_csv(os.path.join(DATA, "room_visits.csv"), usecols=["sid", "date", "lang", "s_start", "t_out", "n_visits"],
                    parse_dates=["s_start", "t_out"])
    v = v[v.n_visits >= 3]
    s = v.groupby("sid").agg(date=("date", "first"), lang=("lang", "first"), start=("s_start", "first"),
                             last=("t_out", "max"))
    s["dur"] = (s["last"] - s.start).dt.total_seconds() / 60
    s = s[(s.dur > 5) & (s.dur <= MAX_DUR)].copy()

    # ---- day level: opening hours 09:00-18:00
    day = w[(w.index.hour >= 9) & (w.index.hour < 18)].groupby(w[(w.index.hour >= 9) & (w.index.hour < 18)].index.date) \
           .agg(day_mm=("rain_mm", "sum"), day_wet_hours=("wet", "sum"), day_temp=("temp_c", "mean"))
    day.index = pd.to_datetime(day.index).strftime("%Y-%m-%d")
    s = s.join(day, on="date")
    s["rainy_day"] = np.where(s.day_mm >= 1, 1.0, np.where(s.day_wet_hours == 0, 0.0, np.nan))
    # ---- session level
    h0 = s.start.dt.floor("h")
    s["wet_arrival"] = h0.map(w.wet).astype(float)
    s["wet_hours_2h"] = sum(((h0 + pd.Timedelta(hours=k)).map(w.wet).astype(float) for k in range(2)))
    s["any_wet_2h"] = (s.wet_hours_2h > 0).astype(float)

    # ---- controls
    close = pd.read_csv(os.path.join(DATA, "closing_times.csv")).set_index("date").close_h
    hr = s.start.dt.hour.clip(9, 20)
    ctrl = pd.concat([pd.get_dummies(hr, prefix="h", drop_first=True),
                      pd.get_dummies(s.start.dt.dayofweek, prefix="wd", drop_first=True),
                      pd.get_dummies(s.start.dt.month, prefix="m", drop_first=True),
                      pd.get_dummies(s.lang.map(lang_group), prefix="lang", drop_first=True)], axis=1).astype(float)
    ctrl["close_h"] = s.date.map(close).to_numpy()
    ctrl["hours_to_close"] = ctrl.close_h - (s.start.dt.hour + s.start.dt.minute / 60)
    ctrl["temp"] = s.day_temp

    y = np.log(s.dur.to_numpy())
    rows = []
    for name, col, label in [("A. Rainy day (>=1 mm, 09-18h) vs dry day", "rainy_day", "rainy vs dry day"),
                             ("B. Raining when the visitor arrives", "wet_arrival", "wet vs dry arrival hour"),
                             ("C. Rain in the 2 hours after arrival", "any_wet_2h", "any wet hour vs none")]:
        x = s[col].to_numpy()
        ok = ~np.isnan(x) & ~ctrl.isna().any(axis=1).to_numpy()
        raw1, raw0 = s.dur[ok & (x == 1)].mean(), s.dur[ok & (x == 0)].mean()
        b, se = cluster_ols(y[ok], np.column_stack([x[ok], ctrl.to_numpy()[ok]]), s.date.to_numpy()[ok])
        rows.append({"test": name, "comparison": label,
                     "visits_rain": int((ok & (x == 1)).sum()), "visits_dry": int((ok & (x == 0)).sum()),
                     "days_rain": s.date[ok & (x == 1)].nunique(), "days_dry": s.date[ok & (x == 0)].nunique(),
                     "avg_min_rain": raw1, "avg_min_dry": raw0, "raw_diff_min": raw1 - raw0,
                     "adj_effect_pct": (np.exp(b[0]) - 1) * 100, "adj_se_pct": se[0] * np.exp(b[0]) * 100,
                     "adj_effect_min": (np.exp(b[0]) - 1) * raw0, "z": b[0] / se[0]})
    # dose-response: mm on the day (bands)
    s["mm_band"] = pd.cut(s.day_mm, [-0.01, 0.0001, 1, 5, 1000], labels=["0 mm", "0-1 mm", "1-5 mm", ">5 mm"])
    dose = s.groupby("mm_band", observed=True).agg(visits=("dur", "size"), days=("date", "nunique"),
                                                   avg_min=("dur", "mean")).reset_index()
    D = pd.get_dummies(s.mm_band, drop_first=True).astype(float)
    ok = ~ctrl.isna().any(axis=1).to_numpy() & s.mm_band.notna().to_numpy()
    b, se = cluster_ols(y[ok], np.column_stack([D.to_numpy()[ok], ctrl.to_numpy()[ok]]), s.date.to_numpy()[ok])
    dose["adj_effect_pct"] = [0.0] + list((np.exp(b[:3]) - 1) * 100)
    dose["adj_se_pct"] = [0.0] + list(se[:3] * np.exp(b[:3]) * 100)
    res = pd.DataFrame(rows)
    res.round(3).to_csv(os.path.join(OUT, "weather_vs_duration.csv"), index=False)
    dose.round(3).to_csv(os.path.join(OUT, "weather_vs_duration_by_mm.csv"), index=False)
    pd.set_option("display.width", 220)
    print(res.round(2).to_string(index=False))
    print(dose.round(2).to_string(index=False))
    print("days:", s.date.nunique(), "rainy days:", (day.day_mm >= 1).sum(), "dry days:", (day.day_wet_hours == 0).sum())
    chart(res, dose)


def chart(res, dose):
    BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), dpi=200, gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(BG)
    ax = axs[0]; ax.set_facecolor(BG)
    lab = ["Rainy day\n(≥1 mm, 9–18h)", "Raining at\narrival", "Rain within 2h\nof arrival"]
    x = np.arange(len(res))
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.errorbar(x, res.adj_effect_pct, yerr=1.96 * res.adj_se_pct, fmt="o", color="#2a78d6", ecolor="#9cc0ec",
                capsize=4, ms=7, zorder=3)
    for i, r in res.iterrows():
        ax.annotate(f"{r.adj_effect_pct:+.1f}%\n({r.adj_effect_min:+.1f} min)", (i, r.adj_effect_pct), xytext=(10, 0),
                    textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_xlim(-0.5, 2.8)
    ax.set_ylabel("% change in visit duration vs dry", color=INK2, fontsize=9)
    ax.set_title("Effect of rain, with controls", loc="left", fontsize=10.5, color=INK)
    ax2 = axs[1]; ax2.set_facecolor(BG)
    x2 = np.arange(len(dose))
    ax2.axhline(0, color="#c3c2b7", lw=1)
    ax2.errorbar(x2, dose.adj_effect_pct, yerr=1.96 * dose.adj_se_pct, fmt="o", color="#eb6834", ecolor="#f4b393",
                 capsize=4, ms=7, zorder=3)
    ax2.set_xticks(x2); ax2.set_xticklabels([f"{b}\n{d} days" for b, d in zip(dose.mm_band, dose.days)], fontsize=8.5)
    ax2.set_title("By rain on the day (09–18h), vs 0 mm", loc="left", fontsize=10.5, color=INK)
    for a in axs:
        for sp_ in ("top", "right"): a.spines[sp_].set_visible(False)
        a.tick_params(colors=INK2, labelsize=8.5); a.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    fig.suptitle("Are visits longer when it rains? (Feb–Sep 2026, 95% intervals)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.17, wspace=0.22)
    fig.savefig(os.path.join(OUT, "weather_vs_duration.png"), facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
