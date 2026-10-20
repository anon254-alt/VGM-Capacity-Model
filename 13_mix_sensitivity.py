"""
13_mix_sensitivity.py - step 13: does the visitor mix matter enough to need its own forecast?

In the model the visitor mix enters through the audio-guide LANGUAGE group (it changes stay times and
the chance to continue vs leave). For a scenario day we will not know the mix, so we test how much
crowding changes when the mix is replaced, keeping everything else identical (same arrivals, same
random draws, 20 runs per case):
  real        : the day's own mix (the reference)
  typical     : average mix of the same month and weekday (other days) - what a scenario would use
  short_mix   : the real day (Feb-Sep 2026) whose mix implies the SHORTEST visits
  long_mix    : the real day whose mix implies the LONGEST visits
Mix-implied visit length = sum over languages of share x that language's average visit duration.
Also: each language alone on one day (100% of visitors) to show which groups drive the effect.
Outputs: output/mix_sensitivity.csv (per day x case), output/mix_sensitivity_summary.csv,
         output/mix_by_language.csv
Run from the model folder:  python 13_mix_sensitivity.py   (~8 minutes)
"""
import os
import numpy as np
import pandas as pd

from layout import ZONES
from features import LANG_GROUPS, lang_group
from sim import Model, simulate_day, DAY_START, SLOT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
TEST = ("2026-05-18", "2026-06-11")
REPS = 20


def metrics(r):
    i0, i1 = (10 * 60 - DAY_START) // SLOT, (17 * 60 - DAY_START) // SLOT
    o = r["occ"].mean(0)
    return {"people_inside_avg": o[i0:i1].sum(1).mean(), "building_peak": o.sum(1).max(),
            "mean_visit_min": r["mean_visit_min"],
            **{f"avg_{z}": o[i0:i1, zi].mean() for zi, z in enumerate(ZONES)},
            **{f"peak_{z}": o[:, zi].max() for zi, z in enumerate(ZONES)}}


def main():
    m = Model()
    L = m.lang.reindex(columns=range(len(LANG_GROUPS)), fill_value=0)
    S = L.div(L.sum(axis=1), axis=0)
    # average visit duration per language (sessions with >= 3 room visits)
    v = pd.read_csv(os.path.join(DATA, "room_visits.csv"), usecols=["sid", "lang", "s_start", "t_out", "n_visits"],
                    parse_dates=["s_start", "t_out"])
    v = v[v.n_visits >= 3].groupby("sid").agg(lang=("lang", "first"), s=("s_start", "first"), e=("t_out", "max"))
    dur = ((v.e - v.s).dt.total_seconds() / 60).groupby(v.lang.map(lang_group)).mean().reindex(range(len(LANG_GROUPS)))
    index = S @ dur.to_numpy()
    days_all = [d for d in S.index if "2026-02-01" <= d <= "2026-09-13" and not (TEST[0] <= d <= TEST[1])
                and d in m.open.index and m.open.loc[d, "EXH"] and not m.arr[m.arr.date == d].empty]
    short_day, long_day = index[days_all].idxmin(), index[days_all].idxmax()
    days = days_all[4::22]
    print("short-visit mix day:", short_day, round(index[short_day], 1), " long-visit mix day:", long_day,
          round(index[long_day], 1), " test days:", days)
    dts = pd.to_datetime(pd.Series(S.index, index=S.index))
    rows = []
    for d in days:
        t = pd.Timestamp(d)
        same = [x for x in days_all if x != d and dts[x].month == t.month and dts[x].dayofweek == t.dayofweek]
        typical = S.loc[same].mean().to_numpy() if same else S.loc[[x for x in days_all if dts[x].month == t.month]].mean().to_numpy()
        for case, mix in [("real", None), ("typical", typical), ("short_mix", S.loc[short_day].to_numpy()),
                          ("long_mix", S.loc[long_day].to_numpy())]:
            r = simulate_day(d, REPS, model=m, seed=11, lang_mix=mix)
            mix_vec = S.loc[d].to_numpy() if mix is None else mix
            rows.append({"date": d, "case": case, "mix_index_min": float(mix_vec @ dur.to_numpy()),
                         "english_share": mix_vec[0], "n_visitors": r["n_visitors"], **metrics(r)})
        print("done", d)
    res = pd.DataFrame(rows)
    res.round(3).to_csv(os.path.join(OUT, "mix_sensitivity.csv"), index=False)
    # change vs the real mix
    ref = res[res.case == "real"].set_index("date")
    out = []
    for case in ["typical", "short_mix", "long_mix"]:
        x = res[res.case == case].set_index("date")
        rooms_avg = np.concatenate([(x[f"avg_{z}"] / ref[f"avg_{z}"] - 1).to_numpy() for z in ZONES])
        rooms_peak = np.concatenate([(x[f"peak_{z}"] / ref[f"peak_{z}"] - 1).to_numpy() for z in ZONES])
        out.append({"case": case,
                    "people_inside_change_pct": ((x.people_inside_avg / ref.people_inside_avg - 1) * 100).mean(),
                    "people_inside_abs_change_pct": ((x.people_inside_avg / ref.people_inside_avg - 1).abs() * 100).mean(),
                    "building_peak_change_pct": ((x.building_peak / ref.building_peak - 1) * 100).mean(),
                    "room_avg_abs_change_pct": np.abs(rooms_avg).mean() * 100,
                    "room_peak_abs_change_pct": np.abs(rooms_peak).mean() * 100,
                    "room_peak_max_abs_change_pct": np.abs(rooms_peak).max() * 100,
                    "mean_visit_change_min": (x.mean_visit_min - ref.mean_visit_min).mean()})
    summ = pd.DataFrame(out)
    summ.round(2).to_csv(os.path.join(OUT, "mix_sensitivity_summary.csv"), index=False)
    # each language alone, one day
    d0 = days[len(days) // 2]
    base = metrics(simulate_day(d0, 10, model=m, seed=11))
    lrows = []
    for g, name in enumerate(LANG_GROUPS):
        mix = np.zeros(len(LANG_GROUPS)); mix[g] = 1
        r = metrics(simulate_day(d0, 10, model=m, seed=11, lang_mix=mix))
        lrows.append({"language": name, "share_avg": S.loc[days_all].mean().iloc[g], "data_visit_min": dur.iloc[g],
                      "sim_visit_min": r["mean_visit_min"],
                      "people_inside_vs_real_mix_pct": (r["people_inside_avg"] / base["people_inside_avg"] - 1) * 100})
    lres = pd.DataFrame(lrows)
    lres.round(3).to_csv(os.path.join(OUT, "mix_by_language.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    print(res[["date", "case", "mix_index_min", "english_share", "people_inside_avg", "building_peak", "mean_visit_min"]].round(2).to_string(index=False))
    print(lres.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
