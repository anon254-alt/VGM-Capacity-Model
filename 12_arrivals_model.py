"""
12_arrivals_model.py - step 12: fit and validate the tickets -> arrivals conversion (arrivals.py).

Fit (Feb-Sep 2026, excluding every 4th day, which is kept for testing):
  show-up rate by channel; arrival-lag distribution (5-min steps, -180..+240 min) per booked slot hour.
Test 1 - arrivals: for the held-out days, predict people entering per 5 / 15 minutes from that day's
         tickets per slot and compare with the real scans.
Test 2 - crowding: simulate a sample of held-out normal days with real vs predicted arrivals (same random
         draws) and compare people per room (daytime average and peak).
Outputs: params/arrivals.json, output/arrivals_validation.csv, output/arrivals_validation_days.csv,
         output/arrivals_validation.png
Run from the model folder:  python 12_arrivals_model.py   (~4 minutes)
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import arrivals as A
from layout import ZONES
from sim import Model, simulate_day, DAY_START, SLOT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, PAR = (os.path.join(HERE, d) for d in ("data", "output", "params"))
PERIOD = ("2026-02-01", "2026-09-13")
TEST = ("2026-05-18", "2026-06-11")
LAG_BINS = list(range(-180, 241, 5))


def fit(t, l, train_days):
    tt = t[t.date.isin(train_days)]
    g = tt.groupby("channel")[["tickets", "visitors"]].sum()
    show = (g.visitors / g.tickets).round(4).to_dict()
    show["all"] = round(float(tt.visitors.sum() / tt.tickets.sum()), 4)
    l = l.copy()
    l["bin"] = (l.lag_min // 5 * 5).clip(-180, 240).astype(int)
    l["h"] = l.slot_hour.clip(9, 20)
    pmf = {}
    for h, x in list(l.groupby("h")) + [("all", l)]:
        p = x.groupby("bin").visitors.sum().reindex(LAG_BINS, fill_value=0)
        pmf[str(h)] = (p / p.sum()).round(6).tolist()
    return {"show_up": show, "lag_bins": LAG_BINS, "lag_pmf": pmf, "open_min": 540,
            "fitted_on": f"{PERIOD[0]} to {PERIOD[1]}, 3 of every 4 days"}


def main():
    t = pd.read_csv(os.path.join(DATA, "tickets_slot.csv"))
    t = t[t.date.between(*PERIOD)]
    l = pd.read_csv(os.path.join(DATA, "arrival_lags.csv"))      # lags are pooled over all 2026 days
    arr = pd.read_csv(os.path.join(DATA, "arrivals_5min.csv"))
    days = sorted(set(t.date) & set(arr.date))
    test_days = days[3::4]
    train_days = [d for d in days if d not in test_days]
    p = fit(t, l, train_days)
    json.dump(p, open(os.path.join(PAR, "arrivals.json"), "w"))

    # ---------- test 1: arrivals curves on held-out days
    rows, curves = [], []
    for d in test_days:
        pred = A.to_counts(A.expected(t[t.date == d][["slot", "tickets", "channel"]], p)).set_index("slot").visitors
        real = arr[arr.date == d].set_index("slot").visitors
        idx = sorted(set(pred.index) | set(real.index))
        pr, re = pred.reindex(idx, fill_value=0), real.reindex(idx, fill_value=0)
        q = lambda s: s.groupby([x[:3] + f"{int(x[3:]) // 15 * 15:02d}" for x in s.index]).sum()
        p15, r15 = q(pr), q(re)
        day15 = (r15.index >= "09:00") & (r15.index < "17:00")
        rows.append({"date": d, "tickets": t[t.date == d].tickets.sum(), "real_visitors": re.sum(), "pred_visitors": pr.sum(),
                     "total_error_pct": (pr.sum() / re.sum() - 1) * 100,
                     "corr_5min": np.corrcoef(pr, re)[0, 1], "corr_15min": np.corrcoef(p15, r15)[0, 1],
                     "mae_15min": np.abs(p15 - r15)[day15].mean(), "mean_15min": r15[day15].mean(),
                     "peak15_real": r15.max(), "peak15_pred": p15.max()})
        curves.append(pd.DataFrame({"date": d, "slot": p15.index, "pred": p15.values, "real": r15.values}))
    res = pd.DataFrame(rows)
    res["mae_15min_pct"] = res.mae_15min / res.mean_15min * 100
    res.round(3).to_csv(os.path.join(OUT, "arrivals_validation_days.csv"), index=False)
    cur = pd.concat(curves)

    # ---------- test 2: crowding with real vs predicted arrivals
    m = Model()
    normal = [d for d in test_days if d in m.open.index and m.open.loc[d, "EXH"] and not (TEST[0] <= d <= TEST[1])][::4]
    i0, i1 = (9 * 60 - DAY_START) // SLOT, (18 * 60 - DAY_START) // SLOT
    sim_rows = []
    for d in normal:
        pred = A.to_counts(A.expected(t[t.date == d][["slot", "tickets", "channel"]], p))
        o_real = simulate_day(d, 15, model=m, seed=5)["occ"].mean(0)[i0:i1]
        o_pred = simulate_day(d, 15, model=m, seed=5, arrivals=pred)["occ"].mean(0)[i0:i1]
        for zi, z in enumerate(ZONES):
            sim_rows.append({"date": d, "zone": z, "avg_real_arr": o_real[:, zi].mean(), "avg_pred_arr": o_pred[:, zi].mean(),
                             "peak_real_arr": o_real[:, zi].max(), "peak_pred_arr": o_pred[:, zi].max()})
        sim_rows.append({"date": d, "zone": "TOTAL", "avg_real_arr": o_real.sum(1).mean(), "avg_pred_arr": o_pred.sum(1).mean(),
                         "peak_real_arr": o_real.sum(1).max(), "peak_pred_arr": o_pred.sum(1).max()})
    sres = pd.DataFrame(sim_rows)
    rooms = sres[sres.zone != "TOTAL"]
    tot = sres[sres.zone == "TOTAL"]
    summary = {
        "test_days": len(res), "sim_days": len(normal),
        "show_up_all": p["show_up"]["all"], "show_up_by_channel": p["show_up"],
        "daily_total_abs_error_pct": float(res.total_error_pct.abs().mean()),
        "daily_total_bias_pct": float(res.total_error_pct.mean()),
        "corr_15min_median": float(res.corr_15min.median()),
        "mae_15min_pct_of_mean": float(res.mae_15min.sum() / res.mean_15min.sum() * 100),
        "peak15_abs_error_pct": float((np.abs(res.peak15_pred / res.peak15_real - 1)).mean() * 100),
        "sim_people_inside_diff_pct": float((tot.avg_pred_arr / tot.avg_real_arr - 1).mean() * 100),
        "sim_room_avg_abs_diff_pct": float((np.abs(rooms.avg_pred_arr / rooms.avg_real_arr - 1)).mean() * 100),
        "sim_room_peak_abs_diff_pct": float((np.abs(rooms.peak_pred_arr / rooms.peak_real_arr - 1)).mean() * 100),
        "sim_building_peak_diff_pct": float((tot.peak_pred_arr / tot.peak_real_arr - 1).mean() * 100),
    }
    pd.Series(summary).to_csv(os.path.join(OUT, "arrivals_validation.csv"), header=["value"])
    sres.round(2).to_csv(os.path.join(OUT, "arrivals_validation_sim.csv"), index=False)
    print(json.dumps(summary, indent=1, default=float))
    chart(cur, res)


def chart(cur, res):
    BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=200)
    fig.patch.set_facecolor(BG)
    avg = cur[(cur.slot >= "08:30") & (cur.slot < "18:00")].groupby("slot")[["real", "pred"]].mean()
    ex = res.sort_values("real_visitors").iloc[len(res) * 3 // 4].date        # a busy example day
    one = cur[(cur.date == ex) & (cur.slot >= "08:30") & (cur.slot < "18:00")].set_index("slot")
    for ax, d, title in [(axs[0], avg, f"Average of {len(res)} held-out days"), (axs[1], one, f"One day: {ex}")]:
        ax.set_facecolor(BG)
        x = np.arange(len(d))
        ax.plot(x, d.real, color="#2a78d6", lw=2, label="Real entries (ticket scans)")
        ax.plot(x, d.pred, color="#eb6834", lw=2, ls="--", label="Predicted from tickets sold")
        ax.set_xticks(x[::4]); ax.set_xticklabels(d.index[::4], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel("People entering per 15 min", color=INK2, fontsize=9)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(True, axis="y", color=GRID, lw=0.8)
    axs[0].legend(frameon=False, fontsize=8.5, loc="lower center")
    fig.suptitle("Tickets sold → people entering: predicted vs real (held-out days, Feb–Sep 2026)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.12, wspace=0.15)
    fig.savefig(os.path.join(OUT, "arrivals_validation.png"), facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
