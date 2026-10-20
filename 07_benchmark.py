"""
07_benchmark.py - step 7: how well do we predict PEOPLE PER ROOM, and does the logit beat a simple
counted transition table at that level?

Three routing versions are simulated on the same validation days (the 25 normal days of
06_validate.py), with identical arrivals, stay times and random draws - only the next-room choice
differs:
  logit_calibrated   : the model (logit + calibration constant)
  logit_uncalibrated : logit without the calibration constant
  counted_table      : P(next room | current room) counted from the estimation sample (no other factors)

Metrics on people per room per 15-min slot (09:00-18:00) vs measured (audio guide, scaled):
  total inside, room daytime averages, room x slot correlation and error, peak error per room.
Also: a slot-by-slot table for one room (default Room 1).

Outputs: output/benchmark_summary.csv, output/benchmark_by_room.csv,
         output/room_slot_table_<room>.csv, output/room_slot_<room>.png
Run from the model folder:  python 07_benchmark.py [--room R01]   (about 12 minutes)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layout import ZONES, ZONE_IDX, ROOM_LABEL, N
from sim import Model, simulate_day, DAY_START, SLOT
import importlib
val = importlib.import_module("06_validate")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, PAR = (os.path.join(HERE, d) for d in ("data", "output", "params"))
TEST = ("2026-05-18", "2026-06-11")
REPS = 15
S0, S1 = 9 * 60, 18 * 60                     # slots compared: 09:00-17:45
VERSIONS = {"logit_calibrated": {}, "logit_uncalibrated": {"continue_shift": 0.0},
            "counted_table": {"router": "counted"}}


def counted_table(v):
    """counts[current room, next room or EXIT] from the estimation sample (+1 smoothing)."""
    e = v[(v.split == "est") & ~v.date.between(*TEST)].sort_values(["sid", "visit_no"])
    cur = e.zone.map(ZONE_IDX).to_numpy()
    nxt = e.groupby("sid").zone.shift(-1).map(ZONE_IDX).fillna(N).astype(int).to_numpy()
    counts = np.ones((N, N + 1))
    np.add.at(counts, (cur, nxt), 1)
    json.dump({"counts": counts.tolist(), "rows": ZONES, "cols": ZONES + ["EXIT"]},
              open(os.path.join(PAR, "counted_table.json"), "w"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--room", default="R01"); a = ap.parse_args()
    v = val.load_visits_for_measurement()
    v = v[v.n_visits >= 3]
    counted_table(v)
    m = Model()
    scans = m.arr.groupby("date").visitors.sum()
    days = sorted(set(v.date) & set(scans.index))
    normal = [d for d in days if m.open.loc[d, "EXH"] and not (TEST[0] <= d <= TEST[1])][3::7]
    i0, i1 = (S0 - DAY_START) // SLOT, (S1 - DAY_START) // SLOT
    slots = [f"{(S0 + k * SLOT) // 60:02d}:{(S0 + k * SLOT) % 60:02d}" for k in range(i1 - i0)]

    meas = np.stack([val.measured(v, scans, d)[i0:i1] for d in normal])          # (days, slots, 13)
    sims = {}
    for name, kw in VERSIONS.items():
        sims[name] = np.stack([simulate_day(d, REPS, model=m, seed=21, **kw)["occ"].mean(0)[i0:i1] for d in normal])
        print("done", name, file=sys.stderr)

    rows, by_room = [], []
    for name, s in sims.items():
        err = s - meas
        big = meas >= 20
        peak_m, peak_s = meas.max(1), s.max(1)                                      # (days, 13)
        rows.append({
            "version": name,
            "people_inside_measured": meas.sum(2).mean(), "people_inside_simulated": s.sum(2).mean(),
            "room_daily_avg_abs_error_pct": float(np.mean(np.abs(s.mean((0, 1)) / meas.mean((0, 1)) - 1)) * 100),
            "room_slot_correlation": float(np.corrcoef(meas.ravel(), s.ravel())[0, 1]),
            "room_slot_mean_abs_error_people": float(np.abs(err).mean()),
            "room_slot_abs_error_pct_cells_ge20": float((np.abs(err)[big] / meas[big]).mean() * 100),
            "room_peak_abs_error_pct": float((np.abs(peak_s - peak_m) / peak_m).mean() * 100),
        })
        for zi, z in enumerate(ZONES):
            by_room.append({"version": name, "room": ROOM_LABEL[z],
                            "measured_avg": meas[:, :, zi].mean(), "simulated_avg": s[:, :, zi].mean(),
                            "avg_error_pct": (s[:, :, zi].mean() / meas[:, :, zi].mean() - 1) * 100,
                            "peak_measured": peak_m[:, zi].mean(), "peak_simulated": peak_s[:, zi].mean()})
    summ = pd.DataFrame(rows).set_index("version")
    summ.round(2).to_csv(os.path.join(OUT, "benchmark_summary.csv"))
    pd.DataFrame(by_room).round(1).to_csv(os.path.join(OUT, "benchmark_by_room.csv"), index=False)

    # ---------- slot table for one room (calibrated logit)
    zi = ZONE_IDX[a.room]
    mr, sr = meas[:, :, zi], sims["logit_calibrated"][:, :, zi]
    cr = sims["counted_table"][:, :, zi]
    tab = pd.DataFrame({
        "slot": slots,
        "measured_avg": mr.mean(0),
        "predicted_avg": sr.mean(0),
        "avg_error": (sr - mr).mean(0),                         # bias: + = model too high on average
        "mean_abs_error": np.abs(sr - mr).mean(0),              # typical miss on a single day
        "mean_abs_error_pct_of_avg": np.abs(sr - mr).mean(0) / mr.mean(0) * 100,
        "counted_table_mean_abs_error": np.abs(cr - mr).mean(0),
    })
    tab.round(1).to_csv(os.path.join(OUT, f"room_slot_table_{a.room}.csv"), index=False)
    chart(tab, a.room, len(normal))
    print(summ.round(1).T.to_string())
    print(tab.round(1).to_string(index=False))


def chart(tab, room, n_days):
    BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x = np.arange(len(tab))
    ax.plot(x, tab.measured_avg, color="#2a78d6", lw=2, label="Measured (average of days)", zorder=3)
    ax.plot(x, tab.predicted_avg, color="#eb6834", lw=2, label="Predicted (average of days)", zorder=3)
    ax.fill_between(x, tab.predicted_avg - tab.mean_abs_error, tab.predicted_avg + tab.mean_abs_error,
                    color="#eb6834", alpha=0.15, lw=0, label="Predicted ± typical miss on a single day", zorder=2)
    ax.set_xticks(x[::4]); ax.set_xticklabels(tab.slot[::4])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9); ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_ylabel("People in the room (15-min average)", color=INK2)
    fig.suptitle(f"{ROOM_LABEL[room]}: predicted vs measured people per 15 minutes", x=0.07, ha="left",
                 fontsize=14, fontweight="bold", color=INK, y=0.97)
    ax.set_title(f"{n_days} validation days, Feb-Sep 2026 (09:00-18:00)", loc="left", fontsize=10, color=INK2, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.6)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.1)
    fig.savefig(os.path.join(OUT, f"room_slot_{room}.png"), facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
