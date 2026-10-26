"""
06_validate.py - step 6: does the simulation reproduce measured crowding?

Measured occupancy = audio-guide room visits of the sampled sessions (>= 3 room visits), scaled up to
all scanned visitors of the day (same visit definition the model learned from).

Tests
  A. Normal days (exhibition wing open, outside the closure-test period): every 7th day.
     Compare simulated vs measured average people per room and hour (10:00-17:00).
  B. Closure test, 18 May - 11 Jun 2026 (exhibition wing closed; never used in estimation):
     simulate the days as they were (wing closed). Like-for-like comparison, per 1,000 visitors:
       measured change  = measured closure days vs measured normal days
       predicted change = simulated closure days vs simulated normal days
     Also reported: the pure closure effect on the same days (simulated closed vs open, same random draws).

Outputs: output/validation_summary.json, output/validation_rooms.csv,
         output/validation_normal_days.png, output/validation_closure.png
Run from the model folder:  python 06_validate.py   (about 10 minutes)
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from layout import ZONES, ROOM_LABEL, N
from sim import Model, simulate_day, occupancy, DAY_START, SLOT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
TEST_PERIOD = ("2026-05-18", "2026-06-11")
REPS_NORMAL, REPS_CLOSURE = 15, 10
H0, H1 = 10, 17                                   # daytime window for comparisons
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_MEAS, C_PRED = "#2a78d6", "#eb6834"


_ARR_H = None


SAMPLE_RATE = 0.1503          # share of audio-guide sessions (3+ rooms) in room_visits.csv (01_build_room_visits.py)
_pj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params", "audio_guide.json")
AG_RATIO = json.load(open(_pj)).get("non_ag_time_ratio", 1.0) if os.path.exists(_pj) else 1.0


def load_visits_for_measurement():
    return pd.read_csv(os.path.join(DATA, "room_visits.csv"), parse_dates=["s_start", "t_in", "t_out"],
                       usecols=["sid", "split", "date", "visit_no", "n_visits", "zone", "s_start", "t_in", "t_out"])


def _arrivals_by_hour():
    global _ARR_H
    if _ARR_H is None:
        a = pd.read_csv(os.path.join(DATA, "arrivals_5min.csv"))
        a["h"] = a.slot.str[:2].astype(int)
        _ARR_H = a.groupby(["date", "h"]).visitors.sum()
    return _ARR_H


def measured(v, scans, date, min_sessions=5):
    """Measured people per zone per slot. Each sampled session is scaled up by the audio-guide share
    of its START HOUR that day (scanned visitors in that hour / sampled sessions starting then):
    morning visitors take the audio guide more often, so one factor per day would overstate mornings.
    Hours with fewer than min_sessions sampled sessions use the day's overall factor."""
    d = v[v.date == date]
    n_slots = (23 * 60 - DAY_START) // SLOT
    zi = d.zone.map({z: i for i, z in enumerate(ZONES)}).to_numpy()
    tin = (d.t_in.dt.hour * 60 + d.t_in.dt.minute + d.t_in.dt.second / 60).to_numpy()
    tout = (d.t_out.dt.hour * 60 + d.t_out.dt.minute + d.t_out.dt.second / 60).to_numpy()
    sess = d.drop_duplicates("sid").set_index("sid").s_start.dt.hour
    n_h = sess.value_counts()
    arr = _arrivals_by_hour()
    day_factor = scans[date] / len(sess)
    fac = {h: (arr.get((date, h), 0) / n if n >= min_sessions else day_factor) for h, n in n_h.items()}
    # each sampled session = 1/SAMPLE_RATE audio-guide visitors + the rest, who spend less time in the rooms
    # (non_ag_time_ratio, see 08_build_crowding.py)
    fac = {h: f if f <= 1 / SAMPLE_RATE else 1 / SAMPLE_RATE + (f - 1 / SAMPLE_RATE) * AG_RATIO for h, f in fac.items()}
    w = d.sid.map(sess).map(fac).to_numpy()
    occ = np.zeros((n_slots, N))
    for f in np.unique(w):
        sel = w == f
        occ += occupancy(zi[sel], tin[sel], tout[sel], n_slots) * f
    return occ


def window(occ_slots):
    """Average over the daytime window -> (13,) people per zone."""
    i0 = (H0 * 60 - DAY_START) // SLOT
    i1 = (H1 * 60 - DAY_START) // SLOT
    return occ_slots[i0:i1].mean(0)


def main():
    m = Model()
    v = load_visits_for_measurement()
    v = v[v.n_visits >= 3]
    scans = m.arr.groupby("date").visitors.sum()
    exh_open = m.open["EXH"]
    days = sorted(set(v.date) & set(scans.index))
    normal = [d for d in days if exh_open.get(d, False) and not (TEST_PERIOD[0] <= d <= TEST_PERIOD[1])][3::7]
    closure = [d for d in days if TEST_PERIOD[0] <= d <= TEST_PERIOD[1]]

    # ---------- A. normal days
    rows, cells = [], []
    for d in normal:
        sim = simulate_day(d, REPS_NORMAL, model=m).get("occ").mean(0)
        mea = measured(v, scans, d)
        i0, i1 = (H0 * 60 - DAY_START) // SLOT, (H1 * 60 - DAY_START) // SLOT
        for h in range(H0, H1):                    # hourly cells per room
            a, b = (h * 60 - DAY_START) // SLOT, (h * 60 - DAY_START) // SLOT + 4
            for zi, z in enumerate(ZONES):
                cells.append((d, h, z, mea[a:b, zi].mean(), sim[a:b, zi].mean()))
        rows.append((d, *window(mea), *window(sim)))
        print("normal", d, file=sys.stderr)
    cells = pd.DataFrame(cells, columns=["date", "hour", "zone", "measured", "simulated"])
    by_room = cells.groupby("zone")[["measured", "simulated"]].mean()
    big = cells[cells.measured >= 20]
    stats_normal = {
        "days": normal, "reps_per_day": REPS_NORMAL,
        "room_hour_cells": int(len(cells)),
        "correlation_room_hour": float(cells.measured.corr(cells.simulated)),
        "mape_room_hour_cells_ge20_people": float((abs(big.simulated - big.measured) / big.measured).mean()),
        "total_inside_measured": float(cells.groupby(["date", "hour"]).measured.sum().mean()),
        "total_inside_simulated": float(cells.groupby(["date", "hour"]).simulated.sum().mean()),
    }

    # ---------- B. closure test (exhibition wing closed)
    clos_rows = []
    for d in closure:
        as_was = simulate_day(d, REPS_CLOSURE, model=m, seed=11)
        cf_open = simulate_day(d, REPS_CLOSURE, model=m, seed=11, extra_open=("EXH",))
        mea = measured(v, scans, d)
        k = 1000 / scans[d]
        clos_rows.append((d, *(window(mea) * k), *(window(as_was["occ"].mean(0)) * k),
                          *(window(cf_open["occ"].mean(0)) * k)))
        print("closure", d, file=sys.stderr)
    cols = [f"meas_{z}" for z in ZONES] + [f"sim_{z}" for z in ZONES] + [f"cf_{z}" for z in ZONES]
    C = pd.DataFrame(clos_rows, columns=["date"] + cols).set_index("date")
    Nn = pd.DataFrame(rows, columns=["date"] + [f"meas_{z}" for z in ZONES] + [f"sim_{z}" for z in ZONES]).set_index("date")
    k_norm = (1000 / scans.reindex(Nn.index)).to_numpy()[:, None]
    norm_meas_1000 = (Nn[[f"meas_{z}" for z in ZONES]].to_numpy() * k_norm).mean(0)
    norm_sim_1000 = (Nn[[f"sim_{z}" for z in ZONES]].to_numpy() * k_norm).mean(0)
    clos_meas_1000 = C[[f"meas_{z}" for z in ZONES]].to_numpy().mean(0)
    clos_sim_1000 = C[[f"sim_{z}" for z in ZONES]].to_numpy().mean(0)
    clos_cf_1000 = C[[f"cf_{z}" for z in ZONES]].to_numpy().mean(0)
    rooms = pd.DataFrame({
        "room": [ROOM_LABEL[z] for z in ZONES],
        "normal_measured": by_room.reindex(ZONES).measured.to_numpy(),
        "normal_simulated": by_room.reindex(ZONES).simulated.to_numpy(),
        "closure_measured_per1000": clos_meas_1000,
        "closure_simulated_per1000": clos_sim_1000,
        "measured_change_pct": 100 * (clos_meas_1000 / norm_meas_1000 - 1),
        "predicted_change_pct": 100 * (clos_sim_1000 / norm_sim_1000 - 1),
        "predicted_change_same_days_pct": 100 * (clos_sim_1000 / clos_cf_1000 - 1),
    }, index=ZONES)
    rooms.round(2).to_csv(os.path.join(OUT, "validation_rooms.csv"))
    rr = rooms.drop(index="EXH")
    stats_closure = {
        "days": closure, "reps_per_day": REPS_CLOSURE,
        "correlation_measured_vs_predicted_change": float(rr.measured_change_pct.corr(rr.predicted_change_pct)),
        "mean_abs_error_change_pct_points": float((rr.measured_change_pct - rr.predicted_change_pct).abs().mean()),
        "correlation_room_level_closure_occupancy": float(np.corrcoef(clos_meas_1000[:12], clos_sim_1000[:12])[0, 1]),
    }
    json.dump({"normal_days": stats_normal, "closure_test": stats_closure},
              open(os.path.join(OUT, "validation_summary.json"), "w"), indent=2)
    print(json.dumps({"normal_days": {k: v for k, v in stats_normal.items() if k != "days"},
                      "closure_test": {k: v for k, v in stats_closure.items() if k != "days"}}, indent=1))
    print(rooms.round(1).to_string())
    charts(cells, rr)


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)


def charts(cells, rr):
    # A. measured vs simulated per room (normal days, daytime average)
    by = cells.groupby("zone")[["measured", "simulated"]].mean().reindex(ZONES)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200); fig.patch.set_facecolor(BG); style(ax)
    x = np.arange(len(ZONES)); w = 0.38
    ax.bar(x - w / 2 - 0.01, by.measured, w, color=C_MEAS, label="Measured (audio guide, scaled)", zorder=3)
    ax.bar(x + w / 2 + 0.01, by.simulated, w, color=C_PRED, label="Simulated (model)", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([z.replace("R0", "R").replace("R", "Room ") if z != "EXH" else "Exhib." for z in ZONES],
                                         rotation=0, fontsize=8.5)
    ax.set_ylabel("Average people in the room, 10:00-17:00", color=INK2)
    fig.suptitle("The model reproduces where people are on normal days", x=0.07, ha="left", fontsize=14,
                 fontweight="bold", color=INK, y=0.97)
    ax.set_title(f"Average over {cells.date.nunique()} validation days, Feb-Sep 2026", loc="left", fontsize=10,
                 color=INK2, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.1)
    fig.savefig(os.path.join(OUT, "validation_normal_days.png"), facecolor=BG); plt.close(fig)

    # B. closure: measured vs predicted change per room
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200); fig.patch.set_facecolor(BG); style(ax)
    x = np.arange(len(rr))
    ax.bar(x - w / 2 - 0.01, rr.measured_change_pct, w, color=C_MEAS, label="Measured change", zorder=3)
    ax.bar(x + w / 2 + 0.01, rr.predicted_change_pct, w, color=C_PRED, label="Predicted change (model)", zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"Room {int(z[1:])}" for z in rr.index], fontsize=8.5)
    ax.set_ylabel("Change in people per 1,000 visitors (%)", color=INK2)
    fig.suptitle("Exhibition wing closed: predicted vs measured shift of visitors", x=0.07, ha="left",
                 fontsize=14, fontweight="bold", color=INK, y=0.97)
    ax.set_title("18 May - 11 Jun 2026 vs normal days (closure period not used to estimate the model); 10:00-17:00", loc="left",
                 fontsize=10, color=INK2, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.1)
    fig.savefig(os.path.join(OUT, "validation_closure.png"), facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
