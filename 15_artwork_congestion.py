"""
15_artwork_congestion.py - step 15: congestion at famous artworks vs total people in the room.

The paper's "congestion begets congestion" is about ARTWORKS: under congestion visitors engage with more
exhibits, including less prominent ones. Two checks for six famous works (14_build_artworks.py):
A. Slot level (5 min, 10:00-17:00, normal days): people at the artwork vs people in its room
   (08_build_crowding.py). Correlation, within-day correlation, and the elasticity
   (log people at artwork ~ log people in room, day and hour fixed effects): 1 = the artwork keeps the
   same share of the room; < 1 = as the room fills, a smaller share is at the famous work.
B. Visitor level (30% of sessions): room crowding when the visitor enters (5-min slot before entry,
   vs the room's own 90th percentile) and what they do in the room: play the famous stop,
   listen to it fully, minutes at it, number of different stops played in the room.
   Linear models with day fixed effects and entry-hour controls, errors clustered by day.
Outputs: output/artwork_congestion_slots.csv, output/artwork_congestion_visits.csv,
         output/artwork_congestion.png
Run from the model folder:  python 15_artwork_congestion.py
"""
import glob, os, importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cd = importlib.import_module("09_crowding_descriptives")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "output")
TEST = ("2026-05-18", "2026-06-11")
ART = [("sunflowers", "Sunflowers", "R05"), ("bedroom", "The Bedroom", "R05"), ("potato_eaters", "The Potato Eaters", "R03"),
       ("irises", "Irises", "R09"), ("wheatfield_crows", "Wheatfield with Crows", "R10"), ("almond_blossom", "Almond Blossom", "R12")]
BANDS = [0, 0.5, 0.75, 0.9, 1.1, np.inf]
LAB = ["<50%", "50-75%", "75-90%", "90-110%", ">110%"]


def load(pattern):
    return pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, pattern))))


def fe_fit(y, X, groups):
    b, se = cd.demean_fit(np.asarray(y, float), np.asarray(X, float), groups)
    return b, se


def main():
    room = load("crowding/crowding_*.csv")
    art = load("artworks/art_slots_*.csv")
    room = room[~room.date.between(*TEST)]
    p90 = room[(room.slot >= "11:00") & (room.slot < "15:00") & (room.est_people > 1)].groupby("zone").est_people.quantile(0.9)

    # days each artwork's stop was available (some stops are switched off, e.g. Almond Blossom in July 2026)
    daily = art.groupby(["artwork", "date"]).ag_people.sum()
    avail = {a: set(g.index.get_level_values(1)[g >= 0.25 * g.median()]) for a, g in daily.groupby(level=0)}

    # ---------------- A. slot level
    rowsA, binned = [], {}
    for short, name, z in ART:
        r = room[room.zone == z][["date", "slot", "est_people"]].rename(columns={"est_people": "room"})
        a = art[art.artwork == short][["date", "slot", "est_people"]].rename(columns={"est_people": "art"})
        x = r.merge(a, on=["date", "slot"], how="left").fillna({"art": 0})
        x = x[x.date.isin(avail[short])]
        x = x[(x.slot >= "10:00") & (x.slot < "17:00") & (x.room > 5)]
        x["h"] = x.slot.str[:2]
        corr = np.corrcoef(x.art, x.room)[0, 1]
        dm = x[["art", "room"]] - x.groupby("date")[["art", "room"]].transform("mean")
        corr_w = np.corrcoef(dm.art, dm.room)[0, 1]
        H = pd.get_dummies(x.h, drop_first=True).to_numpy(float)
        b, se = fe_fit(np.log(x.art + 1), np.column_stack([np.log(x.room + 1), H]), x.date.to_numpy())
        rel = x.room / p90[z]
        x["band"] = pd.cut(rel, BANDS, labels=LAB, right=False)
        share = (x.groupby("band", observed=True).art.sum() / x.groupby("band", observed=True).room.sum()).reindex(LAB)
        rowsA.append({"artwork": name, "room": z, "days": x.date.nunique(), "slots": len(x), "avg_people_at_artwork": x.art.mean(),
                      "avg_people_in_room": x.room.mean(), "avg_share_of_room": x.art.sum() / x.room.sum(),
                      "corr": corr, "corr_within_day": corr_w, "elasticity": b[0], "elasticity_se": se[0],
                      **{f"share_{l}": share[l] for l in LAB}})
        q = pd.qcut(x.room, 12, duplicates="drop")
        binned[name] = x.groupby(q, observed=True)[["room", "art"]].mean()
    A_ = pd.DataFrame(rowsA)
    A_.round(4).to_csv(os.path.join(OUT, "artwork_congestion_slots.csv"), index=False)

    # ---------------- B. visitor level
    v = load("artworks/art_visits_*.csv")
    v = v[~v.date.between(*TEST)]
    v["t_in"] = pd.to_datetime(v.t_in)
    A, di, p90a, _, _ = cd.load_crowding()
    from layout import ZONE_IDX
    rowsB = []
    for short, name, z in ART:
        x = v[v.zone == z].copy()
        x = x[x.date.isin(di) & x.date.isin(avail[short])]
        x["rel"] = cd.lookup(A, di, x.date, x.t_in, np.full(len(x), ZONE_IDX[z])) / p90a[ZONE_IDX[z]]
        x["band"] = cd.band(x.rel.to_numpy())
        x = x[(x.band >= 0) & (x.t_in.dt.hour.between(9, 17))]
        D = cd.band_dummies(x.band.to_numpy(), "").to_numpy()
        H = pd.get_dummies(x.t_in.dt.hour, drop_first=True).to_numpy(float)
        played = x[f"played_{short}"].to_numpy(float)
        outcomes = {"plays the artwork's stop": (played, np.ones(len(x), bool), "pp"),
                    "listens to it fully (if played)": ((x[f"eng_{short}"] == 3).to_numpy(float), played == 1, "pp"),
                    "minutes at the artwork (if played)": (x[f"listen_{short}"].to_numpy(float), played == 1, "min"),
                    "different stops played in the room": (x.n_distinct.to_numpy(float), np.ones(len(x), bool), "stops")}
        for oname, (y, mask, unit) in outcomes.items():
            mask = mask & ~np.isnan(y)
            b, se = fe_fit(y[mask], np.column_stack([D[mask], H[mask]]), x.date.to_numpy()[mask])
            base = y[mask][x.band.to_numpy()[mask] == cd.BASE].mean()
            for j, lab in enumerate([l for i, l in enumerate(LAB) if i != cd.BASE]):
                rowsB.append({"artwork": name, "outcome": oname, "unit": unit, "band": lab, "base_level": base,
                              "effect": b[j] * (100 if unit == "pp" else 1), "se": se[j] * (100 if unit == "pp" else 1),
                              "n": int(mask.sum())})
    B = pd.DataFrame(rowsB)
    B.round(4).to_csv(os.path.join(OUT, "artwork_congestion_visits.csv"), index=False)
    pd.set_option("display.width", 220)
    print(A_.round(3).to_string(index=False))
    print(B[B.band.isin(["90-110%", ">110%"])].round(3).to_string(index=False))
    chart(binned, A_)


def chart(binned, A_):
    BG, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.4), dpi=200)
    fig.patch.set_facecolor(BG)
    for ax, (name, b) in zip(axs.ravel(), binned.items()):
        r = A_[A_.artwork == name].iloc[0]
        ax.set_facecolor(BG)
        xs = np.linspace(0, b.room.max() * 1.05, 10)
        ax.plot(xs, xs * r.avg_share_of_room, color="#8a8984", lw=1.2, ls="--", label="same share of the room")
        ax.plot(b.room, b.art, "o-", color=BLUE, lw=2, ms=5, label="people at the artwork")
        ax.set_title(f"{name} ({r.room.replace('R0', 'Room ').replace('R1', 'Room 1')})", loc="left", fontsize=10.5, color=INK)
        ax.text(0.02, 0.95, f"elasticity {r.elasticity:.2f}", transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8); ax.grid(True, color=GRID, lw=0.8)
        ax.set_xlabel("people in the room", color=INK2, fontsize=8.5)
        ax.set_ylabel("people at the artwork", color=INK2, fontsize=8.5)
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Famous artworks: people at the artwork vs people in the room (5-min slots, 10:00–17:00, Feb–Sep 2026)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.08, hspace=0.42, wspace=0.25)
    fig.savefig(os.path.join(OUT, "artwork_congestion.png"), facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    main()
