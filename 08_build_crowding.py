"""
08_build_crowding.py - step 8: crowding measure from ALL audio-guide sessions (not the 15% sample).

For one month at a time (laptop memory):
  - map every stop to a room, merge consecutive stops in the same room into room visits
    (same rules as 01_build_room_visits.py)
  - keep sessions with >= 3 room visits
  - scale each session to all visitors by the audio-guide share of its START HOUR that day
    (factor = scanned visitors in that hour / sessions starting in that hour; day's factor if < 5 sessions)
  - visitors WITHOUT an audio guide spend less time in the rooms (visit-duration file: ratio
    non_ag_time_ratio ~0.88, params/audio_guide.json from 17_visit_duration_nps.py), so a session counts as
    1 audio-guide visitor + (factor - 1) others weighted by that ratio:  weight = 1 + (factor - 1) x ratio
  - people per room per 5-minute slot = sum over visits of (weight x minutes in the slot / 5)
Outputs (model/data/crowding/):
  crowding_<YYYY-MM>.csv : date, slot (HH:MM), zone, ag_people (raw audio-guide users), est_people
  weights_<YYYY-MM>.csv  : date, start_hour, sessions, scans, factor, weight
Run from the Analysis folder:  python model/08_build_crowding.py --month 2026-08
"""
import argparse, glob, json, os, sys, unicodedata
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
RAW = os.path.join(ANALYSIS, "..", "Raw Data")
AUDIO = sorted(glob.glob(os.path.join(RAW, "20260915 - Audiotour data VGM 20260201 - 20260916", "*.csv")))
COLS = ["Unique_sessie_id", "serverDate", "sessie_start", "sessie_end", "chosen_language",
        "station_name", "engagement_level", "engagement_stage", "Stop_start_time", "Tour_tijd", "Sessie_tijd"]
ZONES = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "EXH"]
SLOT, DAY0, DAY1 = 5, 8 * 60, 23 * 60
LAST_STOP_MIN, MIN_VISITS, MIN_SESS = 10, 3, 5
FILE_MONTHS = {0: ("2026-02", "2026-04"), 1: ("2026-05", "2026-07"), 2: ("2026-08", "2026-09")}


def norm(s):
    return " ".join(unicodedata.normalize("NFC", str(s)).replace('"', "").split())


def occupancy(zone, a, b, w, n_slots):
    a = np.clip(a, DAY0, DAY1 - 1e-6) - DAY0
    b = np.clip(b, DAY0, DAY1 - 1e-6) - DAY0
    k = b > a
    zone, a, b, w = zone[k], a[k], b[k], w[k]
    s0, s1 = (a // SLOT).astype(int), (b // SLOT).astype(int)
    n = s1 - s0 + 1
    rows = np.repeat(np.arange(len(a)), n)
    sl = s0[rows] + (np.arange(len(rows)) - np.repeat(np.cumsum(n) - n, n))
    ov = np.clip(np.minimum(b[rows], (sl + 1) * SLOT) - np.maximum(a[rows], sl * SLOT), 0, None) / SLOT
    out = np.zeros((n_slots, len(ZONES)))
    np.add.at(out, (sl, zone[rows]), ov * w[rows])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--month", required=True); a = ap.parse_args()
    fi = [i for i, (m0, m1) in FILE_MONTHS.items() if m0 <= a.month <= m1][0]
    path = AUDIO[fi]
    mp = pd.read_csv(os.path.join(ANALYSIS, "station_room_mapping.csv"), encoding="utf-8-sig", dtype=str)
    zone_of = {k: ZONES.index(z) for k, z in zip(mp.station_name, mp.zone_id)}
    with open(path, encoding="utf-8-sig") as f:
        hdr = f.readline().startswith("Unique_sessie_id")
    parts = []
    for c in pd.read_csv(path, sep=";", encoding="utf-8-sig", header=0 if hdr else None, names=None if hdr else COLS,
                         usecols=[0, 1, 2, 3, 5, 8], dtype=str, chunksize=1_000_000):
        c.columns = ["sid", "date", "s_start", "s_end", "station", "t"]
        c = c[c.date.str.startswith(a.month)].drop_duplicates()
        c["zone"] = c.station.map(lambda x: zone_of.get(norm(x), -1))
        parts.append(c[c.zone >= 0].drop(columns="station"))
    d = pd.concat(parts, ignore_index=True).drop_duplicates()
    for col in ["s_start", "s_end", "t"]:
        d[col] = pd.to_datetime(d[col], errors="coerce")
    d = d.dropna().sort_values(["sid", "t"])
    d["sid"] = pd.factorize(d.sid)[0]
    newv = (d.zone != d.zone.shift()) | (d.sid != d.sid.shift())
    d["v"] = newv.cumsum()
    v = d.groupby("v", sort=False).agg(sid=("sid", "first"), date=("date", "first"), zone=("zone", "first"),
                                        s_start=("s_start", "first"), s_end=("s_end", "first"),
                                        t_in=("t", "min"), last=("t", "max"))
    v["t_out"] = v.groupby("sid").t_in.shift(-1)
    last = v.t_out.isna()
    v.loc[last, "t_out"] = np.minimum(v.loc[last, "s_end"], v.loc[last, "last"] + pd.Timedelta(minutes=LAST_STOP_MIN))
    v = v[v.groupby("sid").zone.transform("size") >= MIN_VISITS]
    # hourly weights from ticket scans
    arr = pd.read_csv(os.path.join(HERE, "data", "arrivals_5min.csv"))
    arr = arr[arr.date.str.startswith(a.month)]
    arr["h"] = arr.slot.str[:2].astype(int)
    scans_h = arr.groupby(["date", "h"]).visitors.sum()
    scans_d = arr.groupby("date").visitors.sum()
    sess = v.groupby("sid").agg(date=("date", "first"), h=("s_start", lambda s: s.iloc[0].hour))
    cnt = sess.groupby(["date", "h"]).size().rename("sessions").reset_index()
    cnt["scans"] = [scans_h.get((r.date, r.h), 0) for r in cnt.itertuples()]
    dayf = (scans_d / sess.groupby("date").size()).rename("dayf")
    cnt = cnt.join(dayf, on="date")
    cnt["factor"] = np.where(cnt.sessions >= MIN_SESS, cnt.scans / cnt.sessions, cnt.dayf)
    out_dir = os.path.join(HERE, "data", "crowding"); os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(HERE, "params", "audio_guide.json")
    ratio = json.load(open(pj)).get("non_ag_time_ratio", 1.0) if os.path.exists(pj) else 1.0
    cnt["weight"] = 1 + (cnt.factor - 1).clip(lower=0) * ratio
    cnt.drop(columns="dayf").rename(columns={"h": "start_hour"}).to_csv(
        os.path.join(out_dir, f"weights_{a.month}.csv"), index=False)
    fac = cnt.set_index(["date", "h"]).weight
    sess["w"] = [fac.get((r.date, r.h), np.nan) for r in sess.itertuples()]
    v["w"] = v.sid.map(sess.w)
    v = v.dropna(subset=["w"])
    n_slots = (DAY1 - DAY0) // SLOT
    slots = [f"{(DAY0 + i * SLOT) // 60:02d}:{(DAY0 + i * SLOT) % 60:02d}" for i in range(n_slots)]
    rows = []
    for date, g in v.groupby("date"):
        tin = (g.t_in.dt.hour * 60 + g.t_in.dt.minute + g.t_in.dt.second / 60).to_numpy()
        tout = (g.t_out.dt.hour * 60 + g.t_out.dt.minute + g.t_out.dt.second / 60).to_numpy()
        z = g.zone.to_numpy()
        raw = occupancy(z, tin, tout, np.ones(len(g)), n_slots)
        est = occupancy(z, tin, tout, g.w.to_numpy(), n_slots)
        keep = raw.sum(1) > 0
        for si in np.flatnonzero(keep):
            for zi in range(len(ZONES)):
                rows.append((date, slots[si], ZONES[zi], round(raw[si, zi], 2), round(est[si, zi], 2)))
    pd.DataFrame(rows, columns=["date", "slot", "zone", "ag_people", "est_people"]).to_csv(
        os.path.join(out_dir, f"crowding_{a.month}.csv"), index=False)
    print(f"{a.month}: {sess.shape[0]:,} sessions, {len(v):,} room visits, {v.date.nunique()} days", file=sys.stderr)


if __name__ == "__main__":
    main()
