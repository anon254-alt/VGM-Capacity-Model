"""
14_build_artworks.py - step 14: congestion at the most famous artworks (from ALL audio-guide sessions).

For a few famous artworks (audio-guide stops, English station names, which ~all languages use):
  Sunflowers, The Bedroom (Room 5), The Potato Eaters (Room 3), Irises (Room 9),
  Wheatfield with Crows (Room 10), Almond Blossom (Room 12)
A visitor is "at" an artwork from the start of its stop until the start of their next stop (max 15 min;
the last stop of a session: max 5 min). Same session filter and hourly scaling as 08_build_crowding.py.

Outputs per month (model/data/artworks/):
  art_slots_<YYYY-MM>.csv  : date, slot (5-min), artwork, ag_people, est_people (people at the artwork)
  art_visits_<YYYY-MM>.csv : one row per room visit to the artworks' rooms (30% of sessions):
                             date, sid, zone, t_in, t_out, start_hour, w, n_plays (stops played in the
                             visit), n_distinct, and per artwork in that room: played_<a>, eng_<a>
                             (max engagement 0-3), listen_<a> (minutes at the artwork)
Run from the Analysis folder (laptop):  python model/14_build_artworks.py --month 2026-08   (~40 s)
"""
import argparse, glob, os, sys, unicodedata
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
RAW = os.path.join(ANALYSIS, "..", "Raw Data")
AUDIO = sorted(glob.glob(os.path.join(RAW, "20260915 - Audiotour data VGM 20260201 - 20260916", "*.csv")))
COLS = ["Unique_sessie_id", "serverDate", "sessie_start", "sessie_end", "chosen_language",
        "station_name", "engagement_level", "engagement_stage", "Stop_start_time", "Tour_tijd", "Sessie_tijd"]
ZONES = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "EXH"]
ART = {"Sunflowers": ("sunflowers", "R05"), "The Bedroom": ("bedroom", "R05"),
       "The Potato Eaters": ("potato_eaters", "R03"), "Irises": ("irises", "R09"),
       "Wheatfield with Crows": ("wheatfield_crows", "R10"), "Almond Blossom": ("almond_blossom", "R12")}
SLOT, DAY0, DAY1 = 5, 8 * 60, 23 * 60
LAST_STOP_MIN, MAX_LISTEN, LAST_LISTEN, MIN_VISITS = 10, 15, 5, 3
FILE_MONTHS = {0: ("2026-02", "2026-04"), 1: ("2026-05", "2026-07"), 2: ("2026-08", "2026-09")}
SAMPLE = 0.30


def norm(s):
    return " ".join(unicodedata.normalize("NFC", str(s)).replace('"', "").split())


def occupancy(k, a, b, w, n_keys, n_slots):
    a = np.clip(a, DAY0, DAY1 - 1e-6) - DAY0
    b = np.clip(b, DAY0, DAY1 - 1e-6) - DAY0
    ok = b > a
    k, a, b, w = k[ok], a[ok], b[ok], w[ok]
    s0, s1 = (a // SLOT).astype(int), (b // SLOT).astype(int)
    n = s1 - s0 + 1
    rows = np.repeat(np.arange(len(a)), n)
    sl = s0[rows] + (np.arange(len(rows)) - np.repeat(np.cumsum(n) - n, n))
    ov = np.clip(np.minimum(b[rows], (sl + 1) * SLOT) - np.maximum(a[rows], sl * SLOT), 0, None) / SLOT
    out = np.zeros((n_slots, n_keys))
    np.add.at(out, (sl, k[rows]), ov * w[rows])
    return out


def mins(t):
    return (t.dt.hour * 60 + t.dt.minute + t.dt.second / 60).to_numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--month", required=True); a = ap.parse_args()
    path = AUDIO[[i for i, (m0, m1) in FILE_MONTHS.items() if m0 <= a.month <= m1][0]]
    mp = pd.read_csv(os.path.join(ANALYSIS, "station_room_mapping.csv"), encoding="utf-8-sig", dtype=str)
    zone_of = {k: ZONES.index(z) for k, z in zip(mp.station_name, mp.zone_id)}
    art_names = list(ART)
    with open(path, encoding="utf-8-sig") as f:
        hdr = f.readline().startswith("Unique_sessie_id")
    parts = []
    for c in pd.read_csv(path, sep=";", encoding="utf-8-sig", header=0 if hdr else None, names=None if hdr else COLS,
                         usecols=[0, 1, 2, 3, 5, 6, 8], dtype=str, chunksize=1_000_000):
        c.columns = ["sid", "date", "s_start", "s_end", "station", "eng", "t"]
        c = c[c.date.str.startswith(a.month)].drop_duplicates()
        st = c.station.map(norm)
        c["zone"] = st.map(lambda x: zone_of.get(x, -1))
        c["art"] = st.map(lambda x: art_names.index(x) if x in ART else -1)
        c["st"] = st
        parts.append(c[c.zone >= 0].drop(columns="station"))
    d = pd.concat(parts, ignore_index=True)
    for col in ["s_start", "s_end", "t"]:
        d[col] = pd.to_datetime(d[col], errors="coerce")
    d["eng"] = pd.to_numeric(d.eng, errors="coerce").fillna(0)
    d = d.dropna(subset=["s_start", "s_end", "t"]).sort_values(["sid", "t"]).reset_index(drop=True)
    d["sid"] = pd.factorize(d.sid)[0]
    # time at each stop = until the next stop of the session (capped)
    nxt = d.groupby("sid").t.shift(-1)
    lis = (nxt - d.t).dt.total_seconds() / 60
    d["listen"] = np.where(nxt.isna(), np.minimum(LAST_LISTEN, ((d.s_end - d.t).dt.total_seconds() / 60).clip(0)),
                           lis.clip(0, MAX_LISTEN))
    # room visits (same rules as 08)
    d["v"] = ((d.zone != d.zone.shift()) | (d.sid != d.sid.shift())).cumsum()
    v = d.groupby("v", sort=False).agg(sid=("sid", "first"), date=("date", "first"), zone=("zone", "first"),
                                        s_start=("s_start", "first"), s_end=("s_end", "first"),
                                        t_in=("t", "min"), last=("t", "max"), n_plays=("t", "size"))
    v["t_out"] = v.groupby("sid").t_in.shift(-1)
    lastv = v.t_out.isna()
    v.loc[lastv, "t_out"] = np.minimum(v.loc[lastv, "s_end"], v.loc[lastv, "last"] + pd.Timedelta(minutes=LAST_STOP_MIN))
    keep_sid = v.groupby("sid").zone.transform("size") >= MIN_VISITS
    v = v[keep_sid]
    good = set(v.sid)
    # hourly weights (from 08_build_crowding.py)
    wt = pd.read_csv(os.path.join(HERE, "data", "crowding", f"weights_{a.month}.csv"))
    fac = wt.set_index(["date", "start_hour"]).factor
    sess = v.groupby("sid").agg(date=("date", "first"), h=("s_start", lambda s: s.iloc[0].hour))
    sess["w"] = [fac.get((r.date, r.h), np.nan) for r in sess.itertuples()]

    # ---------- people at each artwork per 5-min slot
    x = d[(d.art >= 0) & d.sid.isin(good)].copy()
    x["w"] = x.sid.map(sess.w)
    x = x.dropna(subset=["w"])
    n_slots = (DAY1 - DAY0) // SLOT
    slots = [f"{(DAY0 + i * SLOT) // 60:02d}:{(DAY0 + i * SLOT) % 60:02d}" for i in range(n_slots)]
    rows = []
    for date, g in x.groupby("date"):
        t0 = mins(g.t)
        t1 = t0 + g.listen.to_numpy()
        raw = occupancy(g.art.to_numpy(), t0, t1, np.ones(len(g)), len(ART), n_slots)
        est = occupancy(g.art.to_numpy(), t0, t1, g.w.to_numpy(), len(ART), n_slots)
        for si in np.flatnonzero(raw.sum(1) > 0):
            for ai, name in enumerate(art_names):
                rows.append((date, slots[si], ART[name][0], round(raw[si, ai], 2), round(est[si, ai], 2)))
    out_dir = os.path.join(HERE, "data", "artworks"); os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows, columns=["date", "slot", "artwork", "ag_people", "est_people"]).to_csv(
        os.path.join(out_dir, f"art_slots_{a.month}.csv"), index=False)

    # ---------- room visits to the artworks' rooms (sample of sessions)
    rooms = {ZONES.index(z) for _, z in ART.values()}
    rng = np.random.default_rng(int(a.month.replace("-", "")))
    samp = set(rng.choice(sorted(good), int(len(good) * SAMPLE), replace=False))
    vv = v[v.zone.isin(rooms) & v.sid.isin(samp)].copy()
    dd = d[d.v.isin(vv.index)]
    vv["n_distinct"] = dd.groupby("v").st.nunique().reindex(vv.index)
    for name, (short, z) in ART.items():
        ai = art_names.index(name)
        g = dd[dd.art == ai].groupby("v").agg(eng=("eng", "max"), listen=("listen", "sum"))
        vv[f"played_{short}"] = np.where(vv.zone == ZONES.index(z), vv.index.isin(g.index).astype(float), np.nan)
        vv[f"eng_{short}"] = g.eng.reindex(vv.index)
        vv[f"listen_{short}"] = g.listen.reindex(vv.index)
    vv["start_hour"] = vv.s_start.dt.hour
    vv["w"] = vv.sid.map(sess.w)
    vv["zone"] = vv.zone.map(lambda i: ZONES[i])
    vv = vv.drop(columns=["s_start", "s_end", "last"])
    vv.to_csv(os.path.join(out_dir, f"art_visits_{a.month}.csv"), index=False)
    print(f"{a.month}: {len(good):,} sessions, {len(x):,} artwork plays, {len(vv):,} sampled room visits", file=sys.stderr)


if __name__ == "__main__":
    main()
