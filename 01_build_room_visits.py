"""
01_build_room_visits.py - step 1 of the crowding model.

Turns audio-guide stops into ROOM VISITS for a random sample of sessions:
  - every stop is mapped to a room (station_room_mapping.csv, in this folder; rebuild with build_station_mapping.py); unmapped stops are dropped
  - consecutive stops in the same room are merged into one visit
  - visit end = start of the next room's first stop; for the last visit
    = min(session end, last stop start + LAST_STOP_MIN)
Sample: sessions whose hashed id falls in the first SAMPLE_PCT % (deterministic, so re-runs give the
same sample). split = 'est' (estimation) or 'test' (held out for validation).

Output: model/data/room_visits.csv (one row per room visit)
Run from the Analysis folder:  python model/01_build_room_visits.py
"""
import glob, os, sys, unicodedata
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
RAW = os.path.join(ANALYSIS, "..", "Raw Data")
AUDIO_GLOB = os.path.join(RAW, "20260915 - Audiotour data VGM 20260201 - 20260916", "*.csv")
COLS = ["Unique_sessie_id", "serverDate", "sessie_start", "sessie_end", "chosen_language",
        "station_name", "engagement_level", "engagement_stage", "Stop_start_time", "Tour_tijd", "Sessie_tijd"]
SAMPLE_PCT, EST_PCT = 15, 10          # 15% of sessions sampled; first 10 points = estimation, rest = test
LAST_STOP_MIN = 10


def norm(s):
    return " ".join(unicodedata.normalize("NFC", str(s)).replace('"', "").split())


def main():
    mp = pd.read_csv(os.path.join(HERE, "station_room_mapping.csv"), encoding="utf-8-sig", dtype=str)
    zone_of = dict(zip(mp.station_name, mp.zone_id))
    parts = []
    for path in sorted(glob.glob(AUDIO_GLOB)):
        with open(path, encoding="utf-8-sig") as f:
            hdr = f.readline().startswith("Unique_sessie_id")
        for c in pd.read_csv(path, sep=";", encoding="utf-8-sig", header=0 if hdr else None,
                             names=None if hdr else COLS, usecols=[0, 1, 2, 3, 4, 5, 8],
                             dtype=str, chunksize=1_000_000):
            c.columns = ["sid", "date", "s_start", "s_end", "lang", "station", "t"]
            h = pd.util.hash_pandas_object(c.sid, index=False).values % 100
            c = c[h < SAMPLE_PCT].assign(bucket=h[h < SAMPLE_PCT])
            parts.append(c.drop_duplicates())
        print("read", os.path.basename(path), file=sys.stderr)
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    df["zone"] = df.station.map(lambda x: zone_of.get(norm(x)))
    df = df.dropna(subset=["zone"])
    for col in ["s_start", "s_end", "t"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["t", "s_start", "s_end"]).sort_values(["sid", "t"])

    # merge consecutive stops in the same room into one visit
    new_visit = (df.zone != df.groupby("sid").zone.shift()) | (df.sid != df.sid.shift())
    df["visit_no"] = new_visit.groupby(df.sid).cumsum()
    v = (df.groupby(["sid", "visit_no"], sort=False)
           .agg(date=("date", "first"), lang=("lang", "last"), bucket=("bucket", "first"),
                s_start=("s_start", "first"), s_end=("s_end", "first"), zone=("zone", "first"),
                t_in=("t", "min"), last_stop=("t", "max"), n_stops=("t", "size"))
           .reset_index())
    v["t_out"] = v.groupby("sid").t_in.shift(-1)
    last = v.t_out.isna()
    v.loc[last, "t_out"] = np.minimum(v.loc[last, "s_end"], v.loc[last, "last_stop"] + pd.Timedelta(minutes=LAST_STOP_MIN))
    v["is_last"] = last
    v["stay_min"] = (v.t_out - v.t_in).dt.total_seconds() / 60
    v["min_inside"] = (v.t_in - v.s_start).dt.total_seconds() / 60
    v["n_visits"] = v.groupby("sid").visit_no.transform("max")
    v["split"] = np.where(v.bucket < EST_PCT, "est", "test")
    v["sid"] = pd.util.hash_pandas_object(v.sid, index=False).values  # anonymous numeric id
    out = v[["sid", "split", "date", "lang", "s_start", "s_end", "visit_no", "n_visits", "zone",
             "t_in", "t_out", "stay_min", "min_inside", "n_stops", "is_last"]]
    out.to_csv(os.path.join(HERE, "data", "room_visits.csv"), index=False)
    print(f"{out.sid.nunique():,} sessions, {len(out):,} room visits "
          f"(est {out[out.split=='est'].sid.nunique():,} / test {out[out.split=='test'].sid.nunique():,})", file=sys.stderr)


if __name__ == "__main__":
    main()
