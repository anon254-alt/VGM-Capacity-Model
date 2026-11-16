"""
01_build_room_visits.py - step 1 of the crowding model.

Turns audio-guide stops into ROOM VISITS for a random sample of sessions:
  - every stop is mapped to a room (station_room_mapping.csv, in this folder; rebuild with build_station_mapping.py); unmapped stops are dropped
  - consecutive stops in the same room are merged into one visit
  - visit end = start of the next room's first stop; for the last visit
    = min(session end, last stop start + LAST_STOP_MIN)
Sample: sessions whose hashed id falls in the first SAMPLE_PCT % (deterministic, so re-runs give the
same sample). split = 'est' (estimation) or 'test' (held out for validation).

Data source (v1.4, 16 Nov 2026): the colleague's cleaned and labeled file (Analysis/Audiotour data - labeled/,
"label level 1" = room-presence evidence, rules v1.0). Default rule "listened":
  - drop stops the visitor only selected without listening (engagement_stage = "Selected Only")
  - drop stops labeled Unresolved, EXCEPT in the exhibition wing: every listened stop there is labeled Unresolved
    (its stations - "Vincent", "Theo", "Jo", "Prologue"... - are mapped to the wing in station_room_mapping.csv)
  Keeps ~70% of stops and 99.3% of sessions. Strong-only was tested and rejected: the exhibition wing and Room 6
  have no Strong (or Moderate) stops, and paths lose half their rooms.
  --rule labels --labels Strong,Moderate   keep only the given labels instead
  --raw                      use the original raw export (all stops, as in v1.0-v1.3)
  --file PATH                another labeled file with the same columns (e.g. a pre-filtered extract)

Output: model/data/room_visits.csv (one row per room visit)
Run:  python 01_build_room_visits.py
"""
import argparse, glob, os, sys, unicodedata
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
RAW = os.path.join(ANALYSIS, "..", "Raw Data")
AUDIO_GLOB = os.path.join(RAW, "20260915 - Audiotour data VGM 20260201 - 20260916", "*.csv")
LABELED = os.path.join(ANALYSIS, "Audiotour data - labeled", "Audiotour data - cleaned and labeled.csv")
COLS = ["Unique_sessie_id", "serverDate", "sessie_start", "sessie_end", "chosen_language",
        "station_name", "engagement_level", "engagement_stage", "Stop_start_time", "Tour_tijd", "Sessie_tijd"]
SAMPLE_PCT, EST_PCT = 15, 10          # 15% of sessions sampled; first 10 points = estimation, rest = test
LAST_STOP_MIN = 10


def norm(s):
    return " ".join(unicodedata.normalize("NFC", str(s)).replace('"', "").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="original raw export, all stops")
    ap.add_argument("--rule", default="listened", choices=["listened", "labels"])
    ap.add_argument("--labels", default="Strong", help="with --rule labels: labels to keep (comma-separated)")
    ap.add_argument("--file", default=LABELED, help="labeled file to read")
    a = ap.parse_args()
    keep = set(a.labels.split(","))
    files = sorted(glob.glob(AUDIO_GLOB)) if a.raw else [a.file]
    print("source:", "raw export (all stops)" if a.raw else (f"labeled file, labels {sorted(keep)}" if a.rule == "labels" else "labeled file, rule 'listened'"), file=sys.stderr)
    mp = pd.read_csv(os.path.join(HERE, "station_room_mapping.csv"), encoding="utf-8-sig", dtype=str)
    zone_of = dict(zip(mp.station_name, mp.zone_id))
    parts = []
    n_in = n_kept = 0
    for path in files:
        import gzip
        with (gzip.open(path, "rt", encoding="utf-8-sig") if path.endswith(".gz") else open(path, encoding="utf-8-sig")) as f:
            hdr = f.readline().startswith("Unique_sessie_id")
        use = [0, 1, 2, 3, 4, 5, 8] if a.raw else (lambda col: col in COLS[:6] + [COLS[7], COLS[8], "label level 1", "label"])
        for c in pd.read_csv(path, sep=";", encoding="utf-8-sig", header=0 if hdr else None,
                             names=None if hdr else COLS, usecols=use,
                             dtype=str, chunksize=1_000_000):
            c.columns = ["sid", "date", "s_start", "s_end", "lang", "station"] + ([] if a.raw else ["stage"]) + ["t"] + ([] if a.raw else ["label"])
            if not a.raw:
                n_in += len(c)
                if a.rule == "labels":
                    ok = c.label.isin(keep)
                else:
                    exh = c.station.map(lambda x: zone_of.get(norm(x))) == "EXH"
                    ok = (c.stage != "Selected Only") & ((c.label != "Unresolved") | exh)
                c = c[ok].drop(columns=["label", "stage"])
                n_kept += len(c)
            h = pd.util.hash_pandas_object(c.sid, index=False).values % 100
            c = c[h < SAMPLE_PCT].assign(bucket=h[h < SAMPLE_PCT])
            parts.append(c.drop_duplicates())
        print("read", os.path.basename(path), file=sys.stderr)
    if not a.raw:
        print(f"labeled file: kept {n_kept:,} of {n_in:,} stops ({n_kept / max(n_in, 1):.0%})", file=sys.stderr)
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
