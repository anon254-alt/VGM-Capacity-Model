"""
02_build_arrivals.py - step 2: visitors entering per 5 minutes, from the ticket-scan file.
Output: model/data/arrivals_5min.csv  (date, slot 'HH:MM', visitors)
Run from the Analysis folder:  python model/02_build_arrivals.py
Optional: --start 2025-09-01 --out arrivals_5min_from2025.csv  (longer history, e.g. for the visit-duration file)
"""
import argparse
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TICKETS = os.path.join(HERE, "..", "..", "Raw Data", "20260915 - Ticket sales and visitor data VGM.csv")
ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2026-01-01"); ap.add_argument("--out", default="arrivals_5min.csv")
args = ap.parse_args()
parts = []
for c in pd.read_csv(TICKETS, sep=";", encoding="utf-8-sig", chunksize=1_000_000, dtype=str,
                     usecols=["Scan_date", "Scan_time", "Is_scanned", "Number_of_visitors"]):
    c = c[(c.Is_scanned == "1") & (c.Scan_date >= args.start)]
    t = pd.to_datetime(c.Scan_time.str[:8], format="%H:%M:%S", errors="coerce")
    c = c.assign(slot=t.dt.floor("5min").dt.strftime("%H:%M"),
                 v=pd.to_numeric(c.Number_of_visitors, errors="coerce"))
    parts.append(c.groupby(["Scan_date", "slot"]).v.sum())
a = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
a.columns = ["date", "slot", "visitors"]
a.to_csv(os.path.join(HERE, "data", args.out), index=False)
print(len(a), "rows;", a.date.nunique(), "days")
