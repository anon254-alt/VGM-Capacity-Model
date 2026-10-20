"""
11_build_tickets.py - step 11: tickets sold per time slot, and when ticket holders actually arrive.

Every visitor needs a timed ticket (15-min slots), also Friends/partner passes and desk sales, so
arrivals = tickets sold x show-up rate, shifted by how early/late people come vs their slot.
From the raw ticket file (2026 ticket dates, cancelled lines removed):
  data/tickets_slot.csv : date, slot (HH:MM, booked), channel, tickets, visitors (= scanned people)
      channel: regular (VGM direct), reseller, group (groups, schools, tour operators, guided tours),
               friday (Vincent op Vrijdag programme), other
  data/arrival_lags.csv : slot_hour, lag_min (scan time - booked slot start, whole minutes,
                          clipped to -180..+240), visitors   (same-day scans only)
Run from the Analysis folder (laptop, needs the raw file):  python model/11_build_tickets.py   (~1 min)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TICKETS = os.path.join(HERE, "..", "..", "Raw Data", "20260915 - Ticket sales and visitor data VGM.csv")
COLS = ["Ticket_date", "Ticket_time", "Scan_date", "Scan_time", "Is_scanned", "Is_cancelled", "Product_funnel",
        "Product_type_extra", "Number_of_tickets", "Number_of_visitors"]
GROUP = {"Groepsbezoek", "Groepsbezoek - Intern", "Touroperator", "Basisonderwijs", "Voortgezet onderwijs", "MBO",
         "Kinderworkshop"}


def channel(c):
    ch = np.select([c.Product_funnel == "Vincent op Vrijdag", c.Product_funnel.isin(GROUP) | (c.Product_type_extra == "Rondleidingen"),
                    c.Product_funnel == "Reseller", c.Product_funnel == "Regulier"],
                   ["friday", "group", "reseller", "regular"], "other")
    return ch


def main():
    slots, lags = [], []
    for c in pd.read_csv(TICKETS, sep=";", encoding="utf-8-sig", chunksize=1_000_000, dtype=str, usecols=COLS):
        c = c[(c.Ticket_date >= "2026-01-01") & (c.Is_cancelled == "0")].copy()
        c["tickets"] = pd.to_numeric(c.Number_of_tickets, errors="coerce").fillna(0)
        c["visitors"] = np.where(c.Is_scanned == "1", pd.to_numeric(c.Number_of_visitors, errors="coerce").fillna(0), 0)
        c["slot"] = c.Ticket_time.str[:5]
        c["channel"] = channel(c)
        slots.append(c.groupby(["Ticket_date", "slot", "channel"])[["tickets", "visitors"]].sum())
        s = c[(c.Is_scanned == "1") & (c.Scan_date == c.Ticket_date)]
        lag = (pd.to_datetime(s.Scan_time.str[:8], format="%H:%M:%S", errors="coerce")
               - pd.to_datetime(s.Ticket_time.str[:8], format="%H:%M:%S", errors="coerce")).dt.total_seconds() / 60
        lags.append(pd.DataFrame({"slot_hour": s.Ticket_time.str[:2].astype(int), "channel": s.channel,
                                  "lag_min": np.floor(lag.clip(-180, 240)), "visitors": s.visitors})
                    .dropna().groupby(["slot_hour", "channel", "lag_min"]).visitors.sum())
    t = pd.concat(slots).groupby(level=[0, 1, 2]).sum().reset_index().rename(columns={"Ticket_date": "date"})
    l = pd.concat(lags).groupby(level=[0, 1, 2]).sum().reset_index()
    out = os.path.join(HERE, "data")
    t.to_csv(os.path.join(out, "tickets_slot.csv"), index=False)
    l.to_csv(os.path.join(out, "arrival_lags.csv"), index=False)
    print(len(t), "slot rows,", t.date.nunique(), "days;", int(l.visitors.sum()), "scans with lag")


if __name__ == "__main__":
    main()
