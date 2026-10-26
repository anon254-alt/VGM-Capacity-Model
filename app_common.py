"""app_common.py - shared helpers for the Streamlit app (app.py, accuracy_page.py)."""
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from layout import ZONES
from sim import Model, simulate_day
import arrivals as A

# ---------- look (validated reference palette: series blue, neutral reference, sequential blue ramp)
BLUE, BLUE_DARK = "#2a78d6", "#3987e5"
GREY, GREY_DARK = "#8a8984", "#9b9a95"
CRIT = "#d03b3b"
SEQ_LIGHT = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_DARK = ["#1f2a38", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]
SHORT = {z: (f"Room {int(z[1:])}" if z != "EXH" else "Exhibition") for z in ZONES}
SLOT_RANGE = ("08:30", "21:30")


def dark_mode():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


@st.cache_resource
def get_model():
    return Model()


@st.cache_data
def ticket_table():
    return pd.read_csv(os.path.join(A.HERE, "data", "tickets_slot.csv"))


def day_tickets(date):
    t = ticket_table()
    return t[t.date == date][["slot", "tickets", "channel"]]


@st.cache_data
def typical_tickets(date, real_days):
    """Average tickets per slot (and channel) over 2026 days with data: same month and weekday if any,
    else the same weekday in all months. Returns (table, number of days used, same_month)."""
    t = ticket_table()
    t = t[t.date.isin(real_days)]
    d = pd.to_datetime(t.date)
    x = pd.Timestamp(date)
    same_wd = d.dt.dayofweek.eq(x.dayofweek)
    same_m = same_wd & d.dt.month.eq(x.month)
    sel, same_month = (same_m, True) if same_m.any() else (same_wd, False)
    g = t[sel.to_numpy()]
    out = g.groupby(["slot", "channel"], as_index=False).tickets.sum()
    out["tickets"] = (out.tickets / g.date.nunique()).round(1)
    return out[(out.slot >= "08:00") & (out.slot <= "21:45")], g.date.nunique(), same_month


@st.cache_data(show_spinner=False)
def run_sim(date, reps, closed, extra_open, arrivals_csv, seed, ag_share=None):
    """arrivals_csv: None = the day's real entries, else CSV text of slot,visitors.
    ag_share: audio-guide take-up for the scenario (0-1), None = as usual for that day."""
    arr = None if arrivals_csv is None else pd.read_csv(io.StringIO(arrivals_csv), dtype={"slot": str})
    r = simulate_day(date, reps, closed=list(closed), extra_open=list(extra_open), model=get_model(),
                     arrivals=arr, seed=seed, ag_share=ag_share)
    o = r["occ"]
    return {"slots": r["slots"], "mean": o.mean(0), "p10": np.percentile(o, 10, axis=0),
            "p90": np.percentile(o, 90, axis=0), "n_visitors": r["n_visitors"],
            "mean_visit_min": r["mean_visit_min"], "ag_share": r["ag_share"]}


def entries_15(arr_df):
    a = arr_df.copy()
    a["q"] = a.slot.str[:3] + (a.slot.str[3:5].astype(int) // 15 * 15).map("{:02d}".format)
    return a.groupby("q").visitors.sum()


