"""
accuracy_page.py - "Accuracy vs history" page of the app: how close the model is to measured crowding.

Uses two small precomputed files (16_accuracy_data.py):
  data/measured_15min.csv.gz  - measured people per room per 15 min (all audio-guide sessions, scaled by hour)
  data/simulated_15min.csv.gz - the model run on each day as it happened (real entries, rooms as they were)
A single day can also be re-simulated live.
"""
import os
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from layout import ZONES, ROOM_LABEL
from app_common import get_model, run_sim, dark_mode, BLUE, BLUE_DARK, GREY, GREY_DARK, SHORT

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = ("2026-05-18", "2026-06-11")            # exhibition wing closed: not used to fit the model
NEIGHBOURS = (("2026-04-20", "2026-05-17"), ("2026-06-12", "2026-07-09"))   # 4 normal weeks before and after
ORANGE = "#eb6834"
DAY0, DAY1 = "10:00", "17:00"                   # 'daytime' window for averages


@st.cache_data
def load_accuracy():
    meas = pd.read_csv(os.path.join(HERE, "data", "measured_15min.csv.gz"))
    sim = pd.read_csv(os.path.join(HERE, "data", "simulated_15min.csv.gz"))
    d = sim.merge(meas, on=["date", "slot", "zone"], how="left").fillna({"measured": 0.0})
    close = get_model().close
    h = d.slot.str[:2].astype(int) + d.slot.str[3:5].astype(int) / 60
    d = d[(h >= 9) & (h < d.date.map(close).fillna(17.75))]          # opening hours of each day
    d["period"] = np.where(d.date.between(*TEST), "closure", "normal")
    q = pd.read_csv(os.path.join(HERE, "data", "measured_days.csv")).set_index("date").reliable
    d.loc[~d.date.map(q).fillna(False).astype(bool), "period"] = "unreliable"   # audio-guide data gaps
    return d


def kpi(col, label, value, note=None):
    col.metric(label, value)
    if note:
        col.caption(note)


def summary(d):
    """Headline accuracy numbers for a set of days."""
    day = d[(d.slot >= DAY0) & (d.slot < DAY1)]
    inside = day.groupby(["date", "slot"])[["measured", "simulated"]].sum().groupby("date").mean()
    ra = day.groupby(["date", "zone"])[["measured", "simulated"]].mean()
    ra = ra[ra.measured >= 10]
    pk = d.groupby(["date", "zone"])[["measured", "simulated"]].max()
    pk = pk[pk.measured >= 10]
    big = d[d.measured >= 20]
    return {
        "days": d.date.nunique(),
        "inside_meas": inside.measured.mean(), "inside_sim": inside.simulated.mean(),
        "inside_bias": (inside.simulated.mean() / inside.measured.mean() - 1) * 100,
        "inside_day_err": (inside.simulated / inside.measured - 1).abs().mean() * 100,
        "room_avg_err": (ra.simulated / ra.measured - 1).abs().mean() * 100,
        "slot_corr": np.corrcoef(d.measured, d.simulated)[0, 1],
        "slot_mae": (d.simulated - d.measured).abs().mean(),
        "slot_err_pct": ((big.simulated - big.measured).abs() / big.measured).mean() * 100,
        "peak_err": (pk.simulated / pk.measured - 1).abs().mean() * 100,
        "peak_bias": (pk.simulated / pk.measured - 1).mean() * 100,
    }


def lines_chart(x, meas, pred, height=300, title=None, band=None):
    dark = dark_mode()
    blue, grey = (BLUE_DARK, GREY_DARK) if dark else (BLUE, GREY)
    f = go.Figure()
    if band is not None:
        f.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(band[1]) + list(band[0])[::-1], fill="toself",
                               fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), hoverinfo="skip",
                               name="Model range (10–90% of runs)"))
    f.add_trace(go.Scatter(x=x, y=meas, name="Measured", line=dict(color=grey, width=2, dash="dot"),
                           hovertemplate="Measured %{x}: %{y:.0f}<extra></extra>"))
    f.add_trace(go.Scatter(x=x, y=pred, name="Model", line=dict(color=blue, width=2),
                           hovertemplate="Model %{x}: %{y:.0f}<extra></extra>"))
    f.update_layout(height=height, margin=dict(l=10, r=10, t=36 if title else 10, b=10), hovermode="x unified",
                    title=dict(text=title, font=dict(size=13)) if title else None, showlegend=title is None,
                    legend=dict(orientation="h", y=1.1), xaxis=dict(dtick=8 if title else 4, tickangle=0),
                    yaxis=dict(rangemode="tozero"))
    return f


def accuracy_page():
    st.title("How accurate is the model?")
    st.caption("The model is run on each past day as it happened (real entries, rooms open or closed as they were) and "
               "compared with measured crowding: people per room per 15 minutes from all audio-guide sessions, scaled to "
               "all visitors by hour. 'Measured' is itself an estimate (only ~40% of visitors use the audio guide).")
    try:
        d = load_accuracy()
    except FileNotFoundError:
        st.error("Accuracy data not found. Run `python 16_accuracy_data.py` in the model folder first.")
        st.stop()

    # ================================================================ overall
    normal, closure = d[d.period == "normal"], d[d.period == "closure"]
    s, sc = summary(normal), summary(closure)
    st.subheader("On average")
    st.caption(f"{s['days']} normal days, Feb–Sep 2026, opening hours. The model was fitted on a 10% sample of visitors on "
               f"these days, so this is mostly in-sample. The {sc['days']} days with the exhibition wing closed "
               f"({TEST[0][8:]}/05–{TEST[1][8:]}/06) were not used to fit it: a fairer, out-of-sample test. "
               f"{d[d.period == 'unreliable'].date.nunique()} days with audio-guide data gaps are left out "
               f"({', '.join(sorted(d[d.period == 'unreliable'].date.unique()))}).")
    k = st.columns(4)
    kpi(k[0], "People in the building (10–17h)", f"{s['inside_sim']:,.0f}",
        f"model, vs {s['inside_meas']:,.0f} measured ({s['inside_bias']:+.1f}%)")
    kpi(k[1], "Error on a room's daily average", f"{s['room_avg_err']:.0f}%",
        f"{sc['room_avg_err']:.0f}% in the out-of-sample closure period")
    kpi(k[2], "Typical miss per room & 15 min", f"{s['slot_mae']:.0f} people",
        f"{s['slot_err_pct']:.0f}% in slots with 20+ people")
    kpi(k[3], "Error on a room's daily peak", f"{s['peak_err']:.0f}%",
        f"peaks {'under' if s['peak_bias'] < 0 else 'over'}-estimated by {abs(s['peak_bias']):.0f}% on average")
    with st.expander("What these numbers mean"):
        st.markdown(f"""
- **People in the building:** average people inside between 10:00 and 17:00, model vs measured, over all normal days.
  On a single day the model is typically {s['inside_day_err']:.0f}% off.
- **Error on a room's daily average:** for each room and day, the model's 10–17h average vs the measured one
  (rooms with at least 10 people); the typical absolute difference.
- **Typical miss per room & 15 min:** the average absolute difference in one room in one 15-minute slot. Correlation
  between model and measured across all rooms and slots: {s['slot_corr']:.2f}.
- **Peaks:** the highest 15-minute value of the day in each room. The model averages many simulated days, which smooths
  peaks; a single measured day is noisier, so peaks are under-estimated.
""")

    t1, t2, t3, t4 = st.tabs(["By room", "Through the day", "Day by day", "Table"])
    day = normal[(normal.slot >= DAY0) & (normal.slot < DAY1)]
    by_room = day.groupby("zone")[["measured", "simulated"]].mean().reindex(ZONES)
    pk = normal.groupby(["date", "zone"])[["measured", "simulated"]].max().groupby("zone").mean().reindex(ZONES)
    dark = dark_mode()
    blue, grey = (BLUE_DARK, GREY_DARK) if dark else (BLUE, GREY)
    with t1:
        labels = [ROOM_LABEL[z] for z in ZONES]
        f = go.Figure()
        f.add_trace(go.Bar(y=labels, x=by_room.measured, name="Measured", orientation="h", marker_color=grey,
                           hovertemplate="%{y}<br>Measured: %{x:.0f}<extra></extra>"))
        f.add_trace(go.Bar(y=labels, x=by_room.simulated, name="Model", orientation="h", marker_color=blue,
                           text=[f"{(a / b - 1) * 100:+.0f}%" for a, b in zip(by_room.simulated, by_room.measured)],
                           textposition="outside", textfont=dict(size=12), hovertemplate="%{y}<br>Model: %{x:.0f}<extra></extra>"))
        f.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, height=560, margin=dict(l=10, r=40, t=30, b=10),
                        yaxis=dict(autorange="reversed"), xaxis=dict(title="average people, 10:00–17:00"),
                        legend=dict(orientation="h", y=1.06))
        st.plotly_chart(f, use_container_width=True, theme="streamlit")
        st.caption("Average people per room between 10:00 and 17:00 over all normal days. Labels = model vs measured.")
    with t2:
        choice = st.selectbox("Room", ["Whole building"] + ZONES,
                              format_func=lambda z: z if z == "Whole building" else ROOM_LABEL[z])
        x = normal if choice == "Whole building" else normal[normal.zone == choice]
        prof = x.groupby(["date", "slot"])[["measured", "simulated"]].sum().groupby("slot").mean()
        prof = prof[prof.index <= "21:00"]
        st.plotly_chart(lines_chart(prof.index, prof.measured, prof.simulated, height=380), use_container_width=True,
                        theme="streamlit")
        err = (prof.simulated - prof.measured)
        worst = err.abs().idxmax()
        st.caption(f"Average over all normal days. Largest gap at {worst}: model {prof.simulated[worst]:.0f} vs measured "
                   f"{prof.measured[worst]:.0f}. Evening slots only include Fridays (late opening).")
    with t3:
        di = d[(d.period != "unreliable") & (d.slot >= DAY0) & (d.slot < DAY1)].groupby(["date", "slot", "period"])[["measured", "simulated"]].sum() \
            .groupby(["date", "period"]).mean().reset_index()
        f = go.Figure()
        lim = [0, max(di.measured.max(), di.simulated.max()) * 1.05]
        f.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#c3c2b7", width=1, dash="dash"),
                               hoverinfo="skip", name="Model = measured"))
        for per, col, name in [("normal", blue, "Normal days"), ("closure", ORANGE, "Exhibition wing closed (out-of-sample)")]:
            q = di[di.period == per]
            f.add_trace(go.Scatter(x=q.measured, y=q.simulated, mode="markers", name=name, text=q.date,
                                   marker=dict(color=col, size=8, line=dict(color="#fcfcfb", width=1)),
                                   hovertemplate="%{text}<br>Measured %{x:.0f}<br>Model %{y:.0f}<extra></extra>"))
        f.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08),
                        xaxis=dict(title="measured people in the building (avg 10–17h)", range=lim),
                        yaxis=dict(title="model", range=lim))
        st.plotly_chart(f, use_container_width=True, theme="streamlit")
        st.caption("Each dot is a day. Dots on the dashed line = the model got the day's average crowding exactly right.")
    with t4:
        tab = pd.DataFrame({"Room": [ROOM_LABEL[z] for z in ZONES],
                            "Measured average": by_room.measured.values, "Model average": by_room.simulated.values,
                            "Average error": (by_room.simulated / by_room.measured - 1).values * 100,
                            "Measured peak (avg day)": pk.measured.values, "Model peak (avg day)": pk.simulated.values,
                            "Peak error": (pk.simulated / pk.measured - 1).values * 100,
                            "Typical miss per 15 min": normal.assign(e=(normal.simulated - normal.measured).abs())
                            .groupby("zone").e.mean().reindex(ZONES).values})
        st.dataframe(tab, hide_index=True, use_container_width=True, height=36 * 14 + 3, column_config={
            c: st.column_config.NumberColumn(format="%+.1f%%" if "error" in c else "%.0f") for c in tab.columns[1:]})
        st.download_button("Download accuracy by room (CSV)", tab.round(2).to_csv(index=False), "model_accuracy_by_room.csv",
                           "text/csv")
        st.download_button("Download all days, rooms and slots (CSV)", d.round(2).to_csv(index=False),
                           "model_vs_measured_all_days.csv", "text/csv")

    # ================================================================ out-of-sample check
    st.divider()
    st.subheader("Out-of-sample check: exhibition wing closed, 18 May – 11 June 2026")
    cdays = sorted(closure.date.unique())
    st.caption(f"The exhibition wing was closed for {len(cdays)} days ({pd.Timestamp(cdays[0]):%d %b} – "
               f"{pd.Timestamp(cdays[-1]):%d %b %Y}). These days were kept out when the model was fitted, so they test "
               "whether it predicts a layout change it has not seen. (The wing was also closed 1–12 Feb and 7–16 Sep "
               "2026; those days were used to fit and calibrate the model, so they count as normal, in-sample days here.)")
    k = st.columns(4)
    kpi(k[0], "People in the building (10–17h)", f"{sc['inside_sim']:,.0f}",
        f"model, vs {sc['inside_meas']:,.0f} measured ({sc['inside_bias']:+.1f}%); in-sample {s['inside_bias']:+.1f}%")
    kpi(k[1], "Error on a room's daily average", f"{sc['room_avg_err']:.0f}%", f"in-sample days: {s['room_avg_err']:.0f}%")
    kpi(k[2], "Typical miss per room & 15 min", f"{sc['slot_mae']:.0f} people", f"in-sample days: {s['slot_mae']:.0f} people")
    kpi(k[3], "Error on a room's daily peak", f"{sc['peak_err']:.0f}%", f"in-sample days: {s['peak_err']:.0f}%")

    # does the model get the CHANGE caused by the closure right? closure vs 4 normal weeks before and after
    nb = d[(d.period == "normal") & (d.date.between(*NEIGHBOURS[0]) | d.date.between(*NEIGHBOURS[1]))]
    win = lambda x: x[(x.slot >= DAY0) & (x.slot < DAY1)].groupby("zone")[["measured", "simulated"]].mean().reindex(ZONES)
    a_cl, a_nb = win(closure), win(nb)
    rooms = [z for z in ZONES if z != "EXH"]
    chg = ((a_cl / a_nb - 1) * 100).loc[rooms]
    tot_cl = closure[(closure.slot >= DAY0) & (closure.slot < DAY1) & (closure.zone != "EXH")].groupby(["date", "slot"])[["measured", "simulated"]].sum().mean()
    tot_nb = nb[(nb.slot >= DAY0) & (nb.slot < DAY1) & (nb.zone != "EXH")].groupby(["date", "slot"])[["measured", "simulated"]].sum().mean()
    tchg = (tot_cl / tot_nb - 1) * 100
    corr = np.corrcoef(chg.measured, chg.simulated)[0, 1]
    o1, o2 = st.tabs(["Change caused by the closure", "Through the day"])
    with o1:
        labels = [ROOM_LABEL[z] for z in rooms]
        f = go.Figure()
        f.add_trace(go.Bar(y=labels, x=chg.measured, name="Measured change", orientation="h", marker_color=grey,
                           hovertemplate="%{y}<br>Measured: %{x:+.1f}%<extra></extra>"))
        f.add_trace(go.Bar(y=labels, x=chg.simulated, name="Model change", orientation="h", marker_color=blue,
                           hovertemplate="%{y}<br>Model: %{x:+.1f}%<extra></extra>"))
        f.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
        f.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, height=520, margin=dict(l=10, r=20, t=30, b=10),
                        yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.07),
                        xaxis=dict(title="change in average people (10–17h) vs the 4 normal weeks before and after", ticksuffix="%"))
        st.plotly_chart(f, use_container_width=True, theme="streamlit")
        st.caption(f"With the wing closed, people in the 12 collection rooms went up {tchg.measured:+.1f}% (measured); "
                   f"the model predicted {tchg.simulated:+.1f}%. Room by room, the correlation between measured and model changes is "
                   f"{corr:.2f}: the model gets the level of each room right (see the numbers above) but "
                   + ("only partly" if corr > 0.2 else "not reliably") + " which rooms gain or lose. Measured changes "
                   "also reflect season and artwork moves during those weeks.")
    with o2:
        choice = st.selectbox("Room", ["Whole building"] + ZONES, key="oos_room",
                              format_func=lambda z: z if z == "Whole building" else ROOM_LABEL[z])
        x = closure if choice == "Whole building" else closure[closure.zone == choice]
        prof = x.groupby(["date", "slot"])[["measured", "simulated"]].sum().groupby("slot").mean()
        prof = prof[prof.index <= "21:00"]
        st.plotly_chart(lines_chart(prof.index, prof.measured, prof.simulated, height=360), use_container_width=True,
                        theme="streamlit", key="oos_chart")
        st.caption("Average over the closure days. To look at one of them, pick it under 'Check a specific day' below: "
                   + ", ".join(f"{pd.Timestamp(x):%d %b}" for x in cdays) + ".")

    # ================================================================ one day
    st.divider()
    st.subheader("Check a specific day")
    days = sorted(d.date.unique())
    c1, c2, c3 = st.columns([1.2, 1, 1])
    pick = c1.date_input("Day", value=dt.date.fromisoformat("2026-08-14"), min_value=dt.date.fromisoformat(days[0]),
                         max_value=dt.date.fromisoformat(days[-1]), format="DD/MM/YYYY", key="acc_day")
    date = pick.isoformat()
    if date not in days:
        st.warning("No measured data for this day (museum closed or missing data). Pick another day.")
        st.stop()
    reps = c2.select_slider("Simulation runs", [20, 40, 80], value=20, key="acc_reps")
    rerun = c3.button("Re-simulate this day", help="Runs the model now (10–40 s) and adds the range across runs.")
    x = d[d.date == date]
    slots = sorted(x.slot.unique())
    M = x.pivot(index="slot", columns="zone", values="measured").reindex(index=slots, columns=ZONES).fillna(0)
    P = x.pivot(index="slot", columns="zone", values="simulated").reindex(index=slots, columns=ZONES).fillna(0)
    band = None
    if rerun or st.session_state.get("acc_live") == (date, reps):
        st.session_state["acc_live"] = (date, reps)
        m = get_model()
        closed = tuple(z for z, o in zip(ZONES, m.open_zones(date)) if not o)
        with st.spinner("Simulating the day…"):
            r = run_sim(date, reps, closed, (), None, 0)
        idx = [r["slots"].index(sl) for sl in slots]
        P = pd.DataFrame(r["mean"][idx], index=slots, columns=ZONES)
        band = (pd.DataFrame(r["p10"][idx], index=slots, columns=ZONES), pd.DataFrame(r["p90"][idx], index=slots, columns=ZONES))

    one = x.assign(simulated=[P.loc[sl, z] for sl, z in zip(x.slot, x.zone)])
    sd = summary(one)
    per_day = d[(d.slot >= DAY0) & (d.slot < DAY1)].groupby(["date", "zone"])[["measured", "simulated"]].mean()
    per_day = per_day[per_day.measured >= 10]
    ok_days = d[d.period != "unreliable"].date.unique()
    per_day = per_day[per_day.index.get_level_values(0).isin(ok_days)]
    day_err = (per_day.simulated / per_day.measured - 1).abs().groupby("date").mean() * 100
    better = (day_err > sd["room_avg_err"]).mean() * 100
    k = st.columns(4)
    kpi(k[0], "People in the building (10–17h)", f"{sd['inside_sim']:,.0f}",
        f"model, vs {sd['inside_meas']:,.0f} measured ({sd['inside_bias']:+.1f}%)")
    kpi(k[1], "Error on room averages", f"{sd['room_avg_err']:.0f}%", f"more accurate than {better:.0f}% of days")
    kpi(k[2], "Typical miss per room & 15 min", f"{sd['slot_mae']:.0f} people")
    kpi(k[3], "Error on room peaks", f"{sd['peak_err']:.0f}%",
        f"peaks {'under' if sd['peak_bias'] < 0 else 'over'}-estimated by {abs(sd['peak_bias']):.0f}%")
    if d[d.date == date].period.iloc[0] == "closure":
        st.info("This day is in the exhibition-wing closure period, which was not used to fit the model.")
    elif d[d.date == date].period.iloc[0] == "unreliable":
        st.warning("Audio-guide data for this day is incomplete, so the 'measured' line is not reliable.")

    st.markdown(f"<span style='color:{blue};font-weight:700'>━━</span> Model &nbsp;&nbsp; "
                f"<span style='color:{grey};font-weight:700'>┅┅</span> Measured"
                + ("&nbsp;&nbsp; <span style='background:rgba(42,120,214,0.18);padding:0 10px'>&nbsp;</span> "
                   "Model range, 10–90% of runs" if band is not None else ""), unsafe_allow_html=True)
    tot = lines_chart(slots, M.sum(1), P.sum(1), height=300, title="Whole building")
    st.plotly_chart(tot, use_container_width=True, theme="streamlit", key="acc_total")
    cols = st.columns(3)
    for i, z in enumerate(ZONES):
        b = None if band is None else (band[0][z], band[1][z])
        cols[i % 3].plotly_chart(lines_chart(slots, M[z], P[z], height=230, title=ROOM_LABEL[z], band=b),
                                 use_container_width=True, theme="streamlit", key=f"acc_{z}")
    out = pd.DataFrame([(sl, ROOM_LABEL[z], M.loc[sl, z], P.loc[sl, z]) for sl in slots for z in ZONES],
                       columns=["slot", "room", "measured", "model"]).round(1)
    st.download_button("Download this day (CSV)", out.to_csv(index=False), f"model_vs_measured_{date}.csv", "text/csv")
