"""
app.py - VGM crowding scenario tool (Streamlit web app on top of the simulation).

Run locally (from the model folder):
    pip install -r requirements.txt
    streamlit run app.py
Then open the address it prints (usually http://localhost:8501).

Put online: push the model folder (code + params/ + the small data files listed in README) to GitHub
and deploy app.py on Streamlit Community Cloud (or any server that runs `streamlit run app.py`).
The big room_visits.csv is NOT needed by the app.
"""
import io
import os
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from layout import ZONES, ROOM_LABEL
from sim import Model, simulate_day, ag_share_by_hour
import arrivals as A

st.set_page_config(page_title="VGM crowding tool", page_icon="🖼️", layout="wide")

from app_common import *  # noqa: F401,F403  (helpers, colours, cached model and simulation)


def scenario_page():
    # =====================================================================  sidebar: inputs
    m = get_model()
    st.sidebar.title("Scenario inputs")
    try:
        start_date = dt.date.fromisoformat(st.query_params.get("date", "2026-08-14"))   # ?date=YYYY-MM-DD in the URL
    except ValueError:
        start_date = dt.date(2026, 8, 14)
    date = st.sidebar.date_input("Day", value=start_date, min_value=dt.date(2026, 1, 1),
                                 max_value=dt.date(2027, 12, 31), format="DD/MM/YYYY")
    date = date.isoformat()
    has_real = not m.arr[m.arr.date == date].empty
    real_days = tuple(sorted(set(m.arr.date)))
    typ_t, typ_n, typ_month = typical_tickets(date, real_days)
    real_t = day_tickets(date) if has_real else typ_t
    weekday = pd.Timestamp(date).day_name()
    if has_real:
        st.sidebar.caption(f"{weekday}. Real data available: {int(real_t.tickets.sum()):,} tickets sold.")
    else:
        st.sidebar.caption(f"{weekday}. No data for this day: the model uses that weekday's typical closing time and "
                           f"visitor mix. Typical sales = average of {typ_n} {weekday}s"
                           f"{' in the same month' if typ_month else ''} in 2026 ({typ_t.tickets.sum():,.0f} tickets).")

    st.sidebar.subheader("1. Tickets sold")
    modes = (["Real sales of the day", "Scale the day's sales"] if has_real else
             ["Typical sales for this day", "Scale typical sales"]) + ["Same number in every slot", "Type tickets per slot"]
    mode = st.sidebar.radio("How to set ticket sales", modes, label_visibility="collapsed")

    tickets = None                                  # None = real entries (scans) of the day
    if mode == "Typical sales for this day":
        tickets = typ_t
    elif mode in ("Scale the day's sales", "Scale typical sales"):
        f = st.sidebar.slider("Factor on every slot", 0.5, 1.5, 1.10, 0.05, format="×%.2f")
        tickets = real_t.assign(tickets=real_t.tickets * f)
    elif mode == "Same number in every slot":
        n = st.sidebar.number_input("Tickets per 15-min slot", 0, 600, 170, 10)
        c1, c2 = st.sidebar.columns(2)
        all_slots = [f"{h:02d}:{mm:02d}" for h in range(9, 22) for mm in (0, 15, 30, 45)]
        first = c1.selectbox("First slot", all_slots, index=0)
        last = c2.selectbox("Last slot", all_slots, index=all_slots.index("16:30"))
        tickets = A.flat_tickets(n, first, last)
    elif mode == "Type tickets per slot":
        if real_t.empty:
            start = A.flat_tickets(170)
        else:
            start = real_t.groupby("slot", as_index=False).tickets.sum().round(0)
            start = start[(start.slot >= "08:00") & (start.slot <= "21:45")]
        up = st.sidebar.file_uploader("…or upload a CSV (columns: slot, tickets)", type="csv")
        if up is not None:
            start = pd.read_csv(up, dtype={"slot": str})[["slot", "tickets"]]
        st.sidebar.caption("Edit the numbers; add rows at the bottom for extra slots (HH:MM).")
        tickets = st.sidebar.data_editor(start.reset_index(drop=True), num_rows="dynamic", hide_index=True,
                                         column_config={"slot": st.column_config.TextColumn("Slot"),
                                                        "tickets": st.column_config.NumberColumn("Tickets", min_value=0, step=1)},
                                         key=f"editor_{date}", height=300)
        tickets = tickets.dropna()
        tickets = tickets[tickets.slot.astype(str).str.match(r"^\d{2}:\d{2}$")]

    st.sidebar.subheader("2. Rooms")
    open_today = pd.Series(m.open_zones(date), index=ZONES)
    closed_default = [z for z in ZONES if not open_today[z]]
    closed = st.sidebar.multiselect("Closed rooms", ZONES, default=closed_default, format_func=lambda z: ROOM_LABEL[z])
    if closed_default:
        st.sidebar.caption("Closed on this day in reality: " + ", ".join(SHORT[z] for z in closed_default))
    extra_open = [z for z in closed_default if z not in closed]

    st.sidebar.subheader("3. Visitors")
    usual_ag = float(np.mean(ag_share_by_hour(m, date)[9:17]))
    ag_share = None
    if st.sidebar.toggle("Change audio-guide take-up", value=False):
        ag_share = st.sidebar.slider("Visitors with an audio guide", 0, 100, int(round(usual_ag * 20) * 5), 5,
                                     format="%d%%") / 100
    st.sidebar.caption(f"Usual for this day: about {usual_ag:.0%} (more in the morning, fewer in the afternoon). "
                       "Visitors without a guide spend ~12% less time in the rooms.")

    st.sidebar.subheader("4. Display and accuracy")
    capacity = st.sidebar.number_input("Flag rooms above (people, 0 = off)", 0, 1000, 0, 10)
    reps = st.sidebar.select_slider("Simulation runs", [10, 20, 40, 80], value=20,
                                    help="More runs = smoother averages but slower (≈ 0.5 s per run).")
    go_btn = st.sidebar.button("Run scenario", type="primary", use_container_width=True)

    # =====================================================================  run
    st.title("Van Gogh Museum – crowding by room")
    st.caption("Expected people in each room per 15 minutes, from the logit + stay-time simulation. "
               "Baseline = the day as it happened (real entries, rooms as they were); for days without data the "
               "baseline uses the scenario's tickets with all rooms open.")

    key = (date, mode, None if tickets is None else tickets.to_csv(index=False), tuple(sorted(closed)), reps, ag_share)
    if go_btn:
        st.session_state["key"] = key
    if st.session_state.get("key") != key:
        st.info("Set the inputs on the left and press **Run scenario**. A run takes about 10–40 seconds.")
        st.stop()

    scen_arr = None if tickets is None else A.to_counts(A.expected(tickets))
    scen_csv = None if scen_arr is None else scen_arr.to_csv(index=False)
    with st.spinner("Simulating visitors…"):
        if has_real:
            base = run_sim(date, reps, tuple(closed_default), (), None, 0)
            base_arr = m.arr[m.arr.date == date][["slot", "visitors"]]
        else:
            base = run_sim(date, reps, (), (), scen_csv, 0)
            base_arr = scen_arr
        scen = run_sim(date, reps, tuple(sorted(closed)), tuple(extra_open), scen_csv, 0, ag_share)
        scen_arr_show = base_arr if scen_arr is None else scen_arr

    slots = scen["slots"]
    keep = [i for i, s in enumerate(slots) if SLOT_RANGE[0] <= s <= SLOT_RANGE[1]]
    slots = [slots[i] for i in keep]
    S_mean, S_p10, S_p90 = scen["mean"][keep], scen["p10"][keep], scen["p90"][keep]
    B_mean = base["mean"][keep]
    active = (S_mean.sum(1) + B_mean.sum(1)) > 0.5
    lo, hi = int(np.argmax(active)), len(active) - int(np.argmax(active[::-1]))
    slots, S_mean, S_p10, S_p90, B_mean = slots[lo:hi], S_mean[lo:hi], S_p10[lo:hi], S_p90[lo:hi], B_mean[lo:hi]

    # ---------- headline numbers
    inside_s, inside_b = S_mean.sum(1), B_mean.sum(1)
    pk_room = int(np.argmax(S_mean.max(0)))
    pk_slot = slots[int(np.argmax(S_mean[:, pk_room]))]
    k = st.columns(4)
    dv = scen["mean_visit_min"] - base["mean_visit_min"]
    k[0].metric("Visitors entering", f"{scen['n_visitors']:,}", f"{scen['n_visitors'] - base['n_visitors']:+,} vs baseline",
                delta_color="off")
    k[1].metric("Average visit", f"{scen['mean_visit_min']:.0f} min",
                "same as baseline" if abs(dv) < 0.5 else f"{dv:+.0f} min vs baseline", delta_color="off")
    k[2].metric("Peak people in the building", f"{inside_s.max():,.0f}", f"{inside_s.max() - inside_b.max():+,.0f} vs baseline",
                delta_color="off")
    k[3].metric("Busiest room (peak)", f"{SHORT[ZONES[pk_room]]}: {S_mean[:, pk_room].max():.0f}")
    k[3].caption(f"at {pk_slot}")
    if ag_share is not None:
        st.caption(f"Audio-guide take-up in the scenario: {scen['ag_share']:.0%} of visitors (baseline {base['ag_share']:.0%}).")

    dark = dark_mode()
    blue, grey = (BLUE_DARK, GREY_DARK) if dark else (BLUE, GREY)
    ink2 = "#c3c2b7" if dark else "#52514e"
    grid = "#383835" if dark else "#e4e3df"
    open_now = [z for z in ZONES if z not in closed]

    tab1, tab2, tab3, tab4 = st.tabs(["All rooms (heatmap)", "Room by room", "Entries", "Table & downloads"])

    # ---------- heatmap: rooms x time, people (scenario)
    with tab1:
        rows = [ZONES.index(z) for z in ZONES]
        zmat = S_mean[:, rows].T
        text = np.vectorize(lambda v: f"{v:.0f}")(zmat)
        hm = go.Figure(go.Heatmap(
            z=zmat, x=slots, y=[ROOM_LABEL[z] + (" (closed)" if z in closed else "") for z in ZONES],
            colorscale=[[i / (len(SEQ_LIGHT) - 1), c] for i, c in enumerate(SEQ_DARK if dark else SEQ_LIGHT)],
            xgap=2, ygap=2, colorbar=dict(title=dict(text="people", side="right"), thickness=12),
            hovertemplate="%{y}<br>%{x}: <b>%{z:.0f} people</b><extra></extra>"))
        if capacity:
            over = np.where(zmat > capacity, "●", "")
            hm.add_trace(go.Scatter(x=np.repeat([slots], len(ZONES), 0).ravel(), y=np.repeat(hm.data[0].y, len(slots)),
                                    mode="text", text=over.ravel(), textfont=dict(color=CRIT, size=9),
                                    hoverinfo="skip", showlegend=False))
        hm.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(autorange="reversed"),
                         xaxis=dict(dtick=4, tickangle=0))
        st.plotly_chart(hm, use_container_width=True, theme="streamlit")
        st.caption("Scenario, average of the simulation runs." + (f" ● = above {capacity} people." if capacity else ""))

    # ---------- small multiples: one chart per room, scenario vs baseline
    with tab2:
        c_a, c_b = st.columns([3, 1])
        c_a.markdown(f"<span style='color:{blue};font-weight:700'>━━</span> Scenario (average) &nbsp;&nbsp; "
                     f"<span style='background:rgba(42,120,214,0.18);padding:0 10px'>&nbsp;</span> Scenario range, 10–90% of runs &nbsp;&nbsp; "
                     f"<span style='color:{grey};font-weight:700'>┅┅</span> Baseline"
                     + (f" &nbsp;&nbsp; <span style='color:{CRIT}'>╌╌</span> {capacity} people" if capacity else ""),
                     unsafe_allow_html=True)
        shared = c_b.toggle("Same scale for all rooms", value=False)
        ymax = max(S_p90.max(), B_mean.max()) * 1.05
        cols = st.columns(3)
        for i, z in enumerate(ZONES):
            zi = ZONES.index(z)
            f = go.Figure()
            f.add_trace(go.Scatter(x=slots + slots[::-1], y=list(S_p90[:, zi]) + list(S_p10[:, zi])[::-1],
                                   fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                                   hoverinfo="skip", name="Scenario range (10–90% of runs)", showlegend=(i == 0)))
            f.add_trace(go.Scatter(x=slots, y=B_mean[:, zi], name="Baseline", line=dict(color=grey, width=2, dash="dot"),
                                   showlegend=(i == 0), hovertemplate="Baseline %{x}: %{y:.0f}<extra></extra>"))
            f.add_trace(go.Scatter(x=slots, y=S_mean[:, zi], name="Scenario", line=dict(color=blue, width=2),
                                   showlegend=(i == 0), hovertemplate="Scenario %{x}: %{y:.0f}<extra></extra>"))
            if capacity:
                f.add_hline(y=capacity, line=dict(color=CRIT, width=1, dash="dash"))
            title = ROOM_LABEL[z] + (" – closed" if z in closed else "")
            f.update_layout(title=dict(text=title, font=dict(size=13)), height=230, margin=dict(l=10, r=10, t=36, b=10),
                            hovermode="x unified", showlegend=False,
                            xaxis=dict(dtick=8, tickangle=0), yaxis=dict(range=[0, ymax] if shared else None, rangemode="tozero"))
            cols[i % 3].plotly_chart(f, use_container_width=True, theme="streamlit", key=f"room_{z}")
        st.caption("People per room, 15-minute averages. Hover a chart to read the values.")

    # ---------- entries per 15 min
    with tab3:
        e_s, e_b = entries_15(scen_arr_show), entries_15(base_arr)
        idx = sorted(set(e_s.index) | set(e_b.index))
        idx = [s for s in idx if SLOT_RANGE[0] <= s <= SLOT_RANGE[1]]
        f = go.Figure()
        f.add_trace(go.Scatter(x=idx, y=e_b.reindex(idx, fill_value=0), name="Baseline", line=dict(color=grey, width=2, dash="dot")))
        f.add_trace(go.Scatter(x=idx, y=e_s.reindex(idx, fill_value=0), name="Scenario", line=dict(color=blue, width=2)))
        f.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                        yaxis=dict(title="people entering per 15 min", rangemode="tozero"), xaxis=dict(dtick=4, tickangle=0),
                        legend=dict(orientation="h", y=1.08))
        st.plotly_chart(f, use_container_width=True, theme="streamlit")
        if tickets is not None:
            st.caption(f"Scenario: {tickets.tickets.sum():,.0f} tickets → about {scen_arr.visitors.sum():,} people entering "
                       "(93.6% show up; arrival times spread around the booked slot; nobody enters before 09:00).")

    # ---------- table + downloads
    with tab4:
        day = [i for i, s in enumerate(slots) if "10:00" <= s < "17:00"]
        summ = pd.DataFrame({
            "Room": [ROOM_LABEL[z] for z in ZONES],
            "Status": ["closed" if z in closed else "open" for z in ZONES],
            "Average 10–17h": S_mean[day].mean(0), "Baseline average": B_mean[day].mean(0),
            "Peak": S_mean.max(0), "Peak time": [slots[int(np.argmax(S_mean[:, i]))] for i in range(len(ZONES))],
            "Baseline peak": B_mean.max(0),
        })
        summ["Change in average"] = np.where(summ["Baseline average"] > 0.5,
                                             (summ["Average 10–17h"] / summ["Baseline average"] - 1) * 100, np.nan)
        if capacity:
            summ[f"15-min slots above {capacity}"] = (S_mean > capacity).sum(0)
        st.dataframe(summ, hide_index=True, use_container_width=True, height=36 * 14 + 3, column_config={
            "Average 10–17h": st.column_config.NumberColumn(format="%.0f"),
            "Baseline average": st.column_config.NumberColumn(format="%.0f"),
            "Peak": st.column_config.NumberColumn(format="%.0f"),
            "Baseline peak": st.column_config.NumberColumn(format="%.0f"),
            "Change in average": st.column_config.NumberColumn(format="%+.1f%%")})

        wide = pd.DataFrame(S_mean.round(1), columns=[ROOM_LABEL[z] for z in ZONES])
        wide.insert(0, "slot", slots)
        wide["Total inside"] = S_mean.sum(1).round(1)
        long = pd.DataFrame([(s, ROOM_LABEL[z], S_mean[i, zi], S_p10[i, zi], S_p90[i, zi], B_mean[i, zi])
                             for i, s in enumerate(slots) for zi, z in enumerate(ZONES)],
                            columns=["slot", "room", "scenario_mean", "scenario_p10", "scenario_p90", "baseline_mean"]).round(1)
        tag = f"{date}_{'closed-' + '-'.join(closed) if closed else 'all-open'}"
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("People per room (wide CSV)", wide.to_csv(index=False), f"people_per_room_{tag}.csv", "text/csv")
        d2.download_button("Detail incl. baseline & range", long.to_csv(index=False), f"people_per_room_detail_{tag}.csv", "text/csv")
        d3.download_button("Room summary", summ.round(2).to_csv(index=False), f"room_summary_{tag}.csv", "text/csv")
        if tickets is not None:
            d4.download_button("Tickets & entries used", pd.concat(
                [tickets.assign(kind="tickets_per_booked_slot")[["kind", "slot", "tickets"]].rename(columns={"tickets": "value"}),
                 scen_arr.assign(kind="entries_per_5min").rename(columns={"visitors": "value"})]).to_csv(index=False),
                f"tickets_entries_{tag}.csv", "text/csv")

    with st.expander("How to read this / assumptions"):
        st.markdown("""
    - **People per room** are averages over each 15-minute slot, from simulated visitors (the logit decides the next room
      or exit, the stay-time model how long they stay). Compared with measured crowding on past days: people in the
      building within ~1%, a room's daily average typically ~9% off, peaks under-estimated by ~20%
      (see the "Accuracy vs history" page).
    - **Tickets → entries:** 93.6% of tickets are used (82% for the Friday-evening programme); people arrive around their
      booked slot (median 3 minutes early); early arrivals queue until 09:00.
    - **Audio guide:** the behaviour model is estimated on audio-guide users. Visitors without a guide (about 55–60%)
      stay ~20% less in each room, so ~12% less over the whole visit, as the visit-duration file shows (orders with a guide
      stay +14% after controls). Part of that gap is probably who chooses a guide, so take-up changes are an upper bound.
    - **Closing a room:** visitors who would have gone there pick among the open rooms; for the exhibition wing 14% leave
      instead (calibrated on the Feb and Sep 2026 closures).
    - **Not in the model:** crowding effects on behaviour (tested: none in VGM's normal range, so far more tickets than
      today is extrapolation), weather (+2–3% visit time when it rains), groups and guided tours.
    """)


from accuracy_page import accuracy_page  # noqa: E402

pg = st.navigation([st.Page(scenario_page, title="Scenarios", icon=":material/tune:", default=True),
                    st.Page(accuracy_page, title="Accuracy vs history", icon=":material/fact_check:")])
pg.run()
