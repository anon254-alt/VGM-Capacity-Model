"""
04_stay_model.py - step 4: how long visitors stay in each room, plus how they start their visit.

Stay-time model (no weather or crowding yet):
    log(stay minutes) = room
                        + room x time-of-day band (before 10h, 10-11h, 11-12h, 14-16h, after 16h; base 12-14h)
                        + room x "first room of the visit"
                        + language group + hours already inside (+ squared) + revisit of the room
                        + Friday evening + last hour before closing + last 30 minutes before closing
                        + error
Estimated by ordinary least squares on non-final room visits of the estimation split
(final visits have an assumed end time, so they are left out); stays > 90 min are treated as breaks
and dropped. In the simulation a stay = exp(prediction + an error drawn from the real errors of that
room), so the spread of stay times matches the data.

Entry model:
  - first room: shares of each zone as first room, by whether the exhibition wing is open and
    arrival time band (counted from the data)
  - lead time: minutes between starting the visit and the first room (empirical sample)

Outputs: params/stay.json, params/stay_residuals.npz, params/entry.json, output/stay_coefficients.csv
Run from the model folder:  python 04_stay_model.py
"""
import json, os
import numpy as np
import pandas as pd

from layout import ZONES
from features import LANG_GROUPS, lang_group

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, PAR = (os.path.join(HERE, d) for d in ("data", "output", "params"))
TEST_PERIOD = ("2026-05-18", "2026-06-11")
MAX_STAY, MIN_STAY = 90, 0.25
BANDS = ["before_10h", "10h_11h", "11h_12h", "14h_16h", "after_16h"]      # base band: 12h-14h
STAY_FEATURES = ([f"room_{z}" for z in ZONES] +
                 [f"{z}_x_{b}" for z in ZONES for b in BANDS] +              # room-specific time of day
                 [f"{z}_x_first_room" for z in ZONES] +                      # room is the visit's first room
                 [f"lang_{g}" for g in LANG_GROUPS[1:]] +
                 ["hours_inside", "hours_inside_sq", "revisit", "friday_evening",
                  "close_lt60", "close_lt30"])


def _band(clock_h):
    return np.select([clock_h < 10, clock_h < 11, clock_h < 12, clock_h < 14, clock_h < 16],
                     [0, 1, 2, -1, 3], 4)                                    # -1 = base band


def stay_X(zone_idx, lang_g, clock_h, hours, revisit, friday, first_room, hours_to_close):
    n, nz, nb = len(zone_idx), len(ZONES), len(BANDS)
    X = np.zeros((n, len(STAY_FEATURES)))
    r = np.arange(n)
    X[r, zone_idx] = 1
    b = _band(np.asarray(clock_h))
    has = b >= 0
    X[r[has], nz + zone_idx[has] * nb + b[has]] = 1
    o = nz + nz * nb
    X[r[first_room], o + zone_idx[first_room]] = 1
    o += nz
    for g in range(1, len(LANG_GROUPS)):
        X[:, o + g - 1] = lang_g == g
    o += len(LANG_GROUPS) - 1
    X[:, o] = hours
    X[:, o + 1] = hours ** 2
    X[:, o + 2] = revisit
    X[:, o + 3] = friday & (np.asarray(clock_h) >= 17)
    X[:, o + 4] = hours_to_close < 1.0
    X[:, o + 5] = hours_to_close < 0.5
    return X


def main():
    v = pd.read_csv(os.path.join(DATA, "room_visits.csv"), parse_dates=["s_start", "t_in", "t_out"])
    v = v[(v.split == "est") & (v.n_visits >= 3) & ~v.date.between(*TEST_PERIOD)].sort_values(["sid", "visit_no"])
    v["revisit"] = v.groupby(["sid", "zone"]).cumcount() > 0
    zi = {z: i for i, z in enumerate(ZONES)}

    # ---------- stay-time regression
    s = v[(~v.is_last) & (v.stay_min <= MAX_STAY)].copy()
    clock = s.t_in.dt.hour + s.t_in.dt.minute / 60
    close = pd.read_csv(os.path.join(DATA, "closing_times.csv")).set_index("date").close_h
    X = stay_X(s.zone.map(zi).to_numpy(), s.lang.map(lang_group).to_numpy(), clock.to_numpy(),
               (s.min_inside / 60).to_numpy(), s.revisit.to_numpy(), (s.t_in.dt.dayofweek == 4).to_numpy(),
               (s.visit_no == 1).to_numpy(), (s.date.map(close) - clock).to_numpy())
    y = np.log(np.clip(s.stay_min.to_numpy(), MIN_STAY, None))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    sigma2 = resid.var()
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * sigma2)
    pd.DataFrame({"factor": STAY_FEATURES, "beta": beta, "se": se, "multiplier_exp_beta": np.exp(beta)}) \
      .round(4).to_csv(os.path.join(OUT, "stay_coefficients.csv"), index=False)
    rng = np.random.default_rng(1)
    res_by_zone = {}
    for z in ZONES:
        r = resid[s.zone.to_numpy() == z]
        res_by_zone[z] = rng.choice(r, min(5000, len(r)), replace=False)
    np.savez(os.path.join(PAR, "stay_residuals.npz"), **res_by_zone)
    json.dump({"features": STAY_FEATURES, "beta": beta.tolist(), "r2": float(r2), "n": int(len(s)),
               "max_stay_min": MAX_STAY, "min_stay_min": MIN_STAY},
              open(os.path.join(PAR, "stay.json"), "w"), indent=1)

    # check: median observed vs median simulated stay per room
    pred = np.exp(X @ beta + rng.choice(resid, len(resid)))
    chk = pd.DataFrame({"zone": s.zone, "obs": s.stay_min.to_numpy(), "sim": pred}).groupby("zone").median().round(1)

    # ---------- entry: first room shares and lead time
    open_ = pd.read_csv(os.path.join(DATA, "open_rooms.csv"))
    exh_open = open_[open_.zone == "EXH"].set_index("date").open
    f = v[v.visit_no == 1].copy()
    f["exh_open"] = f.date.map(exh_open).fillna(False)
    f["band"] = pd.cut(f.t_in.dt.hour, [0, 11, 14, 24], right=False, labels=["before_11h", "11h_14h", "after_14h"])
    shares = {}
    for (eo, band), g in f.groupby(["exh_open", "band"], observed=True):
        p = g.zone.value_counts(normalize=True).reindex(ZONES, fill_value=0)
        if not eo:
            p["EXH"] = 0
            p /= p.sum()
        shares[f"exh_{'open' if eo else 'closed'}|{band}"] = p.round(5).to_dict()
    lead = ((f.t_in - f.s_start).dt.total_seconds() / 60).clip(0, 30)
    json.dump({"first_room_shares": shares, "lead_minutes_sample": lead.sample(5000, random_state=1).round(2).tolist()},
              open(os.path.join(PAR, "entry.json"), "w"), indent=1)
    print(f"stay model: n={len(s):,}, R2={r2:.3f}")
    print(chk.to_string())
    print(pd.DataFrame(shares).round(3).to_string())


if __name__ == "__main__":
    main()
