"""
03_estimate_logit.py - step 3: estimate the next-room logit (conditional / McFadden logit).

Each room visit ends with a decision: go to another open room, or EXIT. The factors are built in
features.py; the layout in layout.py. Estimation = maximum likelihood with an analytic gradient
(scipy L-BFGS); standard errors from the inverse Hessian.

Samples (sessions with >= MIN_VISITS room visits):
  - estimation: up to N_EST sessions from split 'est', EXCLUDING the closure-test period
  - test:       up to N_TEST sessions from split 'test' on normal days (held out)
  - closure test: all 'test' sessions in TEST_PERIOD (exhibition wing closed 18 May - 11 Jun 2026)
Benchmark: counted transition table P(next room | current room) from the estimation sample.

Outputs: output/logit_coefficients.csv, output/logit_fit.json, params/logit.json,
         data/open_rooms.csv (which zones were open each day)
Run from the model folder:  python 03_estimate_logit.py
"""
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from layout import ZONES, ZONE_IDX, FLOOR, FLOOR_IDX, N
from features import FEATURES, K, J, build_X, lang_group

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, PAR = (os.path.join(HERE, d) for d in ("data", "output", "params"))
for d in (OUT, PAR):
    os.makedirs(d, exist_ok=True)
MIN_VISITS = 3
N_EST, N_TEST = 25_000, 10_000
TEST_PERIOD = ("2026-05-18", "2026-06-11")
OPEN_SHARE = 0.05          # a zone counts as open on a day if >= 5% of sampled sessions visited it
SEED = 7


def load_visits():
    v = pd.read_csv(os.path.join(DATA, "room_visits.csv"),
                    parse_dates=["s_start", "s_end", "t_in", "t_out"])
    # which zones were open each day (data-driven)
    tot = v.groupby("date").sid.nunique()
    share = v.groupby(["date", "zone"]).sid.nunique().unstack(fill_value=0).div(tot, axis=0)
    share = share.reindex(columns=ZONES, fill_value=0)
    open_ = (share >= OPEN_SHARE)
    open_.stack().rename("open").reset_index().rename(columns={"level_1": "zone"}) \
         .to_csv(os.path.join(DATA, "open_rooms.csv"), index=False)
    closing_times(v)
    return v[v.n_visits >= MIN_VISITS].copy(), open_


def closing_times(v):
    """Effective closing time per day = when the last visitors leave (99th percentile of visit ends),
    rounded up to the next 15 minutes. Saved to data/closing_times.csv (hours, e.g. 17.75)."""
    q = v.groupby("date").t_out.quantile(0.99)
    h = q.dt.hour + q.dt.minute / 60 + q.dt.second / 3600
    close = np.ceil(h * 4) / 4
    pd.DataFrame({"date": close.index, "close_h": close.values,
                  "weekday": pd.to_datetime(close.index).dayofweek}).to_csv(os.path.join(DATA, "closing_times.csv"), index=False)


def decisions(v):
    """One row per decision (end of each room visit)."""
    v = v.sort_values(["sid", "visit_no"]).reset_index(drop=True)
    v["cur"] = v.zone.map(ZONE_IDX)
    nxt = v.groupby("sid").zone.shift(-1)
    v["chosen"] = nxt.map(ZONE_IDX).fillna(N).astype(int)          # N = EXIT
    v["hours"] = (v.t_out - v.s_start).dt.total_seconds() / 3600
    v["clock_h"] = v.t_out.dt.hour + v.t_out.dt.minute / 60
    v["friday"] = v.t_out.dt.dayofweek == 4
    close = pd.read_csv(os.path.join(DATA, "closing_times.csv")).set_index("date").close_h
    v["to_close"] = v.date.map(close) - v.clock_h
    v["lang_g"] = v.lang.map(lang_group)
    for z in ZONES:                                                   # zones seen so far (incl. current)
        v[f"seen_{z}"] = (v.zone == z).astype(int).groupby(v.sid).cummax()
    fl = v.zone.map(FLOOR)
    for f in FLOOR_IDX:
        v[f"fseen_{f}"] = (fl == f).astype(int).groupby(v.sid).cummax()
    return v


def arrays(d, open_):
    seen_zone = d[[f"seen_{z}" for z in ZONES]].to_numpy(bool)
    seen_floor = d[[f"fseen_{f}" for f in FLOOR_IDX]].to_numpy(bool)
    X = build_X(d.cur.to_numpy(), d.hours.to_numpy(), seen_zone, seen_floor,
                d.clock_h.to_numpy(), d.friday.to_numpy(), d.lang_g.to_numpy(),
                hours_to_close=d.to_close.to_numpy())
    avail = np.ones((len(d), J), bool)
    avail[:, :N] = open_.reindex(d.date).to_numpy(bool)
    avail[np.arange(len(d)), d.cur.to_numpy()] = False
    ch = d.chosen.to_numpy()
    ok = avail[np.arange(len(d)), ch]
    return X[ok], avail[ok], ch[ok], d[ok]


def probs(beta, X, avail):
    V = np.einsum("djk,k->dj", X, beta, dtype=np.float64)
    V[~avail] = -np.inf
    V -= V.max(1, keepdims=True)
    P = np.exp(V)
    P /= P.sum(1, keepdims=True)
    return P


def negll_grad(beta, X, avail, ch):
    P = probs(beta, X, avail)
    D = len(ch)
    ll = np.log(P[np.arange(D), ch] + 1e-300).sum()
    grad = X[np.arange(D), ch].sum(0, dtype=np.float64) - np.einsum("dj,djk->k", P, X, dtype=np.float64)
    return -ll, -grad


def hessian(beta, X, avail, chunk=20_000):
    H = np.zeros((K, K))
    for s in range(0, len(X), chunk):
        Xc = X[s:s + chunk].astype(np.float64)
        P = probs(beta, X[s:s + chunk], avail[s:s + chunk])
        xb = np.einsum("dj,djk->dk", P, Xc)
        H += np.einsum("dj,djk,djl->kl", P, Xc, Xc) - xb.T @ xb
    return H


def metrics(P, ch):
    D = len(ch)
    return {"decisions": int(D), "avg_loglik": float(np.log(P[np.arange(D), ch] + 1e-300).mean()),
            "top1_accuracy": float((P.argmax(1) == ch).mean())}


def main():
    t0 = time.time()
    v, open_ = load_visits()
    rng = np.random.default_rng(SEED)
    in_test_period = v.date.between(*TEST_PERIOD)
    est_ids = v[(v.split == "est") & ~in_test_period].sid.unique()
    test_ids = v[(v.split == "test") & ~in_test_period].sid.unique()
    clos_ids = v[(v.split == "test") & in_test_period].sid.unique()
    est_ids = rng.choice(est_ids, min(N_EST, len(est_ids)), replace=False)
    test_ids = rng.choice(test_ids, min(N_TEST, len(test_ids)), replace=False)

    sets = {}
    for name, ids in [("est", est_ids), ("test", test_ids), ("closure", clos_ids)]:
        sets[name] = arrays(decisions(v[v.sid.isin(ids)]), open_)
        print(f"{name}: {len(ids):,} sessions, {len(sets[name][2]):,} decisions")
    X, avail, ch, _ = sets["est"]

    res = minimize(negll_grad, np.zeros(K), args=(X, avail, ch), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-6})
    beta = res.x
    H = hessian(beta, X, avail)
    se = np.sqrt(np.diag(np.linalg.pinv(H)))
    print("converged:", res.success, res.message, f"{time.time() - t0:.0f}s")

    # null model: all available options equally likely
    null_ll = -np.log(avail.sum(1)).sum()
    coefs = pd.DataFrame({"factor": FEATURES, "beta": beta, "se": se, "z": beta / se,
                          "multiplier_exp_beta": np.exp(beta)})
    coefs.round(4).to_csv(os.path.join(OUT, "logit_coefficients.csv"), index=False)

    # benchmark: counted transitions P(next | current), add-one smoothing, renormalised on open options
    counts = np.ones((N, J))
    np.add.at(counts, (sets["est"][3].cur.to_numpy(), ch), 1)
    fit = {"estimation": {**metrics(probs(beta, X, avail), ch), "loglik": float(-res.fun),
                          "null_loglik": float(null_ll), "mcfadden_r2": float(1 + res.fun / null_ll)}}
    for name in ("test", "closure"):
        Xs, av, chs, ds = sets[name]
        Pl = probs(beta, Xs, av)
        Pc = counts[ds.cur.to_numpy()] * av
        Pc /= Pc.sum(1, keepdims=True)
        fit[name] = {"logit": metrics(Pl, chs), "counted_table": metrics(Pc, chs)}
    json.dump(fit, open(os.path.join(OUT, "logit_fit.json"), "w"), indent=2)
    json.dump({"features": FEATURES, "beta": beta.tolist(), "se": se.tolist(),
               "open_share_rule": OPEN_SHARE, "estimated_on": "est split, excluding " + " to ".join(TEST_PERIOD)},
              open(os.path.join(PAR, "logit.json"), "w"), indent=1)
    print(json.dumps(fit, indent=1))
    print(coefs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
