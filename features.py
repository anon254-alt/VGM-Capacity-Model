"""
features.py - builds the factor matrix of the next-room logit (shared by estimation and simulation).

For D decisions it returns X with shape (D, 14 options, K factors). Options = 13 zones + EXIT.
EXIT has all factors = 0, so every coefficient reads as "attractiveness relative to leaving".
"""
import numpy as np
from layout import ZONES, N, FLOOR_ARR, FLOOR_IDX, DFLOOR, SAME_FLOOR_FAR, ZONE_IDX, SEQ_DELTA, SEQ

LANG_GROUPS = ["English", "Italian", "Chinese (Mandarin)", "French", "Spanish", "German", "Dutch",
               "Portuguese", "Korean", "Other"]          # English = base
LANG_IDX = {l: i for i, l in enumerate(LANG_GROUPS)}

FEATURES = ([f"const_{z}" for z in ZONES] +
            ["floor_change", "floors_extra", "going_down", "same_floor_far", "visited", "back_to_floor",
             "exh_x_share_seen", "seq_next", "seq_ahead_extra", "seq_back", "seq_back_extra"] +
            ["cont_hours_inside", "cont_hours_inside_sq", "cont_share_seen", "cont_before_11h",
             "cont_after_16h", "cont_friday_evening", "cont_close_lt60", "cont_close_lt30"] +
            [f"cont_lang_{g}" for g in LANG_GROUPS[1:]] +
            [f"cont_from_{z}" for z in ZONES[1:]])       # base: deciding in Room 1
K = len(FEATURES)
F = {f: i for i, f in enumerate(FEATURES)}
J = N + 1                                                # 14 options


def lang_group(lang):
    return LANG_IDX.get(lang, LANG_IDX["Other"])


def seq_delta(open_mask=None):
    """Sequence distance between rooms, counting only OPEN numbered rooms: if Room 5 is closed,
    Room 6 becomes the next room after Room 4. EXH is outside the sequence (delta 0)."""
    if open_mask is None:
        return SEQ_DELTA
    pos = np.cumsum((SEQ > 0) & open_mask) * (SEQ > 0)
    d = pos[None, :] - pos[:, None]
    d[:, ZONE_IDX["EXH"]] = 0
    d[ZONE_IDX["EXH"], :] = 0
    return d


def build_X(cur, hours, seen_zone, seen_floor, clock_h, friday, lang, dtype=np.float32, open_mask=None,
            hours_to_close=None):
    """
    cur        (D,) int   current zone index
    hours      (D,) float hours since the visitor started
    seen_zone  (D,13) bool zones visited so far (including current)
    seen_floor (D,5) bool floors visited so far
    clock_h    (D,) float clock time of the decision in hours (e.g. 14.5)
    friday     (D,) bool
    lang       (D,) int   language group index
    open_mask  (13,) bool rooms open (only used to renumber the sequence around closed rooms)
    hours_to_close (D,) float hours until that day's closing time (None = far from closing)
    """
    D = len(cur)
    X = np.zeros((D, J, K), dtype=dtype)
    r = slice(0, N)                                       # room options (not EXIT)
    X[:, r, F["const_R01"]:F["const_R01"] + N] = np.eye(N, dtype=dtype)[None]
    df = DFLOOR[cur]                                      # (D,13)
    fc = df != 0
    X[:, r, F["floor_change"]] = fc
    X[:, r, F["floors_extra"]] = np.maximum(np.abs(df) - 1, 0)
    X[:, r, F["going_down"]] = df < 0
    X[:, r, F["same_floor_far"]] = SAME_FLOOR_FAR[cur]
    X[:, r, F["visited"]] = seen_zone
    tgt_floor_seen = seen_floor[:, [FLOOR_IDX[f] for f in FLOOR_ARR]]   # (D,13)
    X[:, r, F["back_to_floor"]] = fc & tgt_floor_seen
    # museum sequence (Room 1 -> 12, as numbered and as listed in the audio guide); EXH not in sequence
    d = seq_delta(open_mask)[cur]                         # (D,13) target position - current position
    X[:, r, F["seq_next"]] = d == 1
    X[:, r, F["seq_ahead_extra"]] = np.where(d > 1, d - 1, 0)
    X[:, r, F["seq_back"]] = d < 0
    X[:, r, F["seq_back_extra"]] = np.where(d < 0, -d - 1, 0)
    share_seen = seen_zone[:, :12].sum(1) / 12.0          # share of the 12 collection rooms seen
    X[:, ZONE_IDX["EXH"], F["exh_x_share_seen"]] = share_seen
    ctx = {"cont_hours_inside": hours, "cont_hours_inside_sq": hours ** 2, "cont_share_seen": share_seen,
           "cont_before_11h": clock_h < 11, "cont_after_16h": clock_h >= 16,
           "cont_friday_evening": friday & (clock_h >= 17)}
    if hours_to_close is not None:
        ctx["cont_close_lt60"] = hours_to_close < 1.0
        ctx["cont_close_lt30"] = hours_to_close < 0.5
    for name, val in ctx.items():
        X[:, r, F[name]] = np.asarray(val, dtype=dtype)[:, None]
    for g in range(1, len(LANG_GROUPS)):
        X[:, r, F[f"cont_lang_{LANG_GROUPS[g]}"]] = (lang == g)[:, None]
    for z in ZONES[1:]:
        X[:, r, F[f"cont_from_{z}"]] = (cur == ZONE_IDX[z])[:, None]
    return X
