"""
layout.py - room layout used by the crowding model (shared by estimation and simulation).

Drafted from the floor plans RV0-RV3.jpg and the visitor map; CHECK AND EDIT if needed.
  - Floor 0: Room 1 (entrance level)
  - Floor 1: Rooms 2-5, arranged in a ring around the central stairwell (2-3-4-5-2)
  - Floor 2: Rooms 6-8, ring (6-7-8-6)
  - Floor 3: Rooms 9-12, ring (9-10-11-12-9)
  - Exhibition wing (EXH): separate building reached via the entrance hall (lower level), coded floor -1
Distance is measured in "steps": rooms apart around the ring on the same floor, and floors apart
between floors (no metres: the plans have no scale).
"""
ZONES = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12", "EXH"]
EXIT = "EXIT"
OPTIONS = ZONES + [EXIT]                     # choice options at every decision
ZONE_IDX = {z: i for i, z in enumerate(ZONES)}

FLOOR = {"R01": 0, "R02": 1, "R03": 1, "R04": 1, "R05": 1, "R06": 2, "R07": 2, "R08": 2,
         "R09": 3, "R10": 3, "R11": 3, "R12": 3, "EXH": -1}
FLOORS = [-1, 0, 1, 2, 3]
FLOOR_IDX = {f: i for i, f in enumerate(FLOORS)}

RINGS = [["R02", "R03", "R04", "R05"], ["R06", "R07", "R08"], ["R09", "R10", "R11", "R12"]]

ROOM_LABEL = {
    "R01": "Room 1 - Self-portraits", "R02": "Room 2 - Rural life", "R03": "Room 3 - Peasant life",
    "R04": "Room 4 - Paris", "R05": "Room 5 - Arles", "R06": "Room 6 - Temporary presentation",
    "R07": "Room 7 - Artistic exchange", "R08": "Room 8 - Family", "R09": "Room 9 - Saint-Remy",
    "R10": "Room 10 - Auvers", "R11": "Room 11 - Inspiration", "R12": "Room 12 - The end",
    "EXH": "Exhibition wing",
}


def ring_steps(a, b):
    """Rooms apart around the ring if a and b are on the same floor, else None."""
    for ring in RINGS:
        if a in ring and b in ring:
            i, j, n = ring.index(a), ring.index(b), len(ring)
            return min(abs(i - j), n - abs(i - j))
    return 0 if a == b else None


# pre-computed matrices (13 x 13) used to build features quickly
import numpy as np

N = len(ZONES)
FLOOR_ARR = np.array([FLOOR[z] for z in ZONES])
SAME_FLOOR_FAR = np.zeros((N, N), dtype=np.float32)   # same floor, 2+ rooms apart
for a in ZONES:
    for b in ZONES:
        s = ring_steps(a, b)
        if s is not None and s >= 2:
            SAME_FLOOR_FAR[ZONE_IDX[a], ZONE_IDX[b]] = 1.0
DFLOOR = FLOOR_ARR[None, :] - FLOOR_ARR[:, None]      # target floor - current floor

# position in the museum's sequence (room numbers; the audio guide lists stops in this order).
# EXH is outside the sequence: delta = 0 so sequence factors do not apply to it.
SEQ = np.array([int(z[1:]) if z != "EXH" else 0 for z in ZONES])
SEQ_DELTA = SEQ[None, :] - SEQ[:, None]
SEQ_DELTA[:, ZONE_IDX["EXH"]] = 0
SEQ_DELTA[ZONE_IDX["EXH"], :] = 0
