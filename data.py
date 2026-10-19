"""FoodBridge AI - synthetic scenario data (illustrative, not real-world measurements)."""
import math
from dataclasses import dataclass

SPEED_KMH = 30.0   # average vehicle speed (km/h)
STOP_H = 0.1       # time spent at each pickup / drop-off (hours)


@dataclass
class FoodLot:
    id: str
    donor: str
    node: str
    portions: int
    food_type: str
    expiry_h: float          # deadline, hours from simulation start


@dataclass
class Recipient:
    id: str
    name: str
    node: str
    need: int
    received: int
    accepts: tuple
    active: bool = True


@dataclass
class Vehicle:
    id: str
    capacity: int
    node: str
    available: bool = True
    free_time: float = 0.0


# node id -> (name, kind, x_km, y_km)
NODES = {
    "D":  ("Central Depot", "depot", 0, 0),
    "J1": ("Junction 1", "junction", 4, 1),
    "J2": ("Junction 2", "junction", 8, -2),
    "J3": ("Junction 3", "junction", 6, 6),
    "J4": ("Junction 4", "junction", 2, 5),
    "RA": ("Restaurant A", "donor", 3, 2),
    "BB": ("Bakery B", "donor", 7, 1),
    "GC": ("Grocery C", "donor", 9, 5),
    "RD": ("Restaurant D", "donor", 1, 6),
    "CE": ("Cafe E", "donor", 11, -1),
    "HF": ("Hotel F (standby donor)", "donor", 5, -1),
    "R1": ("Hope Shelter", "recipient", 4, 4),
    "R2": ("Saint Peter's Haven", "recipient", 12, 6),
    "R3": ("Community Kitchen", "recipient", 6, -3),
    "R4": ("Our Mission", "recipient", 2, 9),
}

EDGES = [
    ("D", "RA"), ("D", "J1"), ("D", "J4"), ("RA", "J1"), ("RA", "R1"),
    ("J1", "BB"), ("J1", "J2"), ("J1", "J3"), ("J1", "HF"), ("BB", "J2"),
    ("BB", "GC"), ("HF", "J2"), ("J2", "R3"), ("J2", "CE"), ("J3", "R1"),
    ("J3", "GC"), ("J3", "J4"), ("J4", "RD"), ("J4", "R1"), ("J4", "R4"),
    ("GC", "R2"), ("CE", "R2"), ("RD", "R4"),
]

DETOUR = 1.25                                   # road length vs straight line
SLOW_ROADS = {("J1", "J3"): 1.7, ("J2", "R3"): 1.5}


def edge_weight(a, b):
    """Road distance in km. Always >= straight-line distance, so the A* heuristic is admissible."""
    factor = SLOW_ROADS.get((a, b)) or SLOW_ROADS.get((b, a)) or DETOUR
    ax, ay = NODES[a][2:]
    bx, by = NODES[b][2:]
    return round(math.dist((ax, ay), (bx, by)) * factor, 2)


URGENT_TEMPLATE = dict(donor="Hotel F (urgent)", node="HF", portions=30,
                       food_type="cooked", window_h=1.5)


def new_scenario():
    lots = [
        FoodLot("L1", "Restaurant A", "RA", 40, "cooked", 2.0),
        FoodLot("L2", "Bakery B", "BB", 25, "bread", 5.0),
        FoodLot("L3", "Grocery C", "GC", 30, "fruit", 6.0),
        FoodLot("L4", "Restaurant D", "RD", 20, "cooked", 3.0),
        FoodLot("L5", "Cafe E", "CE", 15, "bread", 4.0),
    ]
    recipients = [
        Recipient("R1", "Hope Shelter", "R1", 80, 40, ("cooked", "bread", "fruit")),
        Recipient("R2", "Saint Peter's Haven", "R2", 60, 0, ("cooked", "bread", "fruit")),
        Recipient("R3", "Community Kitchen", "R3", 50, 10, ("cooked", "bread")),
        Recipient("R4", "Our Mission", "R4", 40, 20, ("bread", "fruit")),
    ]
    vehicles = [
        Vehicle("V1", 50, "D"),
        Vehicle("V2", 40, "D"),
        Vehicle("V3", 30, "D"),
    ]
    return lots, recipients, vehicles
