"""FoodBridge AI core logic (no machine learning):
1. Constraint-checked greedy allocation (Fastest / Fair objectives)
2. A* route planning on a road graph
3. Goal-based dynamic replanning after disruptions
4. Metrics + nearest-recipient baseline
"""
import copy
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field

from data import (NODES, EDGES, SPEED_KMH, STOP_H, FoodLot, URGENT_TEMPLATE,
                  edge_weight, new_scenario)

CHUNK = 10           # portions allocated per greedy step
INF = float("inf")


# --------------------------------------------------------------------------
# Graph + A*
# --------------------------------------------------------------------------
class Graph:
    def __init__(self):
        self.adj = {n: [] for n in NODES}
        for a, b in EDGES:
            w = edge_weight(a, b)
            self.adj[a].append((b, w))
            self.adj[b].append((a, w))

    @staticmethod
    def h(a, b):
        """Heuristic: straight-line distance (never overestimates road distance)."""
        return math.dist(NODES[a][2:], NODES[b][2:])

    def search(self, start, goal, use_heuristic=True):
        """A* (use_heuristic=True) or Dijkstra (False). Returns (path, distance, nodes_expanded)."""
        h = (lambda n: self.h(n, goal)) if use_heuristic else (lambda n: 0.0)
        frontier = [(h(start), 0.0, start)]
        best, parent, closed, expanded = {start: 0.0}, {}, set(), 0
        while frontier:
            _, g, node = heapq.heappop(frontier)
            if node in closed:
                continue
            closed.add(node)
            expanded += 1
            if node == goal:
                path = [node]
                while node in parent:
                    node = parent[node]
                    path.append(node)
                return path[::-1], g, expanded
            for nxt, w in self.adj[node]:
                ng = g + w
                if ng < best.get(nxt, INF):
                    best[nxt] = ng
                    parent[nxt] = node
                    heapq.heappush(frontier, (ng + h(nxt), ng, nxt))
        return [], INF, expanded


GRAPH = Graph()
_ROUTES = {}


def route(a, b):
    """Cached A* route -> (path, km)."""
    if (a, b) not in _ROUTES:
        path, km, _ = GRAPH.search(a, b)
        _ROUTES[(a, b)] = (path, km)
    return _ROUTES[(a, b)]


def dist(a, b):
    return route(a, b)[1]


def search_comparison(start="D"):
    """Nodes expanded by A* vs Dijkstra from `start` to every recipient."""
    rows = []
    for n, (name, kind, *_rest) in NODES.items():
        if kind == "recipient":
            _, km, a = GRAPH.search(start, n, True)
            _, _, d = GRAPH.search(start, n, False)
            rows.append({"Destination": name, "Distance (km)": round(km, 2),
                         "A* nodes expanded": a, "Dijkstra nodes expanded": d})
    return rows


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Delivery:
    lot_id: str
    donor: str
    recipient_id: str
    recipient_name: str
    qty: int
    vehicle_id: str
    pickup_node: str
    drop_node: str
    deadline: float
    t_start: float
    t_pick: float
    t_drop: float
    dist: float
    path: list
    status: str          # ok | late | failed | done
    reason: str = ""


@dataclass
class Plan:
    deliveries: list = field(default_factory=list)
    unallocated: list = field(default_factory=list)   # food nobody can receive in time
    objective: str = ""


@dataclass
class State:
    lots: list
    recipients: list
    vehicles: list
    time: float = 0.0
    done: list = field(default_factory=list)
    plan: Plan = None
    total_food: int = 0
    objective: str = "Fair"
    log: list = field(default_factory=list)


def new_state(objective="Fair"):
    lots, recipients, vehicles = new_scenario()
    s = State(lots, recipients, vehicles, total_food=sum(l.portions for l in lots), objective=objective)
    s.plan = make_plan(s, objective)
    return s


# --------------------------------------------------------------------------
# Allocation + scheduling
# --------------------------------------------------------------------------
def _eval(lot, rec, qty, v, free):
    """Build a delivery for vehicle v given its current (node, free_time)."""
    vnode, t0 = free[v.id]
    p1, d1 = route(vnode, lot.node)
    p2, d2 = route(lot.node, rec.node)
    t_pick = t0 + d1 / SPEED_KMH + STOP_H
    t_drop = t_pick + d2 / SPEED_KMH + STOP_H
    return Delivery(lot.id, lot.donor, rec.id, rec.name, qty, v.id, lot.node, rec.node,
                    lot.expiry_h, t0, t_pick, t_drop, d1 + d2, p1 + p2[1:],
                    "ok" if t_drop <= lot.expiry_h + 1e-9 else "late")


def _reachable(lot, rec, vehicles, now):
    """Constraint check: can ANY available vehicle get this lot to this recipient before expiry?"""
    for v in vehicles:
        t = max(now, v.free_time) + dist(v.node, lot.node) / SPEED_KMH + STOP_H
        t += dist(lot.node, rec.node) / SPEED_KMH + STOP_H
        if t <= lot.expiry_h + 1e-9:
            return True
    return False


def _why(objective, rec, d, ratio):
    if objective == "Fastest":
        return (f"Fastest: {rec.name} is the nearest feasible recipient "
                f"({d.dist:.1f} km trip); {d.vehicle_id} arrives earliest ({d.t_drop:.2f}h <= {d.deadline:.1f}h).")
    return (f"Fair: {rec.name} was {ratio:.0%} unmet when planning, so it gets priority over nearer, "
            f"better-served sites; {d.vehicle_id} arrives at {d.t_drop:.2f}h (deadline {d.deadline:.1f}h).")


def _fail(lot, rec, qty, why):
    return Delivery(lot.id, lot.donor, rec.id, rec.name, qty, "-", lot.node, rec.node,
                    lot.expiry_h, 0.0, 0.0, 0.0, 0.0, [], "failed", why)


def make_plan(state, objective):
    """Plan all remaining food. objective: 'Fastest' | 'Fair' | 'Baseline'."""
    if objective == "Baseline":
        return _baseline(state)
    vehicles = [v for v in state.vehicles if v.available]
    recs = {r.id: r for r in state.recipients}
    unmet = {r.id: r.need - r.received for r in state.recipients if r.active}
    ratio0 = {rid: u / recs[rid].need for rid, u in unmet.items()}
    alloc, unalloc = defaultdict(int), []

    # Step 1: constraint-checked greedy allocation, most urgent lots first
    for lot in sorted([l for l in state.lots if l.portions > 0], key=lambda l: l.expiry_h):
        remaining = lot.portions
        while remaining > 0:
            cands = []
            for rid, u in unmet.items():
                r = recs[rid]
                if u <= 0 or lot.food_type not in r.accepts:       # demand + diet constraints
                    continue
                if not _reachable(lot, r, vehicles, state.time):   # vehicle + deadline constraint
                    continue
                d = dist(lot.node, r.node)
                score = -d if objective == "Fastest" else 10 * (u / r.need) - 0.12 * d
                cands.append((score, rid))
            if not cands:
                unalloc.append(dict(lot_id=lot.id, donor=lot.donor, qty=remaining,
                                    reason=f"No active recipient with unmet need that accepts {lot.food_type} "
                                           f"can be reached before the {lot.expiry_h:.1f}h deadline."))
                break
            rid = max(cands)[1]
            q = min(CHUNK, remaining, unmet[rid])
            alloc[(lot.id, rid)] += q
            unmet[rid] -= q
            remaining -= q

    # Step 2: schedule deliveries on vehicles (earliest arrival that meets the deadline)
    free = {v.id: [v.node, max(state.time, v.free_time)] for v in vehicles}
    maxcap = max((v.capacity for v in vehicles), default=0)
    lots = {l.id: l for l in state.lots}
    pieces = []
    for (lid, rid), qty in alloc.items():
        while qty > 0:
            q = min(qty, maxcap) if maxcap else qty
            pieces.append((lots[lid], recs[rid], q))
            qty -= q
    pieces.sort(key=lambda p: (p[0].expiry_h, p[0].id))

    out = []
    for lot, rec, q in pieces:
        best = None
        for v in vehicles:
            if v.capacity < q:
                continue
            d = _eval(lot, rec, q, v, free)
            if d.status == "ok" and (best is None or d.t_drop < best.t_drop):
                best = d
        if best:
            free[best.vehicle_id] = [best.drop_node, best.t_drop]
            best.reason = _why(objective, rec, best, ratio0[rec.id])
            out.append(best)
        else:
            out.append(_fail(lot, rec, q, "AT RISK: no available vehicle can reach "
                                          f"{rec.name} before the {lot.expiry_h:.1f}h deadline."))
    return Plan(out, unalloc, objective)


def _baseline(state):
    """Naive baseline: input order, nearest recipient takes everything, vehicles round-robin, no deadline check."""
    vehicles = [v for v in state.vehicles if v.available]
    unmet = {r.id: r.need - r.received for r in state.recipients if r.active}
    recs = {r.id: r for r in state.recipients}
    free = {v.id: [v.node, max(state.time, v.free_time)] for v in vehicles}
    out, unalloc, i = [], [], 0
    for lot in [l for l in state.lots if l.portions > 0]:
        remaining = lot.portions
        for rid in sorted(unmet, key=lambda r: dist(lot.node, recs[r].node)):
            r = recs[rid]
            if remaining == 0:
                break
            if unmet[rid] <= 0 or lot.food_type not in r.accepts:
                continue
            q = min(remaining, unmet[rid])
            unmet[rid] -= q
            remaining -= q
            while q > 0 and vehicles:
                v = vehicles[i % len(vehicles)]
                i += 1
                take = min(q, v.capacity)
                d = _eval(lot, r, take, v, free)
                d.reason = "Baseline: nearest recipient, input order, no deadline check."
                free[v.id] = [d.drop_node, d.t_drop]
                out.append(d)
                q -= take
        if remaining > 0:
            unalloc.append(dict(lot_id=lot.id, donor=lot.donor, qty=remaining, reason="No recipient left with unmet need."))
    return Plan(out, unalloc, "Baseline")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(state, plan):
    ds = plan.deliveries
    ok = [d for d in ds if d.status == "ok"]
    got = defaultdict(int)
    for d in ok:
        got[d.recipient_id] += d.qty
    active = [r for r in state.recipients if r.active]
    ratios = [min(1.0, (r.received + got[r.id]) / r.need) for r in active]
    sq = sum(x * x for x in ratios)
    jain = (sum(ratios) ** 2) / (len(ratios) * sq) if sq > 0 else 0.0
    delivered = sum(d.qty for d in state.done) + sum(d.qty for d in ok)
    return {
        "Total distance (km)": round(sum(d.dist for d in state.done + ds), 1),
        "On-time portions": delivered,
        "Wasted / late portions": state.total_food - delivered,
        "Unmet demand": sum(max(0, r.need - r.received - got[r.id]) for r in active),
        "Fairness (Jain index)": round(jain, 3),
        "Min fulfilment %": round(100 * min(ratios), 1) if ratios else 0.0,
    }


# --------------------------------------------------------------------------
# Dynamic replanning
# --------------------------------------------------------------------------
def _freeze(state, T):
    """Everything delivered by time T is final; vehicles stay where they last dropped off."""
    if state.plan:
        lots = {l.id: l for l in state.lots}
        recs = {r.id: r for r in state.recipients}
        vehs = {v.id: v for v in state.vehicles}
        for d in state.plan.deliveries:
            if d.status == "ok" and d.t_drop <= T + 1e-9:
                d.status = "done"
                state.done.append(d)
                lots[d.lot_id].portions -= d.qty
                recs[d.recipient_id].received += d.qty
                vehs[d.vehicle_id].node = d.drop_node
                vehs[d.vehicle_id].free_time = d.t_drop
    state.time = T


def _key_map(plan):
    m = defaultdict(lambda: [0, set()])
    for d in plan.deliveries:
        if d.status in ("ok", "failed"):
            k = (d.lot_id, d.recipient_id, d.status)
            m[k][0] += d.qty
            m[k][1].add(d.vehicle_id)
    return m


def _diff(old, new):
    if old is None:
        return []
    old = Plan([d for d in old.deliveries if d.status != "done"])
    a, b, msgs = _key_map(old), _key_map(new), []
    for k in sorted(set(a) | set(b)):
        lot, rid, st = k
        if k in a and k in b:
            if a[k][1] != b[k][1]:
                msgs.append(f"{lot}->{rid}: vehicle {'/'.join(sorted(a[k][1]))} replaced by {'/'.join(sorted(b[k][1]))}.")
            elif a[k][0] != b[k][0]:
                msgs.append(f"{lot}->{rid}: quantity {a[k][0]} -> {b[k][0]}.")
        elif k in b:
            tag = "now AT RISK (deadline cannot be met)" if st == "failed" else "newly planned"
            msgs.append(f"{b[k][0]} portions {lot}->{rid} {tag}.")
        else:
            tag = "no longer at risk" if st == "failed" else "dropped from plan"
            msgs.append(f"{a[k][0]} portions {lot}->{rid} {tag}.")
    return msgs


def replan(state, objective, event=None, T=None):
    """Return a NEW state: freeze completed work, apply the event, search for a new feasible plan."""
    old_plan = state.plan
    s = copy.deepcopy(state)
    T = s.time if T is None else max(T, s.time)
    _freeze(s, T)
    msgs = []
    if event:
        kind = event["type"]
        if kind == "breakdown":
            v = next(v for v in s.vehicles if v.id == event["vehicle"])
            v.available = False
            msgs.append(f"EVENT t={T:.2f}h: vehicle {v.id} broke down; its undelivered food returns to the pool.")
        elif kind == "cancel":
            r = next(r for r in s.recipients if r.id == event["recipient"])
            r.active = False
            msgs.append(f"EVENT t={T:.2f}h: {r.name} can no longer accept deliveries.")
        elif kind == "urgent":
            t = URGENT_TEMPLATE
            lot = FoodLot(f"L{len(s.lots) + 1}", t["donor"], t["node"], t["portions"], t["food_type"], T + t["window_h"])
            s.lots.append(lot)
            s.total_food += lot.portions
            msgs.append(f"EVENT t={T:.2f}h: urgent donation - {lot.portions} {lot.food_type} portions at {lot.donor}, "
                        f"deadline {lot.expiry_h:.2f}h.")
    s.objective = objective
    s.plan = make_plan(s, objective)
    msgs += _diff(old_plan, s.plan)
    risk = sum(d.qty for d in s.plan.deliveries if d.status == "failed")
    msgs.append(f"New plan ({objective}): {len(s.plan.deliveries)} deliveries, {risk} portions at risk.")
    s.log = state.log + msgs
    return s
