"""FoodBridge AI - Streamlit dashboard.  Run with:  streamlit run app.py"""
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import make_plan, metrics, new_state, replan, search_comparison
from data import EDGES, NODES

st.set_page_config(page_title="FoodBridge AI", page_icon="🍲", layout="wide")

COLORS = {"V1": "#1f77b4", "V2": "#2ca02c", "V3": "#9467bd"}
KIND_STYLE = {"depot": ("#111111", "square"), "junction": ("#bbbbbb", "circle"),
              "donor": ("#ff7f0e", "diamond"), "recipient": ("#17becf", "star")}

# ---------------------------------------------------------------- session state
if "objective" not in st.session_state:
    st.session_state.objective = "Fair"
if "event_time" not in st.session_state:
    st.session_state.event_time = 0.5
if "state" not in st.session_state:
    st.session_state.state = new_state("Fair")


def on_objective():
    st.session_state.state = replan(st.session_state.state, st.session_state.objective)


def _event(ev):
    s = st.session_state.state
    st.session_state.state = replan(s, s.objective, ev, st.session_state.event_time)


def do_breakdown():
    _event({"type": "breakdown", "vehicle": st.session_state.sel_vehicle})


def do_cancel():
    _event({"type": "cancel", "recipient": st.session_state.sel_recipient})


def do_urgent():
    _event({"type": "urgent"})


def do_reset():
    st.session_state.state = new_state(st.session_state.objective)


state = st.session_state.state
plan = state.plan

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🍲 FoodBridge AI")
    st.caption("Constraint-based allocation + A* routing + dynamic replanning (no ML).")
    st.radio("Objective", ["Fair", "Fastest"], key="objective", on_change=on_objective,
             help="Fair: prioritise recipients with the most unmet need. Fastest: nearest feasible recipient.")
    st.slider("Disruption happens at t = (hours)", 0.0, 3.0, step=0.25, key="event_time",
              help="Deliveries finished before this time are final; the rest is replanned.")
    st.button("Reset scenario", on_click=do_reset)
    st.caption("Synthetic data for demonstration; not real measurements.")

# ---------------------------------------------------------------- helpers
def lots_table():
    rows = []
    for l in state.lots:
        ok = sum(d.qty for d in plan.deliveries if d.lot_id == l.id and d.status == "ok")
        risk = sum(d.qty for d in plan.deliveries if d.lot_id == l.id and d.status == "failed")
        risk += sum(u["qty"] for u in plan.unallocated if u["lot_id"] == l.id)
        delivered = sum(d.qty for d in state.done if d.lot_id == l.id)
        status = "Delivered" if l.portions == 0 else ("At risk" if risk else "Planned")
        rows.append({"Lot": l.id, "Donor": l.donor, "Food": l.food_type, "Delivered": delivered,
                     "Remaining": l.portions, "Planned on time": ok, "At risk / unallocated": risk,
                     "Deadline (h)": round(l.expiry_h, 2), "Status": status})
    return pd.DataFrame(rows)


def recipients_table():
    got = defaultdict(int)
    for d in plan.deliveries:
        if d.status == "ok":
            got[d.recipient_id] += d.qty
    rows = []
    for r in state.recipients:
        rows.append({"Recipient": r.name, "Need": r.need, "Received": r.received, "Planned": got[r.id],
                     "Unmet after plan": max(0, r.need - r.received - got[r.id]),
                     "Active": "yes" if r.active else "CANCELLED"})
    return pd.DataFrame(rows)


def plan_table():
    rows = []
    for d in sorted(state.done + plan.deliveries, key=lambda d: (d.vehicle_id, d.t_drop)):
        rows.append({"Vehicle": d.vehicle_id, "Lot": d.lot_id, "Pickup": NODES[d.pickup_node][0],
                     "Drop-off": d.recipient_name, "Portions": d.qty,
                     "Arrives (h)": round(d.t_drop, 2) if d.status != "failed" else None,
                     "Deadline (h)": round(d.deadline, 2), "Status": d.status.upper(), "Why": d.reason})
    return pd.DataFrame(rows)


def build_map():
    fig = go.Figure()
    ex, ey = [], []
    for a, b in EDGES:
        ex += [NODES[a][2], NODES[b][2], None]
        ey += [NODES[a][3], NODES[b][3], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#dddddd", width=2),
                             hoverinfo="skip", showlegend=False))
    seen = set()
    for d in state.done + plan.deliveries:
        if d.status == "failed":
            a, b = NODES[d.pickup_node], NODES[d.drop_node]
            fig.add_trace(go.Scatter(x=[a[2], b[2]], y=[a[3], b[3]], mode="lines",
                                     line=dict(color="red", width=3, dash="dash"),
                                     name="At risk (deadline missed)", legendgroup="risk",
                                     showlegend="risk" not in seen,
                                     hovertext=d.reason, hoverinfo="text"))
            seen.add("risk")
            continue
        xs = [NODES[n][2] for n in d.path]
        ys = [NODES[n][3] for n in d.path]
        done = d.status == "done"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=COLORS.get(d.vehicle_id, "#444"), width=2 if done else 4, dash="dot" if done else "solid"),
            opacity=0.5 if done else 0.85, name=f"{d.vehicle_id} route", legendgroup=d.vehicle_id,
            showlegend=d.vehicle_id not in seen,
            hovertext=f"{d.vehicle_id}: {d.qty} portions {d.donor} -> {d.recipient_name} (arrives {d.t_drop:.2f}h)",
            hoverinfo="text"))
        seen.add(d.vehicle_id)
    for kind, (color, symbol) in KIND_STYLE.items():
        ids = [n for n, v in NODES.items() if v[1] == kind]
        fig.add_trace(go.Scatter(
            x=[NODES[n][2] for n in ids], y=[NODES[n][3] for n in ids],
            mode="markers+text" if kind != "junction" else "markers",
            text=[NODES[n][0] for n in ids], textposition="top center", textfont=dict(size=10),
            marker=dict(size=16 if kind != "junction" else 8, color=color, symbol=symbol,
                        line=dict(width=1, color="#333")), name=kind.capitalize()))
    inactive = [r.node for r in state.recipients if not r.active]
    if inactive:
        fig.add_trace(go.Scatter(x=[NODES[n][2] for n in inactive], y=[NODES[n][3] for n in inactive],
                                 mode="markers", marker=dict(size=26, color="red", symbol="x"),
                                 name="Cancelled recipient"))
    for v in state.vehicles:
        x, y = NODES[v.node][2], NODES[v.node][3]
        fig.add_trace(go.Scatter(
            x=[x + 0.25], y=[y - 0.45], mode="markers+text", text=[v.id], textposition="bottom right",
            marker=dict(size=14, color="red" if not v.available else COLORS.get(v.id, "#444"),
                        symbol="x" if not v.available else "triangle-up"),
            name=f"{v.id} {'(BROKEN)' if not v.available else 'position'}", showlegend=False))
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x"), legend=dict(orientation="h", y=-0.02))
    return fig


def compare():
    rows = {}
    for mode in ["Baseline", "Fastest", "Fair"]:
        rows[mode] = metrics(state, make_plan(state, mode))
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- header metrics
m = metrics(state, plan)
at_risk = sum(d.qty for d in plan.deliveries if d.status == "failed") + sum(u["qty"] for u in plan.unallocated)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Objective", state.objective)
c2.metric("Simulation time", f"{state.time:.2f} h")
c3.metric("On-time portions", m["On-time portions"])
c4.metric("At risk / wasted", at_risk)
c5.metric("Fairness (Jain)", m["Fairness (Jain index)"])

tab1, tab2, tab3, tab4 = st.tabs(["Donations", "Live Map", "AI vs Baseline", "Disruption Simulator"])

with tab1:
    st.subheader("Food lots")
    st.dataframe(lots_table())
    st.subheader("Recipients")
    st.dataframe(recipients_table())
    st.subheader("Current plan and explanations")
    st.dataframe(plan_table())
    for u in plan.unallocated:
        st.warning(f"{u['qty']} portions from {u['donor']} ({u['lot_id']}) unallocated: {u['reason']}")

with tab2:
    st.plotly_chart(build_map())
    st.caption("Solid lines = planned A* routes, dotted = already completed, red dashed = deliveries at risk, "
               "red X = cancelled recipient or broken vehicle.")

with tab3:
    st.subheader("Baseline vs Fastest vs Fair (on the current state)")
    st.caption("Baseline = input order, nearest recipient takes everything, vehicles round-robin, no deadline check.")
    df = compare()
    st.dataframe(df)
    cols = st.columns(4)
    for col, metric_name in zip(cols, ["Total distance (km)", "On-time portions", "Unmet demand", "Fairness (Jain index)"]):
        f = go.Figure(go.Bar(x=list(df.index), y=df[metric_name], marker_color=["#888", "#1f77b4", "#2ca02c"]))
        f.update_layout(title=metric_name, height=300, margin=dict(l=10, r=10, t=40, b=10))
        col.plotly_chart(f)
    st.subheader("A* vs Dijkstra (nodes expanded, from depot)")
    st.dataframe(pd.DataFrame(search_comparison()))

with tab4:
    st.subheader("Trigger a disruption")
    st.caption("Uses the time set in the sidebar. Finished deliveries stay final; everything else is replanned.")
    b1, b2, b3 = st.columns(3)
    with b1:
        avail = [v.id for v in state.vehicles if v.available]
        st.selectbox("Vehicle", avail, key="sel_vehicle")
        st.button("Vehicle Breakdown", on_click=do_breakdown, disabled=not avail)
    with b2:
        active = {r.id: r.name for r in state.recipients if r.active}
        st.selectbox("Recipient", list(active), format_func=lambda i: active[i], key="sel_recipient")
        st.button("Cancel Recipient", on_click=do_cancel, disabled=not active)
    with b3:
        st.write("A new 30-portion cooked-food donation appears at Hotel F (1.5 h deadline).")
        st.button("Urgent Donation Arrives", on_click=do_urgent)
    st.subheader("What the AI changed")
    if state.log:
        for line in reversed(state.log[-15:]):
            st.write("• " + line)
    else:
        st.info("No disruption yet. Trigger one above.")
