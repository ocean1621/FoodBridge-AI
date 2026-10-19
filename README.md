# FoodBridge AI - Intelligent Food Redistribution
🌐 Live Demo: https://ocean1621.github.io/FoodBridge-AI/

An AI planning system that decides **where surplus food should go, which vehicle carries it, and which route it takes**,
and **replans automatically** when something goes wrong (vehicle breakdown, recipient cancels, urgent donation arrives).

This project uses classical AI (search, constraint checking, planning). It does not use machine learning.

## Problem
Given limited vehicles, food quantities, expiry deadlines and recipient needs, what is the best redistribution plan,
and how should it adapt when conditions change?

## Algorithms
| Module | Technique | Where |
|---|---|---|
| Food allocation | Constraint checking + greedy heuristic (most urgent lot first) | `core.py: make_plan` |
| Constraints | Recipient demand, food type accepted, vehicle capacity, expiry deadline | `core.py: _reachable`, `_schedule` step |
| Routing | A* search on a road graph (g = road km, h = straight-line km, admissible) | `core.py: Graph.search` |
| Replanning | Goal-based planning: freeze completed work, update state, search for a new feasible plan | `core.py: replan` |
| Objectives | **Fair** (unmet-need ratio vs distance) and **Fastest** (nearest feasible recipient) | `core.py: make_plan` |

## Evaluation
Compared against a naive **baseline** (input order, nearest recipient takes everything, round-robin vehicles, no deadline check):
total distance, on-time portions, unmet demand, Jain's fairness index and minimum fulfilment %.
The dashboard also compares A* against Dijkstra (nodes expanded).

Observed trade-off: Fair mode spreads food across more recipients, which improves fairness but increases travel distance.

## Run it
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `data.py` - synthetic scenario (5 donors, 4 recipients, 3 vehicles, 15-node road graph)
- `core.py` - allocation, A*, replanning, metrics, baseline
- `app.py` - Streamlit dashboard (Donations, Live Map, AI vs Baseline, Disruption Simulator)

## Limitations
- All data is synthetic and illustrative; safe-food deadlines depend on real handling conditions.
- Greedy allocation is a heuristic, not a guaranteed optimum.
- Vehicles are assumed to be at their last drop-off point when a disruption happens (no mid-route positions).
- Food redistribution optimisation is an existing research area; this project is a small, explainable implementation.
