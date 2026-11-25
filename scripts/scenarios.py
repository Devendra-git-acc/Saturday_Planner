"""Run the deterministic (no-LLM) pipeline on a battery of scenarios and print the plans.
Usage: python scripts/scenarios.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys
from app.schemas import UserInput
from app.cities import get_city
from app.tools.preferences import rules_parse, finalize_defaults, set_window
from app.tools.options import get_activity_options, get_food_options
from app.tools.planner import rule_plan, choose_start
from app.tools.costing import estimate_cost
from app.tools.validate import validate_plan
from app.util import fmt12, fmt_inr

def run(label, d):
    prefs = finalize_defaults(rules_parse(UserInput(**d)))
    city = get_city(prefs.city)
    acts, foods = get_activity_options(prefs, city), get_food_options(prefs, city)
    if prefs.start_min is None:
        s, scores = choose_start(prefs, city, acts.candidates, foods.candidates)
        set_window(prefs, s)
        acts, foods = get_activity_options(prefs, city), get_food_options(prefs, city)
    plan = rule_plan(prefs, city, acts.candidates, foods.candidates)
    print(f"=== {label}  (start {fmt12(prefs.start_min)})")
    if not plan: print("  NO PLAN"); return
    rep = estimate_cost(city, prefs, plan.stops); v = validate_plan(rep, prefs, city)
    for s in rep.stops:
        print(f"  {fmt12(s.start):>8}-{fmt12(s.end):<8} {s.name[:40]:40s} {s.kind[:4]} {fmt_inr(s.cost):>6} +{s.travel.minutes:>2}m/{fmt_inr(s.travel.cost)} {s.crowd_level}")
    print(f"  total {fmt_inr(rep.total_cost)}/{fmt_inr(prefs.budget_inr)} {rep.total_minutes}/{prefs.minutes}min ok={v.ok} clean={v.clean} pen={v.penalty}", [(i.code) for i in v.issues])

run("assignment example", {"city":"Bangalore","budget":2000,"available_time":"4 hours","mood":"tired but wants to do something fun","interests":["food","music","walks"],"constraints":["vegetarian","avoid crowded places"]})
run("blr evening art+coffee", {"city":"Bangalore","budget":1200,"available_time":"3 hours","start_time":"4 pm","interests":["art","coffee"],"constraints":["quiet"]})
run("pune history+food", {"city":"Pune","budget":1800,"available_time":"5 hours","interests":["history","food","books"],"constraints":["vegetarian"]})
run("pune vegan evening", {"city":"Pune","free_text":"relaxed evening, Rs 1500, vegan, no alcohol, 3 hours from 5 pm","interests":["music","food"]})
run("gurugram nature", {"city":"Gurugram","budget":1500,"available_time":"half day","interests":["nature","food"],"constraints":["vegetarian"]})
run("energetic parvati", {"city":"Pune","budget":800,"available_time":"3 hours","mood":"energetic, adventurous","interests":["walks","nature"]})
