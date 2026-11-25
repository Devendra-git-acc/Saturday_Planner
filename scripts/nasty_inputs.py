#!/usr/bin/env python
"""Throw awkward and hostile requests at the agent and print what comes back.

    python scripts/nasty_inputs.py            # rules only, no API key needed
    python scripts/nasty_inputs.py --live     # also use the model if OPENAI_API_KEY is set

Exit code 1 if any request raised, broke an invariant, or got an unexpected status.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.agent import Agent  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.schemas import UserInput  # noqa: E402
from app.trace import Tracer  # noqa: E402
from tests.invariants import check_result, check_trace  # noqa: E402
from tests.nasty import CASES, REJECTED  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="use the model when a key is configured")
    args = ap.parse_args()

    settings = dataclasses.replace(load_settings(), enable_live_data=False)
    llm = build_llm(settings) if args.live else None
    agent = Agent(settings, llm)
    print(f"{len(CASES)} awkward requests, model {'on (' + settings.openai_model + ')' if llm else 'off'}\n")

    failures = 0
    for label, fields, allowed in CASES:
        t0 = time.perf_counter()
        try:
            inp = UserInput(**fields)
            tracer = Tracer()
            res = agent.run(inp, tracer, allow_llm=llm is not None)
            problems = check_result(res, inp) + check_trace(tracer.events)
            if res.status not in allowed:
                problems.append(f"status {res.status}, expected one of {sorted(allowed)}")
            detail = f"{res.status:10} {len(res.stops)} stops  {(res.message or res.headline)[:64]}"
        except Exception as e:  # noqa: BLE001
            problems, detail = [f"raised {type(e).__name__}: {e}"], "raised"
        ms = (time.perf_counter() - t0) * 1000
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {label:44} {ms:6.0f} ms  {detail}")
        for p in problems[:3]:
            print(f"       - {p}")

    print("\nrequests the API refuses with a message:")
    for label, fields, field in REJECTED:
        try:
            UserInput(**fields)
            failures += 1
            print(f"FAIL {label}: accepted")
        except ValidationError as e:
            named = any(field in err["loc"] for err in e.errors())
            failures += not named
            print(f"{'ok  ' if named else 'FAIL'} {label:44} {e.errors()[0]['msg'][:60]}")
    print("\n" + ("all good" if not failures else f"{failures} problem(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
