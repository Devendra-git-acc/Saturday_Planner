#!/usr/bin/env python
"""Check the real OpenAI connection with YOUR key, end to end, in about ten seconds.

    export OPENAI_API_KEY=sk-...        # or put it in .env (never commit it)
    python scripts/smoke_llm.py
    python scripts/smoke_llm.py --model some-other-model
    python scripts/smoke_llm.py --api responses

It (1) sends one tiny JSON request, then (2) plans the assignment's example Saturday with the model
switched on and prints what the model did. Exit code 0 only if the model really took part.
The key is never printed.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.llm import LLMBadOutput, LLMUnavailable, build_llm  # noqa: E402
from app.schemas import UserInput  # noqa: E402
from app.trace import Tracer  # noqa: E402

EXAMPLE = {
    "city": "Bangalore", "budget": 2000, "available_time": "4 hours",
    "mood": "tired but wants to do something fun", "interests": ["food", "music", "walks"],
    "constraints": ["vegetarian", "avoid crowded places"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="override OPENAI_MODEL for this check")
    ap.add_argument("--api", choices=["chat", "responses"], help="override OPENAI_API_MODE for this check")
    args = ap.parse_args()

    settings = load_settings()
    if args.model:
        settings = dataclasses.replace(settings, openai_model=args.model)
    if args.api:
        settings = dataclasses.replace(settings, openai_api_mode=args.api)
    settings = dataclasses.replace(settings, enable_live_data=False)
    if not settings.openai_api_key:
        print("No OPENAI_API_KEY found in the environment or .env. Set it and run again.")
        return 2

    llm = build_llm(settings)
    if llm is None:
        print("Could not create the OpenAI client (is the `openai` package installed? pip install -r requirements.txt).")
        return 2
    print(f"model: {settings.openai_model}   api: {settings.openai_api_mode}   key: set ({len(settings.openai_api_key)} characters)\n")

    print("1) one small JSON request")
    try:
        res = llm.complete_json("ping", [
            {"role": "system", "content": "Reply with one JSON object and nothing else."},
            {"role": "user", "content": 'Return {"ok": true, "word": "saturday"}'},
        ], effort="low")
    except (LLMUnavailable, LLMBadOutput) as e:
        print(f"   FAILED: {e}")
        print("   Common causes: wrong key, no access to this model (try --model), no network, quota. Set OPENAI_MODEL to a model your account can use.")
        return 1
    print(f"   ok: {res.data}  ({res.tokens} tokens, {res.ms} ms)\n")

    print("2) the assignment's example Saturday, model on")
    tracer = Tracer()
    out = Agent(settings, llm).run(UserInput(**EXAMPLE), tracer, allow_llm=True)
    for e in tracer.events:
        print(f"   {e['state']:8} {e['label']:28} {e['summary'][:90]}")
    print(f"\nstatus={out.status}  mode={out.mode}  model calls={out.llm.calls}  tokens={out.llm.tokens}  total time={out.elapsed_ms} ms")
    if out.llm.notes:
        print("notes:", "; ".join(out.llm.notes))
    print("headline:", out.headline)
    if out.mode != "ai":
        print("\nThe plan was produced WITHOUT the model (basic mode). Read the notes above for why.")
        return 1
    print("\nThe model took part. Open the app and compare: the status line should say AI planning is on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
