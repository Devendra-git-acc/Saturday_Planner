"""Ready-made inputs shown as buttons in the UI (and reused by tests / the README)."""
from __future__ import annotations

EXAMPLES: list[dict] = [
    {
        "id": "assignment",
        "title": "The assignment example",
        "blurb": "Vegetarian, tired but wants fun, avoiding crowds, with 2,000 rupees and 4 hours",
        "input": {
            "city": "Bangalore", "budget": 2000, "available_time": "4 hours",
            "mood": "tired but wants to do something fun",
            "interests": ["food", "music", "walks"], "constraints": ["vegetarian", "avoid crowded places"],
        },
    },
    {
        "id": "pune-vegan",
        "title": "Pune evening, vegan",
        "blurb": "Written in plain words, vegan, no alcohol, from 5 pm",
        "input": {"free_text": "Relaxed evening in Pune, around Rs 1500, vegan, no alcohol, 3 hours from 5 pm. I like music and good food."},
    },
    {
        "id": "vague",
        "title": "A vague request",
        "blurb": "One line, no city, so the agent asks before planning",
        "input": {"free_text": "plan something fun for me"},
    },
    {
        "id": "tight",
        "title": "Almost impossible",
        "blurb": "Vegan, no crowds, 300 rupees and three evening hours in Gurugram",
        "input": {
            "city": "Gurugram", "budget": 300, "available_time": "3 hours", "start_time": "7:00 pm",
            "interests": ["food", "music", "nightlife"], "constraints": ["vegan", "avoid crowded places", "no alcohol"],
        },
    },
    {
        "id": "other-city",
        "title": "A city with no data",
        "blurb": "Mumbai has no curated data, so the agent tries live map data first",
        "input": {"city": "Mumbai", "budget": 1500, "available_time": "4 hours", "interests": ["food", "walks"]},
    },
]
