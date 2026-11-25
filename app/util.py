"""Small pure helpers: clock/duration/money parsing and formatting."""
from __future__ import annotations

import re
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt

# --------------------------------------------------------------------- clock


def hm_to_min(s: str) -> int:
    h, m = s.strip().split(":")
    return int(h) * 60 + int(m)


def min_to_hm(m: int) -> str:
    m = int(m) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def fmt12(m: int) -> str:
    m = int(m) % 1440
    h, mm = divmod(m, 60)
    return f"{h % 12 or 12}:{mm:02d} {'am' if h < 12 else 'pm'}"


def plural(n: int, one: str, many: str | None = None) -> str:
    """'1 stop', '3 stops', '2 activities'."""
    return f"{n} {one if n == 1 else (many or one + 's')}"


def fmt_minutes(m: float) -> str:
    m = int(round(m))
    if m < 60:
        return f"{m} min"
    h, mm = divmod(m, 60)
    return f"{h} h" if mm == 0 else f"{h} h {mm} min"


def fmt_inr(n: float) -> str:
    """Rupees with Indian digit grouping: 2000 -> ₹2,000, 125000 -> ₹1,25,000."""
    v = int(round(n))
    sign = "-" if v < 0 else ""
    s = str(abs(v))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"{sign}₹{s}"


@lru_cache(maxsize=4096)
def parse_windows(spec: str) -> tuple[tuple[int, int], ...]:
    """'07:00-11:30,15:00-20:00' -> ((420, 690), (900, 1200))"""
    out: list[tuple[int, int]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        out.append((hm_to_min(a), hm_to_min(b)))
    return tuple(out)


def overlap_minutes(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = radians(lat1), radians(lat2)
    dphi = p2 - p1
    dlmb = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


# ----------------------------------------------------------------- parsing

_N = r"(\d+(?:\.\d+)?)"
_H = r"(?:hours?|hrs?|h)(?![a-z])"
_M = r"(?:minutes?|mins?|m)(?![a-z])"


def parse_duration_minutes(text: str | None) -> int | None:
    """'4 hours' -> 240, 'half day' -> 300, '90 min' -> 90, '2-3 hrs' -> 150."""
    if text is None:
        return None
    s = str(text).lower().strip()
    if not s:
        return None
    if re.search(r"\bhalf[\s-]?(?:a\s+)?day\b", s):
        return 300
    if re.search(r"\b(?:full|whole|entire)[\s-]?day\b|\ball[\s-]?day\b", s):
        return 480
    if re.search(r"\bcouple of hours\b", s):
        return 120
    if re.search(r"\b(?:a\s+)?few hours\b", s):
        return 180
    if re.search(r"\b(?:an?\s+)?hour and a half\b", s):
        return 90
    m = re.search(rf"{_N}\s*(?:-|–|to)\s*{_N}\s*{_H}", s)
    if m:
        return int(round((float(m.group(1)) + float(m.group(2))) / 2 * 60))
    m = re.search(rf"{_N}\s*{_H}(?:\s*(?:and\s*)?{_N}\s*{_M})?", s)
    if m:
        mins = float(m.group(1)) * 60 + (float(m.group(2)) if m.group(2) else 0)
        return int(round(mins))
    m = re.search(rf"{_N}\s*{_M}", s)
    if m:
        return int(round(float(m.group(1))))
    if re.search(r"\ban?\s+hour\b", s):
        return 60
    m = re.fullmatch(rf"\s*{_N}\s*", s)
    if m:
        v = float(m.group(1))
        if 0 < v <= 12:
            return int(round(v * 60))
        if 20 <= v <= 720:
            return int(round(v))
    return None


_DAYPARTS = {
    "noon": 12 * 60,
    "midday": 12 * 60,
    "morning": 9 * 60,
    "afternoon": 14 * 60,
    "evening": 17 * 60,
    "night": 19 * 60,
}


# A number followed by one of these is a quantity ("3 hours", "30 min", "2 km", "4 people"), not a time of day.
_QUANTITY_AFTER = re.compile(
    r"\s*(?:hours?|hrs?|h|minutes?|mins?|m|km|kms|kilomet\w+|people|persons?|pax|friends?|stops?|days?|rs|inr|rupees?|k|l|lakh|lac)(?![a-z])"
)
_CURRENCY_BEFORE = re.compile(r"(?:₹|\brs\.?|\binr|\brupees?)\s*$")
_DECIMAL_AFTER = re.compile(r"[.,]\d")  # '1,500', '1.5k'
# a bare number counts as a time only when the words before it say so: "at 4", "starting 10", "from 5"
_CUE_BEFORE = re.compile(r"\b(?:at|from|after|around|between|starting|starts?|begin(?:ning)?|leav(?:e|ing)|depart\w*)\s+$")
_CLOCK_RX = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\d])")
_RANGE_AFTER = re.compile(r"\s*(?:-|–|—|to|till|until)\s*(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)")


def _hm(h: int, mi: int, ap: str) -> int | None:
    if mi > 59 or h > 24:
        return None
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif not ap and h <= 7:  # "start at 4" on a Saturday outing means 4 PM
        h += 12
    total = h * 60 + mi
    return total if total < 1440 else None


def parse_clock(text: str | None) -> int | None:
    """'4 pm' -> 960, '16:30' -> 990, '3 hours, starting at 4pm' -> 960, 'evening' -> 1020.

    Looks at every number in the text and keeps the first one that is really a time of day: it has
    am/pm or minutes, the whole text is just that number, or words like "at"/"from" lead into it.
    Quantities ("3 hours", "Rs 1,500", "2 km") are skipped. A bare number reads as PM up to 7.
    """
    if not text:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    bare_whole = re.fullmatch(r"\d{1,2}", s) is not None
    for m in _CLOCK_RX.finditer(s):
        before, after = s[: m.start()], s[m.end():]
        if _QUANTITY_AFTER.match(after) or _DECIMAL_AFTER.match(after) or _CURRENCY_BEFORE.search(before):
            continue
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        ap = (m.group(3) or "").replace(".", "")
        if not ap:
            # '4-7pm': the first number borrows the second one's am/pm, unless that would end before it starts
            r = _RANGE_AFTER.match(after)
            if r:
                end_ap = r.group(3).replace(".", "")
                end_t = _hm(int(r.group(1)), int(r.group(2) or 0), end_ap)
                for cand in (end_ap, "am" if end_ap == "pm" else "pm"):
                    t = _hm(h, mi, cand)
                    if t is not None and end_t is not None and t < end_t:
                        return t
        if m.group(2) or ap or bare_whole or _CUE_BEFORE.search(before):
            t = _hm(h, mi, ap)
            if t is not None:
                return t
    for word, val in _DAYPARTS.items():
        if re.search(rf"\b{word}\b", s):
            return val
    return None


_RANGE_CUE = re.compile(r"\b(?:from|between|free|available|window)\s+(?:about\s+|around\s+)?$")
_CLOCK_RANGE_RX = re.compile(
    r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\s*(?:-|–|—|to|till|until)\s*"
    r"(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\d])"
)


def parse_clock_range(text: str | None) -> tuple[int, int] | None:
    """'free 4-7pm' -> (960, 1140); '10am to 2pm' -> (600, 840). None unless it is clearly a time window.

    Needs am/pm, minutes ('16:00-19:00') or a lead-in like 'from'/'between'/'free', so '2-3 hours' and
    '5-6 people' are not read as clock times. A missing am/pm is borrowed from the other end.
    """
    if not text:
        return None
    s = str(text).lower()
    for m in _CLOCK_RANGE_RX.finditer(s):
        before, after = s[: m.start()], s[m.end():]
        if _QUANTITY_AFTER.match(after) or _DECIMAL_AFTER.match(after) or _CURRENCY_BEFORE.search(before):
            continue
        h1, m1, a1, h2, m2, a2 = m.groups()
        a1, a2 = (a1 or "").replace(".", ""), (a2 or "").replace(".", "")
        if not (a1 or a2 or m1 or m2 or _RANGE_CUE.search(before)):
            continue
        opposite = {"am": "pm", "pm": "am"}
        starts = [a1] if a1 else ([a2, opposite[a2]] if a2 else [""])
        ends = [a2] if a2 else ([a1, opposite[a1]] if a1 else [""])
        for ap1 in starts:
            for ap2 in ends:
                t1, t2 = _hm(int(h1), int(m1 or 0), ap1), _hm(int(h2), int(m2 or 0), ap2)
                if t1 is not None and t2 is not None and t2 > t1:
                    return t1, t2
    return None


def parse_money(text: str | None) -> int | None:
    """Finds a rupee amount inside free text: 'under Rs 1,500', '2k', '₹800'."""
    if not text:
        return None
    s = str(text).lower().replace(",", "")
    pats = [
        rf"(?:₹|rs\.?|inr|rupees?)\s*{_N}\s*(k|l|lakh)?(?![a-z])",
        rf"{_N}\s*(k|l|lakh)?\s*(?:₹|rs\.?|inr|rupees?)(?![a-z])",
        rf"{_N}\s*(k)(?![a-z])",
        rf"{_N}\s*(lakh|lac)s?(?![a-z])",
        rf"budget\s*(?:of|is|around|about|:)?\s*{_N}\s*(k)?",
    ]
    for p in pats:
        m = re.search(p, s)
        if m:
            n = float(m.group(1))
            suf = m.group(2)
            if suf == "k":
                n *= 1000
            elif suf in ("l", "lakh", "lac"):
                n *= 100000
            return int(round(n))
    return None


def to_number(v) -> float | None:
    """Lenient number coercion for form input: 2000, '2,000', '₹2000', '2k'."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace(",", "")
    if s in {"", "none", "n/a", "na", "any", "flexible", "no limit", "unlimited"}:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|l|lakh|lac)?(?![a-z])", s)
    if not m:
        raise ValueError("Budget must be a number, for example 2000")
    n = float(m.group(1))
    if m.group(2) == "k":
        n *= 1000
    elif m.group(2) in ("l", "lakh", "lac"):
        n *= 100000
    return n


def clean_label(text: object, limit: int = 40) -> str:
    """Make user-supplied text safe to echo back: letters, digits and basic punctuation only.

    Returns '' when almost nothing readable is left (emoji, markup, punctuation), so callers can
    treat it as 'not given' instead of repeating junk back to the person.
    """
    s = re.sub(r"[^\w\s,.'&()-]", " ", str(text or ""))
    s = re.sub(r"[\s_]+", " ", s).strip(" ,.-")
    s = s[:limit].strip(" ,.-")
    return s if sum(ch.isalpha() for ch in s) >= 2 else ""


def short(e: BaseException | str, n: int = 160) -> str:
    s = str(e).strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"
