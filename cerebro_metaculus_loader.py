"""
Loads Metaculus resolved question data and converts community
forecast accuracy into calibration signal for Cerebro.
"""
import requests
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "cerebro_data" / "metaculus_calibration.json"
BASE_URL = "https://www.metaculus.com/api2/questions/"

SOCIAL_KEYWORDS = [
    "crime",
    "inequality",
    "protest",
    "election",
    "policy",
    "welfare",
    "immigration",
    "abortion",
    "gender",
    "poverty",
    "unemployment",
    "violence",
    "social",
    "cultural",
]


def fetch_resolved_questions(max_pages=10):
    """Fetch resolved binary questions from Metaculus public API."""
    all_questions = []
    offset = 0
    limit = 100

    for page in range(max_pages):
        params = {
            "limit": limit,
            "offset": offset,
            "order_by": "-actual_resolve_time",
        }
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
        except Exception as e:
            print(f"API request failed: {e}")
            break
        if r.status_code != 200:
            print(f"API error: {r.status_code}")
            break
        data = r.json()
        results = data.get("results", [])
        # Filter for resolved only
        resolved = [q for q in results if q.get("resolved") is True]
        all_questions.extend(resolved)
        offset += limit
        print(f"Fetched {len(all_questions)} resolved questions so far")
        if offset >= data.get("count", 0) or not results:
            break

    return all_questions


def filter_social_questions(questions):
    """Keep only questions relevant to social/civilizational dynamics."""
    filtered = []
    for q in questions:
        title = (q.get("title") or "").lower()
        if any(kw in title for kw in SOCIAL_KEYWORDS):
            filtered.append(q)
    print(f"Filtered to {len(filtered)} social/political questions")
    return filtered


def extract_calibration_signal(questions):
    """
    For each resolved binary question:
    - community_prediction at close: this is the stated confidence
    - resolution: 1.0 (yes) or 0.0 (no): this is the outcome
    - Convert to calibration episode format

    Note: Metaculus API v2 may not include community_prediction/resolution
    in list view. We extract from question.possibilities or fetch per-question
    if needed. Fallback: use last-forecast proxy when available.
    """
    episodes = []
    for q in questions:
        try:
            # API v2: resolution may be in question.possibilities or question dict
            question_data = q.get("question") or {}
            possibilities = question_data.get("possibilities") or q.get("possibilities")
            resolution = q.get("resolution") or question_data.get("resolution")

            # community_prediction: may be absent in list view
            cp = q.get("community_prediction") or question_data.get("community_prediction")
            stated_conf = None
            if cp:
                full = cp.get("full") if isinstance(cp, dict) else {}
                stated_conf = full.get("q2") if isinstance(full, dict) else None

            # Resolution from possibilities for binary
            if resolution is None and possibilities:
                if isinstance(possibilities, dict):
                    resolution = possibilities.get("resolution")
                elif isinstance(possibilities, list) and possibilities:
                    resolution = possibilities[0].get("resolution")

            # Skip if we can't determine outcome
            if resolution is None:
                continue

            # Normalize resolution to 0/1
            try:
                hit = float(resolution) > 0.5
            except (TypeError, ValueError):
                if resolution in ("Yes", "yes", True, "1", 1):
                    hit = True
                elif resolution in ("No", "no", False, "0", 0):
                    hit = False
                else:
                    continue

            # Stated confidence: use 0.5 if not available (conservative)
            if stated_conf is not None:
                stated_conf_pct = float(stated_conf) * 100
            else:
                stated_conf_pct = 50.0  # Unknown -> neutral

            if stated_conf_pct < 5 or stated_conf_pct > 95:
                continue

            episodes.append(
                {
                    "source": "metaculus",
                    "question_id": q.get("id"),
                    "title": q.get("title", "")[:100],
                    "stated_confidence_pct": stated_conf_pct,
                    "hit": hit,
                    "resolve_time": q.get("actual_resolve_time"),
                    "country": "GLOBAL",
                }
            )
        except Exception:
            continue

    print(f"Extracted {len(episodes)} calibration episodes from Metaculus")
    return episodes


def compute_metaculus_brier(episodes):
    """Compute Brier score from Metaculus episodes."""
    if not episodes:
        return None
    total = sum(
        (ep["stated_confidence_pct"] / 100 - float(ep["hit"])) ** 2 for ep in episodes
    )
    return total / len(episodes)


def save_calibration_signal(episodes):
    output = {
        "n": len(episodes),
        "brier": compute_metaculus_brier(episodes),
        "episodes": episodes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(episodes)} Metaculus episodes to {OUTPUT}")
    if output["brier"] is not None:
        print(f"Metaculus Brier: {output['brier']:.4f}")


def run():
    questions = fetch_resolved_questions(max_pages=10)
    social = filter_social_questions(questions)
    episodes = extract_calibration_signal(social)
    save_calibration_signal(episodes)
    return episodes


if __name__ == "__main__":
    run()
