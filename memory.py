import json
import os
from datetime import datetime

MEMORY_FILE = "memory_store.json"


def _load() -> list[dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(entries: list[dict]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def remember(question: str, lesson: str) -> None:
    entries = _load()
    entries.append({
        "question": question,
        "lesson": lesson,
        "timestamp": datetime.now().isoformat(),
    })
    _save(entries)
    print(f" SAVED MEMORY: {lesson}")


def recall(question: str, limit: int = 3) -> str:
    entries = _load()
    if not entries:
        return "No past lessons yet."

    q_words = set(question.lower().split())

    def score(entry):
        return len(q_words & set(entry["question"].lower().split()))

    ranked = sorted(entries, key=score, reverse=True)
    top = ranked[:limit]

    if not top or score(top[0]) == 0:
        return "No clearly related past lessons."

    return "\n".join(f"- {e['lesson']}" for e in top)