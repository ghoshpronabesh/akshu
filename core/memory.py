import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_FILE = BASE_DIR / "data" / "memory.json"
ACTIVITY_FILE = BASE_DIR / "data" / "activity.json"


def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "user": {},
            "projects": [],
            "preferences": {},
            "facts": [],
            "conversation": []
        }


def save_memory(memory):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4, ensure_ascii=False)


def remember(category, value):
    memory = load_memory()

    if category not in memory:
        memory[category] = []

    if isinstance(memory[category], list):
        if value not in memory[category]:
            memory[category].append(value)
    else:
        memory[category] = value

    save_memory(memory)
    log_activity("memory", f"Stored: {category} -> {value}")

    return True


def get_memories(category=None):
    memory = load_memory()

    if category:
        return memory.get(category, [])

    return memory


def clear_memory(category=None):
    if category:
        memory = load_memory()

        if category in memory:
            if isinstance(memory[category], list):
                memory[category] = []
            else:
                memory[category] = {}
    else:
        memory = {
            "user": {},
            "projects": [],
            "preferences": {},
            "facts": [],
            "conversation": []
        }

    save_memory(memory)
    log_activity("memory", f"Cleared: {category or 'all memory'}")

    return True


def log_activity(activity_type, message):
    try:
        with open(ACTIVITY_FILE, "r", encoding="utf-8") as file:
            activities = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        activities = []

    activities.append({
        "timestamp": datetime.now().isoformat(),
        "type": activity_type,
        "message": message
    })

    activities = activities[-200:]

    ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ACTIVITY_FILE, "w", encoding="utf-8") as file:
        json.dump(activities, file, indent=4, ensure_ascii=False)
