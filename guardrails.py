import json
import os
from datetime import datetime
from urllib.parse import urlparse

LOG_FILE = "safety_log.json"
MAX_TOOL_CALLS_PER_RUN = 12
BLOCKED_DOMAINS = ["localhost", "127.0.0.1", "internal", "admin"]


def _log(event: dict) -> None:
    entries = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            entries = json.load(f)
    event["timestamp"] = datetime.now().isoformat()
    entries.append(event)
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def check_call(tool_name: str, args: dict, call_count: int) -> tuple[bool, str]:
    if call_count >= MAX_TOOL_CALLS_PER_RUN:
        reason = f"Blocked: exceeded max {MAX_TOOL_CALLS_PER_RUN} tool calls for this run."
        _log({"tool": tool_name, "args": args, "allowed": False, "reason": reason})
        return False, reason

    if tool_name == "fetch_page":
        url = args.get("url", "")
        host = urlparse(url).netloc.lower()
        if not url.startswith(("http://", "https://")):
            reason = "Blocked: fetch_page URL must start with http:// or https://"
            _log({"tool": tool_name, "args": args, "allowed": False, "reason": reason})
            return False, reason
        if any(bad in host for bad in BLOCKED_DOMAINS):
            reason = f"Blocked: domain '{host}' is not allowed."
            _log({"tool": tool_name, "args": args, "allowed": False, "reason": reason})
            return False, reason

    if tool_name == "calculator":
        expr = args.get("expression", "")
        if len(expr) > 200:
            reason = "Blocked: calculator expression too long / suspicious."
            _log({"tool": tool_name, "args": args, "allowed": False, "reason": reason})
            return False, reason

    _log({"tool": tool_name, "args": args, "allowed": True, "reason": "ok"})
    return True, "ok"


def ask_human_approval(action_description: str) -> bool:
    print(f"\n APPROVAL NEEDED: {action_description}")
    answer = input("Approve? (y/n): ").strip().lower()
    approved = answer == "y"
    _log({"tool": "human_approval", "args": {"action": action_description}, "allowed": approved, "reason": "human decision"})
    return approved