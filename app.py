"""AKSHU 5.0 cloud dashboard. No camera, OpenCV, YOLO or server-side TTS required."""
import hmac
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
AGENT_TOKEN = os.environ.get("AKSHU_AGENT_TOKEN", "").strip()
MAX_EVENTS = 120
_state_lock = threading.RLock()
_state = {
    "mode": "idle", "agent_connected": False, "agent_last_seen": None,
    "camera_status": "Waiting for local agent", "model_status": "Not connected",
    "fps": 0.0, "detections": [], "frame": None, "last_error": "",
    "updated_at": None, "events": deque(maxlen=MAX_EVENTS), "next_event_id": 1,
}


def _log(message, level="INFO", source="system"):
    with _state_lock:
        event = {"id": _state["next_event_id"], "time": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
                 "level": level, "source": source, "message": str(message)[:500]}
        _state["next_event_id"] += 1
        _state["events"].append(event)
    print(f"[{event['time']}] [{level}] [{source}] {event['message']}", flush=True)
    return event


def _agent_authorized():
    if not AGENT_TOKEN:
        return False
    supplied = request.headers.get("Authorization", "")
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    else:
        supplied = request.headers.get("X-AKSHU-TOKEN", "").strip()
    return bool(supplied) and hmac.compare_digest(supplied, AGENT_TOKEN)


def _public_state():
    now = time.time()
    seen = _state["agent_last_seen"]
    connected = bool(seen and now - seen < 20)
    _state["agent_connected"] = connected
    return {
        "mode": _state["mode"], "agent_connected": connected,
        "agent_last_seen": datetime.fromtimestamp(seen, timezone.utc).astimezone().isoformat(timespec="seconds") if seen else None,
        "camera_status": _state["camera_status"] if connected else ("Agent offline" if seen else "Waiting for local agent"),
        "model_status": _state["model_status"], "fps": round(float(_state["fps"]), 1),
        "detections": list(_state["detections"]), "frame": _state["frame"],
        "last_error": _state["last_error"], "updated_at": _state["updated_at"],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "akshu-5-dashboard", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.get("/api/status")
def api_status():
    with _state_lock:
        return jsonify(_public_state())


@app.get("/api/events")
def api_events():
    try:
        after = max(0, int(request.args.get("after", "0")))
    except ValueError:
        return jsonify({"error": "after must be an integer"}), 400
    with _state_lock:
        events = [e for e in _state["events"] if e["id"] > after]
        latest = _state["next_event_id"] - 1
    return jsonify({"events": events, "latest_id": latest})


@app.post("/api/agent/update")
def agent_update():
    if not _agent_authorized():
        return jsonify({"error": "Agent authentication failed. Check AKSHU_AGENT_TOKEN."}), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400
    detections = payload.get("detections", [])
    if not isinstance(detections, list) or len(detections) > 100:
        return jsonify({"error": "detections must be a list of at most 100 items."}), 400
    clean = []
    for item in detections:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        clean.append({"name": item["name"][:80], "confidence": round(confidence, 2),
                      "position": str(item.get("position", "unknown"))[:20]})
    try:
        fps_value = max(0.0, min(240.0, float(payload.get("fps", 0) or 0)))
    except (TypeError, ValueError):
        fps_value = 0.0
    announcements = payload.get("announcements", [])
    if not isinstance(announcements, list):
        announcements = []
    announcements = [str(x).strip()[:300] for x in announcements if str(x).strip()][:8]
    frame = payload.get("frame")
    if frame is not None and (not isinstance(frame, str) or len(frame) > 1_500_000):
        return jsonify({"error": "frame must be a base64 JPEG string under 1.5 MB."}), 400
    with _state_lock:
        _state.update({"mode": "live", "agent_last_seen": time.time(), "camera_status": str(payload.get("camera_status", "Connected"))[:80],
                       "model_status": str(payload.get("model_status", "Ready"))[:80], "fps": fps_value,
                       "detections": clean, "last_error": str(payload.get("error", ""))[:300],
                       "updated_at": datetime.now(timezone.utc).isoformat()})
    if frame is not None:
        with _state_lock:
            _state["frame"] = frame
    for message in announcements:
        _log(message, "GUIDANCE", "live-agent")
    return jsonify({"ok": True, "accepted_detections": len(clean), "accepted_announcements": len(announcements)})


@app.post("/api/agent/heartbeat")
def agent_heartbeat():
    if not _agent_authorized():
        return jsonify({"error": "Agent authentication failed."}), 401
    with _state_lock:
        _state["agent_last_seen"] = time.time()
        _state["mode"] = "live"
    return jsonify({"ok": True})


@app.post("/api/agent/disconnect")
def agent_disconnect():
    if not _agent_authorized():
        return jsonify({"error": "Agent authentication failed."}), 401
    with _state_lock:
        _state["agent_last_seen"] = None
        _state["camera_status"] = "Disconnected"
        _state["detections"] = []
        _state["frame"] = None
    _log("Local vision agent disconnected.", "WARNING", "live-agent")
    return jsonify({"ok": True})


@app.post("/api/demo/event")
def demo_event():
    # Demo events are generated in the browser; this route is retained for optional diagnostics only.
    return jsonify({"ok": True})


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Request too large."}), 413


@app.errorhandler(500)
def server_error(_error):
    app.logger.exception("Unhandled dashboard error")
    return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")), threaded=True)
