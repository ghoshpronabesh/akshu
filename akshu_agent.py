"""AKSHU 5.0 Windows camera/YOLO agent. Browser performs all cloud-demo speech."""
import argparse
import base64
import os
import sys
import time
from collections import Counter

import cv2
import requests
from ultralytics import YOLO

CAMERA_INDEX = int(os.getenv("AKSHU_CAMERA_INDEX", "0"))
MODEL_NAME = os.getenv("AKSHU_MODEL", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("AKSHU_CONFIDENCE", "0.45"))
FRAME_WIDTH = int(os.getenv("AKSHU_FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("AKSHU_FRAME_HEIGHT", "480"))
INFERENCE_SIZE = int(os.getenv("AKSHU_INFERENCE_SIZE", "640"))
IMPORTANT_OBJECTS = {"person", "car", "motorcycle", "bicycle", "bus", "truck", "dog", "chair", "bench", "suitcase", "backpack"}


def determine_position(center_x, frame_width):
    if center_x < frame_width * 0.38:
        return "left"
    if center_x > frame_width * 0.62:
        return "right"
    return "ahead"


def detect_objects(model, frame):
    """Adapted from akshu_navigation_5_0.py; retains its YOLO boxes/position logic."""
    results = model.predict(source=frame, imgsz=INFERENCE_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    if not results or results[0].boxes is None:
        return detections
    height, width = frame.shape[:2]
    for box in results[0].boxes:
        try:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = str(model.names[class_id]).lower()
            center_x = (x1 + x2) / 2
            detections.append({"name": name, "confidence": round(confidence, 2),
                               "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                               "position": determine_position(center_x, width)})
        except Exception as exc:
            print(f"[AKSHU DETECTION WARNING] {exc}", flush=True)
    return detections


def describe_scene(detections):
    if not detections:
        return "No recognised objects detected."
    counts = Counter(item["name"] for item in detections)
    parts = [name if count == 1 else f"{count} {name}s" for name, count in sorted(counts.items())]
    if len(parts) == 1:
        return f"I can see {parts[0]}."
    return "I can see " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."


def guidance_messages(detections):
    warnings = []
    for item in detections:
        name, pos = item["name"], item["position"]
        if name not in IMPORTANT_OBJECTS:
            continue
        if name == "car":
            message = f"Car detected {pos}. Please check your path."
        elif name == "person":
            message = f"Person detected {pos}."
        else:
            message = f"{name.capitalize()} detected {pos}."
        if message not in warnings:
            warnings.append(message)
    if warnings:
        return [" ".join(warnings[:3])]
    return [describe_scene(detections)] if detections else ["No recognised objects detected."]


def draw_detections(frame, detections):
    for item in detections:
        label = f"{item['name']} {item['confidence']:.2f} | {item['position']}"
        cv2.rectangle(frame, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (0, 220, 160), 2)
        cv2.putText(frame, label, (item["x1"], max(20, item["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 160), 2)
    return frame


class CloudAgent:
    def __init__(self, dashboard_url, token, send_frame=True):
        self.base = dashboard_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        self.send_frame = send_frame
        self.last_signature = None
        self.last_announcement_at = 0.0
        self.last_upload_at = 0.0
        self.last_heartbeat_at = 0.0
        self.timeout = (4, 12)

    def post(self, path, payload=None):
        response = self.session.post(self.base + path, json=payload or {}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def update(self, detections, fps, annotated_frame, announcements, camera_status="Connected"):
        now = time.monotonic()
        if now - self.last_upload_at < 0.8 and not announcements:
            if now - self.last_heartbeat_at > 5:
                self.post("/api/agent/heartbeat")
                self.last_heartbeat_at = now
            return
        frame_data = None
        if self.send_frame and (now - self.last_upload_at >= 1.0):
            ok, encoded = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ok:
                frame_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        payload = {"camera_status": camera_status, "model_status": "Ready", "fps": round(fps, 1),
                   "detections": [{"name": d["name"], "confidence": d["confidence"], "position": d["position"]} for d in detections],
                   "announcements": announcements}
        if frame_data is not None:
            payload["frame"] = frame_data
        self.post("/api/agent/update", payload)
        self.last_upload_at = now
        self.last_heartbeat_at = now

    def disconnect(self):
        try:
            self.post("/api/agent/disconnect")
        except Exception as exc:
            print(f"[AKSHU] Could not notify cloud about disconnect: {exc}", flush=True)


def open_camera():
    if os.name == "nt":
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if camera.isOpened():
            return camera
        camera.release()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}. Check Windows camera permissions or AKSHU_CAMERA_INDEX.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def main():
    parser = argparse.ArgumentParser(description="AKSHU 5.0 Windows live vision agent")
    parser.add_argument("--url", default=os.getenv("AKSHU_DASHBOARD_URL"), help="Public Render dashboard URL")
    parser.add_argument("--token", default=os.getenv("AKSHU_AGENT_TOKEN"), help="Agent token; keep private")
    parser.add_argument("--no-preview", action="store_true", help="Send detections only, not camera preview images")
    args = parser.parse_args()
    if not args.url or not args.token:
        parser.error("Set AKSHU_DASHBOARD_URL and AKSHU_AGENT_TOKEN, or pass --url and --token.")
    if not args.url.startswith("https://") and "localhost" not in args.url and "127.0.0.1" not in args.url:
        parser.error("Use an https:// dashboard URL for public deployment.")

    cloud = CloudAgent(args.url, args.token, send_frame=not args.no_preview)
    camera = None
    print("\nAKSHU 5.0 — WINDOWS LIVE VISION AGENT", flush=True)
    print(f"Dashboard: {args.url}", flush=True)
    print("Browser speech is handled by the dashboard. No local TTS engine is required.", flush=True)
    try:
        print(f"Loading YOLO model: {MODEL_NAME}", flush=True)
        model = YOLO(MODEL_NAME)
        print("YOLO model ready. Opening camera…", flush=True)
        camera = open_camera()
        print("Camera connected. Press Ctrl+C to stop.", flush=True)
        previous = time.monotonic()
        smoothed_fps = 0.0
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[WARNING] Camera frame unavailable; retrying.", flush=True)
                time.sleep(0.2)
                continue
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            detections = detect_objects(model, frame)
            now = time.monotonic()
            elapsed = max(now - previous, 1e-6)
            previous = now
            measured = 1.0 / elapsed
            smoothed_fps = measured if not smoothed_fps else 0.9 * smoothed_fps + 0.1 * measured
            signature = tuple(sorted((d["name"], d["position"]) for d in detections if d["name"] in IMPORTANT_OBJECTS))
            announcements = []
            if signature and (signature != cloud.last_signature or now - cloud.last_announcement_at >= 7.0):
                announcements = guidance_messages(detections)
                cloud.last_signature = signature
                cloud.last_announcement_at = now
            elif not signature:
                cloud.last_signature = None
            annotated = draw_detections(frame.copy(), detections)
            cv2.putText(annotated, f"AKSHU 5.0 | LIVE | {smoothed_fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            try:
                cloud.update(detections, smoothed_fps, annotated, announcements)
            except requests.RequestException as exc:
                print(f"[CLOUD CONNECTION] {exc}", flush=True)
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping AKSHU agent…", flush=True)
    finally:
        if camera is not None:
            camera.release()
        cloud.disconnect()
        cv2.destroyAllWindows()
        print("Agent stopped.", flush=True)


if __name__ == "__main__":
    main()
