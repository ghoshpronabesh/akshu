"""
AKSHU 4.1 - Continuous Smart Vision (experimental)
Windows + webcam + YOLOv8 + pyttsx3 speech.

Controls:
  Q = quit
  M = mute/unmute voice
  S = speak a scene summary now
  P = pause/resume vision

SAFETY:
This is an experimental visual-assistance prototype, NOT a certified mobility aid.
It cannot reliably detect drop-offs, stairs, traffic gaps, collision risk, or safe
crossing conditions. Never use it as the only source of guidance while walking.
Distance is intentionally not estimated or spoken because monocular estimates
from ordinary webcam images are not dependable safety measurements.
"""

import cv2
import queue
import threading
import time
from collections import Counter, defaultdict

import pyttsx3
from ultralytics import YOLO
# ---------------------------- SETTINGS ---------------------------------

CAMERA_INDEX = 0
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INFERENCE_SIZE = 640

# Speak a changed scene no more often than this.
SUMMARY_INTERVAL_SECONDS = 5.0
SAME_SCENE_COOLDOWN_SECONDS = 12.0
MAX_SPOKEN_OBJECTS = 4

# Keep voice output short. Priority objects are mentioned first.
PRIORITY_OBJECTS = {
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "traffic light", "stop sign", "dog"
}

speech_queue = queue.Queue(maxsize=3)
stop_event = threading.Event()
voice_ready = threading.Event()
voice_muted = threading.Event()

# COCO names are supplied by the loaded YOLO model.
last_scene_signature = None
last_scene_spoken_at = 0.0
# ---------------------------- VOICE ------------------------------------

def enqueue_speech(message):
    """Queue speech without allowing an old backlog to build up."""
    if voice_muted.is_set() or stop_event.is_set():
        return

    message = " ".join(str(message).split()).strip()
    if not message:
        return

    # If the queue is full, discard the oldest pending message.
    try:
        speech_queue.put_nowait(message)
    except queue.Full:
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            pass
        try:
            speech_queue.put_nowait(message)
        except queue.Full:
            print("[VOICE] Queue busy; message skipped.")

    print(f"[VOICE QUEUED] {message}")


def speech_worker():
    """Own the TTS engine on one thread; initialize Windows COM here."""
    engine = None
    com_initialized = False

    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_initialized = True
            print("[VOICE] Windows COM initialized.")
        except ImportError:
            print("[VOICE] pywin32/pythoncom unavailable; trying normal TTS.")

        # Explicit SAPI5 backend is generally the appropriate Windows backend.
        try:
            engine = pyttsx3.init(driverName="sapi5")
        except Exception:
            print("[VOICE] SAPI5 initialization failed; trying default driver.")
            engine = pyttsx3.init()

        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        voices = engine.getProperty("voices")
        if voices:
            # Keep the Windows default voice; do not assume a particular voice ID.
            print(f"[VOICE] Available voices: {len(voices)}")

        print("[VOICE] Engine initialized.")
        engine.say("AKSHU voice system is ready.")
        engine.runAndWait()
        voice_ready.set()
        print("[VOICE] Startup phrase completed.")

        while not stop_event.is_set():
            try:
                message = speech_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                if message and not voice_muted.is_set():
                    print(f"[VOICE] Speaking: {message}")
                    engine.say(message)
                    engine.runAndWait()
                    print("[VOICE] Finished.")
            except Exception as exc:
                print(f"[VOICE] Speech error: {exc}")
                # Reinitialize on the same worker thread if the driver failed.
                try:
                    engine.stop()
                except Exception:
                    pass
            finally:
                speech_queue.task_done()

    except Exception as exc:
        print(f"[VOICE] Worker failed: {exc}")
        print("[VOICE] Try: python -m pip install --upgrade pyttsx3 pywin32")
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        if com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
            # ---------------------------- CAMERA -----------------------------------

def open_camera():
    """Try Windows DirectShow first, then OpenCV's default backend."""
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if camera.isOpened():
        return camera

    camera.release()
    print("[CAMERA] DirectShow failed; trying default backend.")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if camera.isOpened():
        return camera

    camera.release()
    return None


# ------------------------ SCENE UNDERSTANDING --------------------------

def object_position(center_x, frame_width):
    if center_x < frame_width * 0.38:
        return "left"
    if center_x > frame_width * 0.62:
        return "right"
    return "ahead"


def collect_detections(results, model, frame_width):
    """Convert YOLO boxes into a compact list of object observations."""
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0].item())
            class_id = int(box.cls[0].item())
            name = str(model.names.get(class_id, class_id)).lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x = (x1 + x2) // 2
            position = object_position(center_x, frame_width)

            detections.append({
                "name": name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "position": position,
            })

    return detections


def scene_summary(detections):
    """Build a concise spoken summary; avoid narrating every detected box."""
    if not detections:
        return "I cannot identify a clear object in the current view."

    grouped = defaultdict(Counter)
    for item in detections:
        grouped[item["name"]][item["position"]] += 1

    names = sorted(
        grouped.keys(),
        key=lambda name: (
            name not in PRIORITY_OBJECTS,
            -sum(grouped[name].values()),
            name
        )
    )

    phrases = []
    for name in names[:MAX_SPOKEN_OBJECTS]:
        positions = grouped[name]
        total = sum(positions.values())

        if total > 1:
            noun = f"{total} {name}s" if not name.endswith("s") else f"{total} {name}"
        else:
            noun = "a person" if name == "person" else f"a {name}"

        if len(positions) == 1:
            position = next(iter(positions))
            phrases.append(f"{noun} {position}")
        else:
            position_text = ", ".join(
                f"{count} {pos}" for pos, count in positions.items()
            )
            phrases.append(f"{noun}, {position_text}")

    if not phrases:
        return "Objects detected."
    return "I can see " + "; ".join(phrases) + "."


def maybe_announce_scene(detections, now):
    """Speak when the scene changes, with a cooldown to prevent chatter."""
    global last_scene_signature, last_scene_spoken_at

    signature = tuple(sorted(
        (item["name"], item["position"]) for item in detections
    ))

    changed = signature != last_scene_signature
    interval_elapsed = now - last_scene_spoken_at >= SUMMARY_INTERVAL_SECONDS
    cooldown_elapsed = now - last_scene_spoken_at >= SAME_SCENE_COOLDOWN_SECONDS

    if detections and (changed and interval_elapsed or cooldown_elapsed):
        enqueue_speech(scene_summary(detections))
        last_scene_signature = signature
        last_scene_spoken_at = now
    elif not detections and changed and interval_elapsed:
        # Do not repeatedly announce an empty view.
        last_scene_signature = signature


def draw_detections(frame, detections):
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        label = f'{item["name"]} {item["confidence"]:.2f} | {item["position"]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 2
        )
    return frame
# ------------------------------- MAIN ----------------------------------

def main():
    camera = None
    paused = False

    worker = threading.Thread(
        target=speech_worker,
        name="AKSHU-Voice",
        daemon=True
    )
    worker.start()

    print("=" * 60)
    print("AKSHU 4.1 - CONTINUOUS SMART VISION")
    print("=" * 60)
    print("Q: quit | M: mute/unmute | S: speak summary | P: pause/resume")
    print("Safety: experimental prototype, not a certified mobility aid.")

    try:
        print("[VISION] Loading YOLO model...")
        model = YOLO(MODEL_NAME)
        print("[VISION] Model loaded.")

        camera = open_camera()
        if camera is None:
            print("[CAMERA] ERROR: Could not open camera.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print("[CAMERA] Camera opened.")

        if voice_ready.wait(timeout=8):
            enqueue_speech("Camera is active. AKSHU is beginning scene observation.")
        else:
            print("[VOICE] Startup voice did not become ready within 8 seconds.")

        frame_errors = 0

        while not stop_event.is_set():
            if paused:
                frame = None
                key = cv2.waitKey(100) & 0xFF
                if key == ord("p"):
                    paused = False
                    enqueue_speech("Vision resumed.")
                elif key == ord("q"):
                    break
                elif key == ord("m"):
                    if voice_muted.is_set():
                        voice_muted.clear()
                        enqueue_speech("Voice guidance enabled.")
                    else:
                        voice_muted.set()
                        print("[VOICE] Muted.")
                continue

            ok, frame = camera.read()
            if not ok or frame is None:
                frame_errors += 1
                print(f"[CAMERA] Frame unavailable ({frame_errors}/10).")
                if frame_errors >= 10:
                    enqueue_speech("Camera feed lost. Please stop and check the camera.")
                    break
                time.sleep(0.15)
                continue

            frame_errors = 0
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=INFERENCE_SIZE,
                verbose=False
            )

            height, width = frame.shape[:2]
            detections = collect_detections(results, model, width)
            now = time.time()

            maybe_announce_scene(detections, now)
            display = draw_detections(frame.copy(), detections)

            status = "MUTED" if voice_muted.is_set() else "VOICE ON"
            cv2.putText(
                display, f"AKSHU 4.1 | {status} | Q quit / M mute / S summary / P pause",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2
            )
            cv2.putText(
                display, "EXPERIMENTAL - NOT A SAFETY DEVICE",
                (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 165, 255), 2
            )
            cv2.imshow("AKSHU 4.1 - Continuous Smart Vision", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                if voice_muted.is_set():
                    voice_muted.clear()
                    enqueue_speech("Voice guidance enabled.")
                else:
                    voice_muted.set()
                    print("[VOICE] Muted.")
            elif key == ord("s"):
                enqueue_speech(scene_summary(detections))
            elif key == ord("p"):
                paused = True
                print("[VISION] Paused. Press P to resume.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as exc:
        print(f"[ERROR] AKSHU encountered an error: {exc}")
    finally:
        stop_event.set()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

        # Allow the worker to finish its current phrase and exit.
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            pass
