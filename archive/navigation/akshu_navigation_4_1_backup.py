"""
AKSHU 4.2 - Intelligent Continuous Vision
Experimental object-detection and voice-awareness prototype.

Controls (click the camera window first):
  Q = Quit
  M = Mute / unmute
  S = Speak current scene summary
  P = Pause / resume vision

SAFETY:
This is NOT a certified mobility aid. It can miss objects and misclassify
scenes. Do not use it as a substitute for a cane, guide dog, or human support.
It does not reliably detect stairs, drop-offs, moving-vehicle trajectories,
or safe crossing opportunities. Do not rely on it for navigation.
"""

import queue
import threading
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# ----------------------------- SETTINGS ---------------------------------

CAMERA_INDEX = 0
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
INFERENCE_SIZE = 640
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Minimum gap between automatic announcements.
ANNOUNCEMENT_COOLDOWN = 3.5
MAX_SPEECH_QUEUE = 4
MAX_OBJECTS_IN_ANNOUNCEMENT = 4

WINDOW_NAME = "AKSHU 4.2 - Intelligent Continuous Vision"

# These are prioritized for spoken descriptions, not guaranteed danger labels.
PRIORITY_OBJECTS = {
    "car", "motorcycle", "bicycle", "bus", "truck",
    "person", "dog", "traffic light", "stop sign"
}


# ----------------------------- SHARED STATE ----------------------------

speech_queue = queue.Queue(maxsize=MAX_SPEECH_QUEUE)
stop_event = threading.Event()
voice_muted = threading.Event()
voice_ready = threading.Event()


# ----------------------------- VOICE ------------------------------------

def enqueue_speech(message):
    """Queue speech without blocking camera processing."""
    if not message or voice_muted.is_set() or stop_event.is_set():
        return

    try:
        speech_queue.put_nowait(str(message))
    except queue.Full:
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            pass
        try:
            speech_queue.put_nowait(str(message))
        except queue.Full:
            print("[VOICE] Queue full; announcement skipped.")


def speech_worker():
    """Initialize and operate TTS on one dedicated thread."""
    engine = None
    com_initialized = False

    try:
        try:
            import pythoncom
            pythoncom.CoInitialize()
            com_initialized = True
        except ImportError:
            pass

        if pyttsx3 is None:
            print("[VOICE] pyttsx3 unavailable; speech disabled.")
            voice_ready.set()
            return

        try:
            engine = pyttsx3.init(driverName="sapi5")
        except Exception:
            engine = pyttsx3.init()

        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        voice_ready.set()
        print("[VOICE] Voice engine initialized.")

        while True:
            try:
                message = speech_queue.get(timeout=0.25)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            try:
                if message is None:
                    break
                if not voice_muted.is_set():
                    print(f"[VOICE] Speaking: {message}")
                    engine.say(message)
                    engine.runAndWait()
            except Exception as exc:
                print(f"[VOICE ERROR] {exc}")
            finally:
                speech_queue.task_done()

    except Exception as exc:
        print(f"[VOICE ERROR] Initialization failed: {exc}")
        voice_ready.set()
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
        print("[VOICE] Speech worker stopped.")


# ----------------------------- CAMERA -----------------------------------

def open_camera(index):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)

    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return camera

    camera.release()
    return None


# ----------------------------- DETECTION --------------------------------

def collect_detections(results, model, frame_width):
    detections = []
    if not results:
        return detections

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return detections

    names = model.names

    for box in boxes:
        try:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            center_x = (x1 + x2) / 2
            if center_x < frame_width * 0.35:
                position = "left"
            elif center_x > frame_width * 0.65:
                position = "right"
            else:
                position = "ahead"

            detections.append({
                "label": str(label),
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "position": position,
            })
        except Exception as exc:
            print(f"[DETECTION WARNING] {exc}")

    return detections


def ordered_detections(detections):
    """Prioritize selected classes, then higher-confidence detections."""
    return sorted(
        detections,
        key=lambda item: (
            item["label"] not in PRIORITY_OBJECTS,
            -item["confidence"]
        )
    )


def scene_summary(detections):
    """Describe detected objects and their approximate image positions."""
    if not detections:
        return "No recognized objects are visible."

    grouped = defaultdict(int)
    for item in ordered_detections(detections):
        grouped[(item["label"], item["position"])] += 1

    parts = []
    for (label, position), count in list(grouped.items())[:MAX_OBJECTS_IN_ANNOUNCEMENT]:
        quantity = "one" if count == 1 else str(count)
        parts.append(f"{quantity} {label} {position}")

    if not parts:
        return "No recognized objects are visible."

    return "I can see " + ", ".join(parts) + "."


def automatic_announcement(detections, now, state):
    """
    Speak when the visible object/position pattern changes.
    Repeated frames with the same pattern do not generate repeated speech.
    """
    current_signature = tuple(sorted(
        (item["label"], item["position"])
        for item in detections
        if item["confidence"] >= CONFIDENCE_THRESHOLD
    ))

    if current_signature == state["last_signature"]:
        return

    if now - state["last_announcement"] < ANNOUNCEMENT_COOLDOWN:
        return

    if current_signature:
        message = scene_summary(detections)
        enqueue_speech(message)
        print(f"[SCENE] {message}")
    elif state["last_signature"]:
        message = "The previously recognized objects are no longer visible."
        enqueue_speech(message)
        print(f"[SCENE] {message}")

    state["last_signature"] = current_signature
    state["last_announcement"] = now


# ----------------------------- DISPLAY ----------------------------------

def draw_detections(frame, detections):
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        confidence = item["confidence"]
        position = item["position"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        caption = f"{label} {confidence:.2f} | {position}"
        cv2.putText(
            frame, caption, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2
        )
    return frame


# ----------------------------- MAIN --------------------------------------

def main():
    camera = None
    worker = threading.Thread(
        target=speech_worker, name="AKSHU-Speech", daemon=True
    )
    worker.start()

    print("=" * 58)
    print("AKSHU 4.2 - INTELLIGENT CONTINUOUS VISION")
    print("Q: quit | M: mute/unmute | S: scene summary | P: pause")
    print("EXPERIMENTAL ONLY - NOT A SAFETY DEVICE")
    print("=" * 58)

    try:
        print(f"[INFO] Loading model: {MODEL_NAME}")
        model = YOLO(MODEL_NAME)
        print("[INFO] YOLO model loaded.")

        camera = open_camera(CAMERA_INDEX)
        if camera is None:
            raise RuntimeError(
                f"Could not open camera {CAMERA_INDEX}. "
                "Check camera permissions or change CAMERA_INDEX."
            )

        print("[INFO] Camera opened.")
        enqueue_speech("AKSHU is ready. Intelligent vision started.")

        paused = False
        frame_errors = 0
        last_frame = None
        detections = []
        announcement_state = {
            "last_signature": None,
            "last_announcement": 0.0,
        }

        while not stop_event.is_set():
            if paused:
                if last_frame is not None:
                    paused_display = last_frame.copy()
                    cv2.putText(
                        paused_display, "PAUSED - Press P to resume",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 165, 255), 2
                    )
                    cv2.imshow(WINDOW_NAME, paused_display)

                key = cv2.waitKey(100) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                elif key in (ord("p"), ord("P")):
                    paused = False
                    print("[VISION] Resumed.")
                    enqueue_speech("Vision resumed.")
                elif key in (ord("m"), ord("M")):
                    if voice_muted.is_set():
                        voice_muted.clear()
                        print("[VOICE] Unmuted.")
                        enqueue_speech("Voice guidance enabled.")
                    else:
                        voice_muted.set()
                        print("[VOICE] Muted.")
                elif key in (ord("s"), ord("S")):
                    summary = scene_summary(detections)
                    print("[KEYBOARD] S pressed.")
                    print("[SUMMARY]", summary)
                    enqueue_speech(summary)
                continue

            ok, frame = camera.read()
            if not ok or frame is None:
                frame_errors += 1
                print(f"[CAMERA WARNING] Frame read failed ({frame_errors}/20).")
                if frame_errors >= 20:
                    raise RuntimeError("Camera stopped providing frames.")
                time.sleep(0.1)
                continue

            frame_errors = 0
            last_frame = frame.copy()

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=INFERENCE_SIZE,
                verbose=False
            )

            height, width = frame.shape[:2]
            detections = collect_detections(results, model, width)
            automatic_announcement(detections, time.time(), announcement_state)

            display = draw_detections(frame.copy(), detections)
            status = "MUTED" if voice_muted.is_set() else "VOICE ON"

            cv2.putText(
                display,
                f"AKSHU 4.2 | {status} | Q quit / M mute / S summary / P pause",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2
            )
            cv2.putText(
                display, "EXPERIMENTAL - NOT A SAFETY DEVICE",
                (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 165, 255), 2
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("m"), ord("M")):
                if voice_muted.is_set():
                    voice_muted.clear()
                    print("[VOICE] Unmuted.")
                    enqueue_speech("Voice guidance enabled.")
                else:
                    voice_muted.set()
                    print("[VOICE] Muted.")
            elif key in (ord("s"), ord("S")):
                summary = scene_summary(detections)
                print("[KEYBOARD] S pressed.")
                print("[SUMMARY]", summary)
                enqueue_speech(summary)
            elif key in (ord("p"), ord("P")):
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

        # Ensure the speech worker receives a shutdown signal.
        try:
            speech_queue.put_nowait(None)
        except queue.Full:
            try:
                speech_queue.get_nowait()
                speech_queue.task_done()
            except queue.Empty:
                pass
            try:
                speech_queue.put_nowait(None)
            except queue.Full:
                pass

        worker.join(timeout=4)
        print("[INFO] AKSHU stopped.")


if __name__ == "__main__":
    main()
