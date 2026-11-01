"""
AKSHU 5.0 - Continuous Vision and Spoken Guidance
Experimental Windows webcam + YOLO + AKSHU voice prototype.

Controls:
    Q = Quit
    M = Mute / unmute
    S = Speak a scene summary

IMPORTANT:
This is an experimental prototype, not a certified mobility aid.
It can miss obstacles, cannot reliably estimate real-world distance,
and cannot establish that a walking path is safe.
"""

import time
from collections import Counter

import cv2
from ultralytics import YOLO

from core.voice import AkshuVoice


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0
MODEL_NAME = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.45
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INFERENCE_SIZE = 640

# Speak a changed scene after this minimum gap.
SCENE_CHANGE_COOLDOWN_SECONDS = 4.0

# Also speak a fresh scene summary at this interval,
# even when the camera view has not changed.
SCENE_REPEAT_SECONDS = 10.0

# Repeat a continuing potential-hazard warning at this interval.
HAZARD_REPEAT_SECONDS = 6.0

# Avoid repeatedly saying that no objects were recognised.
EMPTY_SCENE_REPEAT_SECONDS = 12.0

WINDOW_NAME = "AKSHU 5.0 - Spoken Vision"

PRIORITY_OBJECTS = {
    "person",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "suitcase",
    "backpack",
}

IMPORTANT_OBJECTS = {
    "person",
    "car",
    "motorcycle",
    "bicycle",
    "bus",
    "truck",
    "dog",
}


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    """Open the Windows camera using DirectShow first."""

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if camera.isOpened():
        return camera

    camera.release()
    print("[CAMERA] DirectShow failed; trying default backend.", flush=True)

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if camera.isOpened():
        return camera

    camera.release()
    return None


# ============================================================
# DETECTION
# ============================================================

def detect_objects(model, frame):
    """Return recognised objects with approximate image positions."""

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_SIZE,
        verbose=False,
    )

    height, width = frame.shape[:2]
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            try:
                class_id = int(box.cls[0].item())
                name = str(model.names[class_id]).lower()
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                center_x = (x1 + x2) / 2

                if center_x < width * 0.38:
                    position = "on the left"
                elif center_x > width * 0.62:
                    position = "on the right"
                else:
                    position = "ahead"

                detections.append({
                    "name": name,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                    "position": position,
                    "center_x": center_x,
                    "center_y": (y1 + y2) / 2,
                    "frame_width": width,
                    "frame_height": height,
                })

            except (ValueError, IndexError, RuntimeError) as exc:
                print(f"[DETECTION WARNING] {exc}", flush=True)

    return detections


# ============================================================
# SPOKEN GUIDANCE
# ============================================================

def speak_message(voice, message, priority=False):
    """Send a message to the existing AKSHU speech worker."""

    if not message:
        return

    message = " ".join(str(message).split()).strip()
    if not message:
        return

    label = (
        "[AKSHU PRIORITY SPEECH]"
        if priority
        else "[AKSHU SPEECH REQUEST]"
    )
    print(label, message, flush=True)

    # This calls the actual speech engine through AkshuVoice.
    voice.speak(message)


def build_scene_signature(detections):
    """Build a stable, order-independent description of the current view."""

    return tuple(sorted(
        (item["name"], item["position"])
        for item in detections
    ))


def describe_scene(detections):
    """Create a concise spoken description of recognised objects."""

    if not detections:
        return "I cannot currently identify any objects clearly."

    counts = Counter(
        (item["name"], item["position"])
        for item in detections
    )

    parts = []

    for (name, position), count in counts.most_common(4):
        quantity = "one" if count == 1 else str(count)

        if name == "person":
            phrase = f"{quantity} person {position}"
        else:
            phrase = f"{quantity} {name} {position}"

        parts.append(phrase)

    return "I can see " + ", ".join(parts) + "."


def find_obstacle_warning(detections):
    """
    Give cautious warnings for selected detections near the centre
    of the camera image. Image position is not a distance measurement.
    """

    candidates = []

    for item in detections:
        if item["name"] not in PRIORITY_OBJECTS:
            continue

        if item["position"] == "ahead":
            candidates.append(item)

    if not candidates:
        return None

    # Prioritise selected important classes, then larger image boxes.
    candidates.sort(
        key=lambda item: (
            item["name"] not in IMPORTANT_OBJECTS,
            -(
                (item["box"][2] - item["box"][0])
                * (item["box"][3] - item["box"][1])
            ),
        )
    )

    item = candidates[0]
    name = item["name"]

    if name == "person":
        return (
            "Person detected ahead. "
            "Please check your path carefully."
        )

    if name in {"car", "motorcycle", "bus", "truck", "bicycle"}:
        return (
            f"Possible {name} detected ahead. "
            "Stay alert and check your surroundings."
        )

    return (
        f"Possible obstacle, {name}, detected ahead. "
        "Please check your path carefully."
    )


# ============================================================
# CAMERA DISPLAY
# ============================================================

def draw_detections(frame, detections, muted):
    """Draw object boxes and the current voice status."""

    for item in detections:
        x1, y1, x2, y2 = item["box"]

        label = (
            f'{item["name"]} '
            f'{item["confidence"]:.2f} '
            f'| {item["position"]}'
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
        )

    voice_status = "MUTED" if muted else "VOICE ON"

    cv2.putText(
        frame,
        f"AKSHU 5.0 | {voice_status} | Q quit | M mute | S summary",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "EXPERIMENTAL - NOT A SAFETY DEVICE",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 165, 255),
        2,
    )

    return frame


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    camera = None
    voice = None

    last_scene_signature = None
    last_scene_announcement = 0.0

    last_hazard_signature = None
    last_hazard_announcement = 0.0

    last_empty_scene_announcement = 0.0

    detections = []
    muted = False
    frame_errors = 0

    print("=" * 64)
    print("AKSHU 5.0 - CONTINUOUS SPOKEN VISION")
    print("=" * 64)
    print("Q: Quit | M: Mute/unmute | S: Speak scene summary")
    print("Vision runs continuously; narration is periodic and event-driven.")
    print("EXPERIMENTAL PROTOTYPE - NOT A CERTIFIED MOBILITY AID")
    print()

    try:
        print("[AKSHU] Starting voice system...", flush=True)
        voice = AkshuVoice()

        # Give the speech worker time to initialise.
        time.sleep(1.5)

        speak_message(
            voice,
            "AKSHU five point zero is starting. Voice guidance is enabled.",
        )

        print("[AKSHU] Loading YOLO model...", flush=True)
        model = YOLO(MODEL_NAME)
        print("[AKSHU] YOLO model loaded.", flush=True)

        camera = open_camera()

        if camera is None:
            speak_message(
                voice,
                "Camera unavailable. AKSHU vision cannot start. "
                "Please use your usual mobility aid.",
                priority=True,
            )
            time.sleep(2.0)
            raise RuntimeError(
                "Could not open the camera. Check Windows camera access."
            )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("[AKSHU] Camera opened.", flush=True)

        speak_message(
            voice,
            "Camera is active. I will continuously analyse the view "
            "and provide regular spoken updates.",
        )

        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                frame_errors += 1
                print(
                    f"[CAMERA WARNING] Frame read failed "
                    f"({frame_errors}/20).",
                    flush=True,
                )

                if frame_errors >= 20:
                    speak_message(
                        voice,
                        "Camera feed lost. AKSHU vision is stopping.",
                        priority=True,
                    )
                    time.sleep(1.5)
                    break

                time.sleep(0.1)
                continue

            frame_errors = 0
            now = time.monotonic()

            # Vision inference continues on every successful frame.
            detections = detect_objects(model, frame)
            scene_signature = build_scene_signature(detections)

            # ----------------------------------------------------
            # 1. Potential-hazard announcements
            # ----------------------------------------------------

            warning = find_obstacle_warning(detections)

            if warning:
                hazard_signature = tuple(sorted(
                    (item["name"], item["position"])
                    for item in detections
                    if (
                        item["name"] in PRIORITY_OBJECTS
                        and item["position"] == "ahead"
                    )
                ))

                hazard_changed = (
                    hazard_signature != last_hazard_signature
                )
                hazard_repeat_due = (
                    now - last_hazard_announcement
                    >= HAZARD_REPEAT_SECONDS
                )

                if hazard_changed or hazard_repeat_due:
                    speak_message(voice, warning, priority=True)
                    last_hazard_announcement = now

                last_hazard_signature = hazard_signature

            else:
                last_hazard_signature = None

            # ----------------------------------------------------
            # 2. Periodic and change-triggered scene narration
            # ----------------------------------------------------

            scene_changed = scene_signature != last_scene_signature

            change_cooldown_finished = (
                now - last_scene_announcement
                >= SCENE_CHANGE_COOLDOWN_SECONDS
            )

            scene_repeat_due = (
                now - last_scene_announcement
                >= SCENE_REPEAT_SECONDS
            )

            # Do not put ordinary scene narration ahead of a current
            # potential-hazard warning.
            if not warning and detections:
                if (
                    (scene_changed and change_cooldown_finished)
                    or scene_repeat_due
                ):
                    speak_message(
                        voice,
                        describe_scene(detections),
                    )
                    last_scene_announcement = now

            elif not warning and not detections:
                if (
                    now - last_empty_scene_announcement
                    >= EMPTY_SCENE_REPEAT_SECONDS
                ):
                    speak_message(
                        voice,
                        "I cannot currently identify any objects clearly.",
                    )
                    last_empty_scene_announcement = now
                    last_scene_announcement = now

            last_scene_signature = scene_signature

            # ----------------------------------------------------
            # 3. Display and keyboard controls
            # ----------------------------------------------------

            display = draw_detections(
                frame.copy(),
                detections,
                muted,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print("[AKSHU] Shutdown requested.", flush=True)
                break

            elif key in (ord("m"), ord("M")):
                if muted:
                    voice.unmute()
                    muted = False
                    speak_message(
                        voice,
                        "Voice guidance enabled. " + describe_scene(detections),
                    )
                else:
                    voice.mute()
                    muted = True
                    print("[AKSHU] Voice muted.", flush=True)

            elif key in (ord("s"), ord("S")):
                voice.unmute()
                muted = False
                print(
                    "[AKSHU TEST] Sending a direct voice announcement.",
                    flush=True,
                )
                speak_message(
                    voice,
                    "This is AKSHU manual speech test. Please confirm you can hear me. "
                    + describe_scene(detections),
                )

    except KeyboardInterrupt:
        print("\n[AKSHU] Stopped by user.", flush=True)

    except Exception as exc:
        print(f"[AKSHU ERROR] {exc}", flush=True)

        if voice is not None:
            speak_message(
                voice,
                "AKSHU encountered an error and is stopping.",
                priority=True,
            )
            time.sleep(2.0)

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        if voice is not None:
            voice.close()

        print("[AKSHU] Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
