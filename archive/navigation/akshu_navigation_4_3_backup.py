
"""
AKSHU 4.3 - Continuous Vision and Spoken Guidance

Experimental Windows webcam + YOLO + AKSHU voice prototype.

Controls:
    Q = Quit
    M = Mute / unmute
    S = Speak a scene summary

IMPORTANT:
This is an experimental prototype, not a certified mobility aid.
It can miss obstacles and cannot establish that a path is safe.
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

# Minimum interval between repeated announcements.
SCENE_COOLDOWN_SECONDS = 4.0
HAZARD_REPEAT_SECONDS = 6.0

WINDOW_NAME = "AKSHU 4.3 - Spoken Vision"

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

    print("[CAMERA] DirectShow failed; trying default backend.")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if camera.isOpened():
        return camera

    camera.release()
    return None


# ============================================================
# DETECTION
# ============================================================

def detect_objects(model, frame):
    """Return recognised objects with image positions."""

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
                print(f"[DETECTION WARNING] {exc}")

    return detections


# ============================================================
# SPOKEN GUIDANCE
# ============================================================

def speak_message(voice, message, priority=False):
    """
    Send messages to the real speech worker.

    Priority messages are logged distinctly. The voice queue itself
    handles speech; printing alone is not considered an announcement.
    """

    if not message:
        return

    message = " ".join(str(message).split()).strip()

    if not message:
        return

    print(
        "[AKSHU PRIORITY SPEECH]" if priority
        else "[AKSHU SPEECH REQUEST]",
        message,
        flush=True,
    )

    # This is the actual speech call, not just a terminal print.
    voice.speak(message)


def describe_scene(detections):
    """Create a concise description of the detected scene."""

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
            phrase = (
                f"{quantity} person {position}"
            )
        else:
            phrase = (
                f"{quantity} {name} {position}"
            )

        parts.append(phrase)

    return "I can see " + ", ".join(parts) + "."


def find_obstacle_warning(detections):
    """
    Provide cautious warnings for selected detections in the
    central part of the camera image.

    Image position is not a reliable measure of actual distance.
    """

    candidates = []

    for item in detections:
        name = item["name"]

        if name not in PRIORITY_OBJECTS:
            continue

        if item["position"] == "ahead":
            candidates.append(item)

    if not candidates:
        return None

    # Prioritise important classes, then larger image detections.
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
        f"AKSHU 4.3 | {voice_status} | Q quit | M mute | S summary",
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

    detections = []
    muted = False
    frame_errors = 0

    print("=" * 60)
    print("AKSHU 4.3 - CONTINUOUS SPOKEN VISION")
    print("=" * 60)
    print("Q: Quit | M: Mute/unmute | S: Speak scene summary")
    print("EXPERIMENTAL PROTOTYPE - NOT A CERTIFIED MOBILITY AID")
    print()

    try:
        # Start the dedicated voice worker.
        print("[AKSHU] Starting voice system...", flush=True)

        voice = AkshuVoice()

        # Allow the speech worker time to initialise.
        time.sleep(1.0)

        speak_message(
            voice,
            "AKSHU is starting. Voice guidance is being enabled.",
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
            raise RuntimeError(
                "Could not open the camera. Check Windows camera access."
            )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("[AKSHU] Camera opened.", flush=True)

        speak_message(
            voice,
            "Camera is active. AKSHU is now describing recognised objects.",
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
                    break

                time.sleep(0.1)
                continue

            frame_errors = 0
            now = time.monotonic()

            # Continuous object detection.
            detections = detect_objects(model, frame)

            # A signature changes when recognised objects or their
            # approximate image positions change.
            scene_signature = tuple(sorted(
                (item["name"], item["position"])
                for item in detections
            ))

            # Check for a potential obstacle ahead.
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

                repeat_due = (
                    now - last_hazard_announcement
                    >= HAZARD_REPEAT_SECONDS
                )

                if hazard_changed or repeat_due:
                    speak_message(
                        voice,
                        warning,
                        priority=True,
                    )
                    last_hazard_announcement = now

                last_hazard_signature = hazard_signature

            else:
                last_hazard_signature = None

            # Describe the scene when it changes, with a cooldown
            # to avoid repeatedly announcing every camera frame.
            scene_changed = scene_signature != last_scene_signature

            cooldown_finished = (
                now - last_scene_announcement
                >= SCENE_COOLDOWN_SECONDS
            )

            if (
                scene_changed
                and cooldown_finished
                and not warning
            ):
                message = describe_scene(detections)

                speak_message(voice, message)

                last_scene_announcement = now

            last_scene_signature = scene_signature

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
                        "Voice guidance enabled.",
                    )
                else:
                    voice.mute()
                    muted = True
                    print("[AKSHU] Voice muted.", flush=True)

            elif key in (ord("s"), ord("S")):
                voice.unmute()
                muted = False
                print("[AKSHU TEST] Sending a direct voice announcement.", flush=True)
                test_message = (
                    "AKSHU voice test. "
                    "I can speak. "
                    + describe_scene(detections)
                )
                voice.speak(test_message)

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
            time.sleep(2)

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        if voice is not None:
            voice.close()

        print("[AKSHU] Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
