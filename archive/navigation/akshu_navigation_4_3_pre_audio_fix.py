
"""
AKSHU 4.3
Intelligent Obstacle Awareness and Priority-Based Voice Guidance

Uses existing project modules:
    core.realtime_vision
    core.voice

Experimental prototype. Not a certified mobility or safety device.
"""

import cv2
import time
from collections import Counter

from core.realtime_vision import RealtimeVisionEngine
from core.voice import AkshuVoice


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

NORMAL_ANNOUNCEMENT_COOLDOWN = 5.0
HAZARD_REPEAT_INTERVAL = 10.0

WINDOW_NAME = "AKSHU 4.3 - Intelligent Navigation"

CONFIDENCE_MINIMUM = 0.35

# Objects that may obstruct a walking path.
# Detection does NOT prove that an object is dangerous.
POTENTIAL_OBSTACLES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bench",
    "chair",
    "couch",
    "dining table",
    "potted plant",
    "suitcase",
    "backpack",
    "handbag",
    "bottle",
    "traffic light",
    "stop sign",
    "parking meter",
    "fire hydrant",
}


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    print("[CAMERA] Connecting...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        camera.release()

        print("[CAMERA] Trying default camera backend...")

        camera = cv2.VideoCapture(CAMERA_INDEX)

    if camera.isOpened():

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        print("[CAMERA] Connected.")

        return camera

    camera.release()

    return None


# ============================================================
# OBJECT INFORMATION
# ============================================================

def get_object_name(obj):

    if not isinstance(obj, dict):
        return "object"

    return str(
        obj.get("class_name", "object")
    ).lower().strip()


def get_object_confidence(obj):

    if not isinstance(obj, dict):
        return 0.0

    try:

        return float(
            obj.get("confidence", 1.0)
        )

    except (TypeError, ValueError):

        return 0.0


def get_object_box(obj):

    if not isinstance(obj, dict):
        return None

    box = obj.get("bbox")

    if box is None:
        box = obj.get("box")

    if box is None:
        box = obj.get("xyxy")

    if box is None:
        return None

    try:

        if isinstance(box, dict):

            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])

        else:

            x1, y1, x2, y2 = map(float, box)

        return x1, y1, x2, y2

    except (ValueError, TypeError, KeyError):

        return None


def get_object_zone(obj, frame_width):

    box = get_object_box(obj)

    if box is None:
        return "ahead"

    x1, _, x2, _ = box

    center_x = (x1 + x2) / 2

    if center_x < frame_width * 0.35:
        return "left"

    if center_x > frame_width * 0.65:
        return "right"

    return "ahead"


# ============================================================
# SCENE INTELLIGENCE
# ============================================================

def get_valid_objects(objects):

    valid = []

    for obj in objects:

        if not isinstance(obj, dict):
            continue

        confidence = get_object_confidence(obj)

        if confidence < CONFIDENCE_MINIMUM:
            continue

        name = get_object_name(obj)

        if name == "object":
            continue

        valid.append(obj)

    return valid


def create_scene_description(objects, frame_width):

    if not objects:
        return "I cannot identify any objects clearly."

    descriptions = []

    grouped = {}

    for obj in objects:

        name = get_object_name(obj)
        zone = get_object_zone(obj, frame_width)

        key = (name, zone)

        grouped[key] = grouped.get(key, 0) + 1

    for (name, zone), count in sorted(grouped.items()):

        if count == 1:

            descriptions.append(
                f"one {name} {zone}"
            )

        else:

            descriptions.append(
                f"{count} {name}s {zone}"
            )

    return "I can see " + ", ".join(descriptions) + "."


# ============================================================
# POTENTIAL OBSTACLE INTELLIGENCE
# ============================================================

def inspect_potential_obstacles(objects, frame_width, frame_height):

    candidates = []

    for obj in objects:

        name = get_object_name(obj)

        if name not in POTENTIAL_OBSTACLES:
            continue

        box = get_object_box(obj)

        if box is None:
            continue

        x1, y1, x2, y2 = box

        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)

        box_area = box_width * box_height

        frame_area = frame_width * frame_height

        if frame_area <= 0:
            continue

        area_ratio = box_area / frame_area

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        zone = get_object_zone(obj, frame_width)

        # Lower image region and bounding-box area are only
        # visual heuristics, NOT calibrated distance measurements.

        lower_region = center_y > frame_height * 0.65

        large_region = area_ratio > 0.035

        very_large_region = area_ratio > 0.10

        if lower_region and (large_region or very_large_region):

            candidates.append({
                "name": name,
                "zone": zone,
                "area": area_ratio,
                "confidence": get_object_confidence(obj)
            })

    return candidates


def create_obstacle_warning(candidates):

    if not candidates:
        return None

    # Prioritize the largest visible candidate.
    candidates.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    obstacle = candidates[0]

    name = obstacle["name"]
    zone = obstacle["zone"]

    if zone == "ahead":

        return (
            f"Potential obstruction detected ahead. "
            f"I can see a {name}. "
            f"Please check your path carefully."
        )

    if zone == "left":

        return (
            f"Potential obstruction visible on your left. "
            f"I can see a {name}. "
            f"Please check your surroundings."
        )

    return (
        f"Potential obstruction visible on your right. "
        f"I can see a {name}. "
        f"Please check your surroundings."
    )


# ============================================================
# SPEECH HELPERS
# ============================================================

def voice_is_muted(voice):

    value = getattr(voice, "is_muted", False)

    if callable(value):
        return bool(value())

    return bool(value)


def speak_message(voice, message):

    if not message:
        return

    print("[AKSHU VOICE]", message)

    voice.speak(message)


def handle_voice_command(command, voice, vision):

    if not command:
        return "continue"

    command = command.lower().strip()

    print("[COMMAND]", command)

    # Check unmute before mute.
    if "unmute" in command:

        voice.unmute()

        speak_message(
            voice,
            "Voice guidance enabled."
        )

        return "continue"

    if "mute" in command:

        voice.mute()

        print("[VOICE] Muted.")

        return "continue"

    if any(word in command for word in [
        "quit",
        "exit",
        "shutdown",
        "close akshu"
    ]):

        return "quit"

    if "pause" in command:

        return "pause"

    if "resume" in command:

        return "resume"

    if any(word in command for word in [
        "what do you see",
        "what is ahead",
        "describe the scene",
        "describe surroundings",
        "scene summary"
    ]):

        analysis = vision.get_latest_analysis()

        objects = analysis.get(
            "results", {}
        ).get("objects", [])

        valid = get_valid_objects(objects)

        message = create_scene_description(
            valid,
            FRAME_WIDTH
        )

        speak_message(voice, message)

        return "continue"

    if "how many people" in command or "count people" in command:

        analysis = vision.get_latest_analysis()

        objects = analysis.get(
            "results", {}
        ).get("objects", [])

        count = sum(
            1 for obj in get_valid_objects(objects)
            if get_object_name(obj) == "person"
        )

        speak_message(
            voice,
            f"I can see {count} people."
        )

        return "continue"

    if "obstacle" in command or "path" in command:

        analysis = vision.get_latest_analysis()

        objects = analysis.get(
            "results", {}
        ).get("objects", [])

        valid = get_valid_objects(objects)

        candidates = inspect_potential_obstacles(
            valid,
            FRAME_WIDTH,
            FRAME_HEIGHT
        )

        message = create_obstacle_warning(candidates)

        if message:

            speak_message(voice, message)

        else:

            speak_message(
                voice,
                "I do not currently detect a potential obstruction "
                "using my visual screening rules. "
                "This does not confirm that the path is clear."
            )

        return "continue"

    # Use the existing AKSHU question-answering system.
    try:

        response = vision.smart_vision.answer(command)

        if response:

            speak_message(voice, str(response))

        else:

            speak_message(
                voice,
                "I could not answer that question."
            )

    except Exception as exc:

        print("[COMMAND ERROR]", exc)

        speak_message(
            voice,
            "I could not process that command."
        )

    return "continue"


# ============================================================
# DISPLAY
# ============================================================

def draw_status(display, object_count, face_count, muted):

    height, width = display.shape[:2]

    voice_status = "MUTED" if muted else "VOICE ON"

    cv2.putText(
        display,
        f"AKSHU 4.3 | {voice_status}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"OBJECTS: {object_count} | FACES: {face_count}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "Q Quit | M Mute | S Summary | P Pause | V Voice",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 255),
        1
    )

    cv2.putText(
        display,
        "EXPERIMENTAL - NOT A SAFETY DEVICE",
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 165, 255),
        1
    )

    return display


# ============================================================
# MAIN AKSHU 4.3 ENGINE
# ============================================================

def main():

    camera = None
    voice = None
    vision = None

    paused = False

    last_scene_signature = None
    last_scene_announcement = 0.0

    last_hazard_signature = None
    last_hazard_announcement = 0.0

    frame_errors = 0

    print("=" * 60)
    print("AKSHU 4.3")
    print("INTELLIGENT OBSTACLE AWARENESS")
    print("PRIORITY-BASED VOICE GUIDANCE")
    print("=" * 60)

    print("[AKSHU] Starting systems...")

    try:

        # Initialize voice.
        voice = AkshuVoice()

        # Reuse existing vision architecture.
        vision = RealtimeVisionEngine(
            process_every_n_frames=5
        )

        print("[AKSHU] Initializing vision engine...")

        if not vision.initialize():

            raise RuntimeError(
                getattr(
                    vision,
                    "last_error",
                    "Vision initialization failed."
                )
            )

        # Initialize camera.
        camera = open_camera()

        if camera is None:

            raise RuntimeError(
                "Unable to open camera."
            )

        print("=" * 60)
        print("[AKSHU] ALL SYSTEMS READY")
        print("=" * 60)

        print("Q = Quit")
        print("M = Mute / Unmute")
        print("S = Scene summary")
        print("P = Pause / Resume")
        print("V = Voice command")

        speak_message(
            voice,
            "AKSHU 4.3 is ready. "
            "Intelligent visual guidance started."
        )

        # ----------------------------------------------------
        # CONTINUOUS LOOP
        # ----------------------------------------------------

        while True:

            # PAUSED MODE
            if paused:

                key = cv2.waitKey(100) & 0xFF

                if key in (ord("q"), ord("Q")):
                    break

                elif key in (ord("p"), ord("P")):

                    paused = False

                    speak_message(
                        voice,
                        "Vision resumed."
                    )

                elif key in (ord("m"), ord("M")):

                    if voice_is_muted(voice):

                        voice.unmute()

                        speak_message(
                            voice,
                            "Voice guidance enabled."
                        )

                    else:

                        voice.mute()

                        print("[VOICE] Muted.")

                continue

            # READ CAMERA
            success, frame = camera.read()

            if not success or frame is None:

                frame_errors += 1

                print(
                    f"[CAMERA] Frame error "
                    f"{frame_errors}/20"
                )

                if frame_errors >= 20:

                    raise RuntimeError(
                        "Camera stopped providing frames."
                    )

                time.sleep(0.1)

                continue

            frame_errors = 0

            height, width = frame.shape[:2]

            # EXISTING REALTIME VISION ENGINE
            display = vision.process_frame(frame)

            if display is None:
                display = frame.copy()

            analysis = vision.get_latest_analysis()

            results = analysis.get("results", {})

            objects = get_valid_objects(
                results.get("objects", [])
            )

            faces = results.get("faces", [])

            # Synchronize SmartVision with current results.
            try:

                vision.smart_vision.update(
                    objects=objects,
                    faces=faces
                )

            except Exception as exc:

                print("[VISION SYNC]", exc)

            now = time.monotonic()

            # ------------------------------------------------
            # PRIORITY 1: POTENTIAL OBSTACLE WARNING
            # ------------------------------------------------

            candidates = inspect_potential_obstacles(
                objects,
                width,
                height
            )

            warning = create_obstacle_warning(
                candidates
            )

            if warning:

                hazard_signature = tuple(
                    sorted(
                        (
                            item["name"],
                            item["zone"]
                        )
                        for item in candidates
                    )
                )

                changed = (
                    hazard_signature != last_hazard_signature
                )

                repeat_due = (
                    now - last_hazard_announcement
                    >= HAZARD_REPEAT_INTERVAL
                )

                if changed or repeat_due:

                    print("[PRIORITY WARNING]", warning)

                    speak_message(
                        voice,
                        warning
                    )

                    last_hazard_announcement = now

                last_hazard_signature = hazard_signature

            else:

                if last_hazard_signature is not None:

                    print(
                        "[OBSTACLE] Previous candidate "
                        "no longer visible."
                    )

                last_hazard_signature = None

            # ------------------------------------------------
            # PRIORITY 2: NORMAL SCENE ANNOUNCEMENT
            # ------------------------------------------------

            scene_signature = tuple(
                sorted(
                    (
                        get_object_name(obj),
                        get_object_zone(obj, width)
                    )
                    for obj in objects
                )
            )

            scene_changed = (
                scene_signature != last_scene_signature
            )

            cooldown_finished = (
                now - last_scene_announcement
                >= NORMAL_ANNOUNCEMENT_COOLDOWN
            )

            # Suppress ordinary descriptions while a potential
            # obstacle warning is being prioritized.
            if (
                scene_changed
                and cooldown_finished
                and not warning
            ):

                if objects:

                    message = create_scene_description(
                        objects,
                        width
                    )

                else:

                    message = (
                        "I cannot currently identify "
                        "any objects clearly."
                    )

                print("[SCENE]", message)

                speak_message(
                    voice,
                    message
                )

                last_scene_announcement = now

            last_scene_signature = scene_signature

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display = draw_status(
                display,
                len(objects),
                len(faces),
                voice_is_muted(voice)
            )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------
            # KEYBOARD CONTROLS
            # ------------------------------------------------

            if key in (ord("q"), ord("Q")):

                break

            elif key in (ord("m"), ord("M")):

                if voice_is_muted(voice):

                    voice.unmute()

                    speak_message(
                        voice,
                        "Voice guidance enabled."
                    )

                else:

                    voice.mute()

                    print("[VOICE] Muted.")

            elif key in (ord("s"), ord("S")):

                message = create_scene_description(
                    objects,
                    width
                )

                speak_message(
                    voice,
                    message
                )

            elif key in (ord("p"), ord("P")):

                paused = True

                print("[VISION] Paused.")

            elif key in (ord("v"), ord("V")):

                print("[AKSHU] Listening for command...")

                command = voice.listen()

                action = handle_voice_command(
                    command,
                    voice,
                    vision
                )

                if action == "quit":
                    break

                elif action == "pause":
                    paused = True

                elif action == "resume":

                    paused = False

                    speak_message(
                        voice,
                        "Vision resumed."
                    )

    except KeyboardInterrupt:

        print("\n[AKSHU] Interrupted by user.")

    except Exception as exc:

        print(
            "[AKSHU ERROR]",
            type(exc).__name__,
            str(exc)
        )

    finally:

        print("[AKSHU] Shutting down...")

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        if voice is not None:

            try:
                voice.close()

            except Exception as exc:
                print("[VOICE CLEANUP]", exc)

        print("[AKSHU] Shutdown complete.")


if __name__ == "__main__":
    main()
