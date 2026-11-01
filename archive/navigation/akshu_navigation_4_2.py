
import time
import cv2

from core.realtime_vision import RealtimeVisionEngine
from core.voice import AkshuVoice


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ANNOUNCEMENT_COOLDOWN = 3.5

WINDOW_NAME = "AKSHU 4.2 - Continuous Vision"


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


def process_voice_command(command, voice, vision):
    command = command.lower().strip()

    if not command:
        return None

    if command in ("quit", "exit", "close akshu"):
        return "quit"

    if command in ("pause", "pause vision"):
        return "pause"

    if command in ("resume", "resume vision"):
        return "resume"

    if command in ("mute", "mute voice"):
        voice.mute()
        print("[VOICE] Muted.")
        return None

    if command in ("unmute", "unmute voice"):
        voice.unmute()
        voice.speak("Voice guidance enabled.")
        return None

    response = vision.smart_vision.answer(command)
    voice.speak(response)
    return None


def main():
    camera = None
    voice = AkshuVoice()
    vision = RealtimeVisionEngine(process_every_n_frames=5)

    paused = False
    frame_errors = 0
    last_signature = None
    last_announcement = 0.0

    print("=" * 55)
    print("AKSHU 4.2 - INTELLIGENT CONTINUOUS VISION")
    print("=" * 55)
    print("Q = Quit | M = Mute | S = Scene summary")
    print("P = Pause/Resume | V = Voice command")
    print("EXPERIMENTAL PROTOTYPE - NOT A SAFETY DEVICE")
    print("=" * 55)

    try:
        print("[AKSHU] Initializing vision engine...")

        if not vision.initialize():
            raise RuntimeError(
                vision.last_error or "Vision initialization failed."
            )

        camera = open_camera(CAMERA_INDEX)

        if camera is None:
            raise RuntimeError(
                "Camera unavailable. Check camera permissions."
            )

        print("[AKSHU] Camera connected.")
        voice.speak("AKSHU is ready. Continuous vision started.")

        while True:

            if paused:
                key = cv2.waitKey(100) & 0xFF

                if key in (ord("q"), ord("Q")):
                    break

                elif key in (ord("p"), ord("P")):
                    paused = False
                    voice.speak("Vision resumed.")
                    print("[VISION] Resumed.")

                elif key in (ord("m"), ord("M")):
                    if voice.is_muted:
                        voice.unmute()
                        voice.speak("Voice guidance enabled.")
                    else:
                        voice.mute()

                continue

            ok, frame = camera.read()

            if not ok or frame is None:
                frame_errors += 1
                print(f"[CAMERA] Frame failure {frame_errors}/20")

                if frame_errors >= 20:
                    raise RuntimeError("Camera stopped providing frames.")

                time.sleep(0.1)
                continue

            frame_errors = 0
            height, width = frame.shape[:2]

            # Use the existing AKSHU vision engine.
            display = vision.process_frame(frame)

            analysis = vision.get_latest_analysis()
            results = analysis.get("results", {})

            objects = results.get("objects", [])
            faces = results.get("faces", [])

            vision.smart_vision.update(
                objects=objects,
                faces=faces
            )

            smart = vision.smart_vision

            # Announce changes in the visible scene.
            signature = smart.scene_signature(width)
            now = time.monotonic()

            if signature != last_signature:
                if now - last_announcement >= ANNOUNCEMENT_COOLDOWN:

                    if signature:
                        message = smart.navigation_summary(width)
                        print("[SCENE]", message)
                        voice.speak(message)

                    elif last_signature:
                        message = (
                            "Previously recognized objects "
                            "are no longer visible."
                        )
                        print("[SCENE]", message)
                        voice.speak(message)

                    last_signature = signature
                    last_announcement = now

            status = "MUTED" if voice.is_muted else "VOICE ON"

            cv2.putText(
                display,
                f"AKSHU 4.2 | {status}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"OBJECTS: {len(objects)} | FACES: {len(faces)}",
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
                0.45,
                (0, 255, 255),
                1
            )

            cv2.putText(
                display,
                "EXPERIMENTAL - NOT A SAFETY DEVICE",
                (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 165, 255),
                2
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            elif key in (ord("m"), ord("M")):
                if voice.is_muted:
                    voice.unmute()
                    voice.speak("Voice guidance enabled.")
                    print("[VOICE] Unmuted.")
                else:
                    voice.mute()
                    print("[VOICE] Muted.")

            elif key in (ord("s"), ord("S")):
                message = smart.navigation_summary(width)
                print("[SUMMARY]", message)
                voice.speak(message)

            elif key in (ord("p"), ord("P")):
                paused = True
                print("[VISION] Paused.")

            elif key in (ord("v"), ord("V")):
                command = voice.listen()

                action = process_voice_command(
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
                    voice.speak("Vision resumed.")

    except KeyboardInterrupt:
        print("\n[AKSHU] Interrupted.")

    except Exception as exc:
        print("[AKSHU ERROR]", type(exc).__name__, str(exc))

    finally:
        print("[AKSHU] Shutting down...")

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        voice.close()

        print("[AKSHU] Shutdown complete.")


if __name__ == "__main__":
    main()
