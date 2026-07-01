"""
MJPEG camera streamer for Raspberry Pi.

Run this on the Raspberry Pi, then enter the printed /video_feed URL in the
Akshu Streamlit sidebar on the laptop.
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Optional, Tuple, Union

import cv2
from flask import Flask, Response, jsonify


app = Flask(__name__)
camera = None


def get_lan_ip() -> str:
    """Return the LAN IP used for outbound traffic, if available."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "raspberrypi.local"
    finally:
        probe.close()


class PiCameraStream:
    """Small adapter around Picamera2 or OpenCV VideoCapture."""

    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        fps: int,
        use_picamera2: bool,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_picamera2 = use_picamera2
        self.picam2 = None
        self.cap = None

    def start(self) -> None:
        if self.use_picamera2:
            try:
                from picamera2 import Picamera2
            except ImportError as exc:
                raise RuntimeError(
                    "Picamera2 is not installed. Install it with: "
                    "sudo apt install -y python3-picamera2"
                ) from exc

            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.fps},
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.0)
            return

        source: Union[int, str]
        try:
            source = int(self.source)
        except ValueError:
            source = self.source

        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")

    def read(self) -> Tuple[bool, Optional[object]]:
        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.cap is None:
            return False, None
        return self.cap.read()

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2 = None


def mjpeg_frames():
    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), app.config["JPEG_QUALITY"]],
        )
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "stream_url": "/video_feed",
            "alternate_stream_url": "/stream.mjpg",
        }
    )


@app.route("/video_feed")
@app.route("/stream.mjpg")
def video_feed():
    return Response(
        mjpeg_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi MJPEG camera stream")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--source", default="0", help="OpenCV camera index or URL")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument(
        "--picamera2",
        action="store_true",
        help="Use the Raspberry Pi Camera Module through Picamera2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.config["JPEG_QUALITY"] = max(1, min(args.quality, 95))

    camera = PiCameraStream(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_picamera2=args.picamera2,
    )
    camera.start()

    lan_ip = get_lan_ip()
    print(f"Camera stream: http://{lan_ip}:{args.port}/video_feed")
    print("Use Ctrl+C to stop.")

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        camera.stop()
