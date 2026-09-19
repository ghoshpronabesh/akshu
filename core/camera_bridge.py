"""
AKSHU 2.0 Camera Bridge
Provides a simple webcam interface for the AKSHU Brain.
"""

import cv2


class CameraBridge:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.camera = None

    def start(self):
        """Open the webcam."""
        if self.camera is not None and self.camera.isOpened():
            return True

        self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            self.camera = None
            return False

        return True

    def capture(self):
        """Capture one frame from the webcam."""
        if not self.start():
            return None

        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def stop(self):
        """Release the webcam."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def status(self):
        """Return camera status."""
        return {
            "available": (
                self.camera is not None
                and self.camera.isOpened()
            ),
            "camera_index": self.camera_index
        }
