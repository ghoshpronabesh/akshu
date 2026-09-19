"""
AKSHU 2.0 Vision Bridge
Connects existing computer-vision modules to the AKSHU Brain.
"""

from typing import Dict, Any, Optional
import numpy as np

from detection.object_detector import ObjectDetector
from detection.ocr_detector import OcrDetector
from detection.face_detector import FaceDetector
from detection.gesture_detector import GestureDetector
from detection.expression_detector import ExpressionDetector


class VisionBridge:
    """Unified interface for AKSHU's existing vision modules."""

    def __init__(self):
        self.object_detector = None
        self.ocr_detector = None
        self.face_detector = None
        self.gesture_detector = None
        self.expression_detector = None

        self.status = {
            "object_detection": False,
            "ocr": False,
            "face_detection": False,
            "gesture_detection": False,
            "expression_detection": False,
        }

    def initialize(self):
        """Initialize vision modules individually so one failure does not break AKSHU."""

        try:
            self.object_detector = ObjectDetector()
            self.status["object_detection"] = True
        except Exception as e:
            print(f"[VISION] Object detector unavailable: {e}")

        try:
            self.ocr_detector = OcrDetector()
            self.status["ocr"] = True
        except Exception as e:
            print(f"[VISION] OCR unavailable: {e}")

        try:
            self.face_detector = FaceDetector(enable_recognition=True)
            self.status["face_detection"] = True
        except Exception as e:
            print(f"[VISION] Face detector unavailable: {e}")

        try:
            self.gesture_detector = GestureDetector()
            self.status["gesture_detection"] = True
        except Exception as e:
            print(f"[VISION] Gesture detector unavailable: {e}")

        try:
            self.expression_detector = ExpressionDetector()
            self.status["expression_detection"] = True
        except Exception as e:
            print(f"[VISION] Expression detector unavailable: {e}")

        return self.status

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run available vision systems on one image.

        Returns structured data that the AKSHU Brain can understand.
        """

        if image is None:
            return {
                "success": False,
                "message": "No image received.",
                "results": {}
            }

        results = {}

        # Object detection
        if self.object_detector is not None:
            try:
                objects = self.object_detector.detect_objects(image)
                results["objects"] = objects
            except Exception as e:
                results["objects"] = []
                results["object_error"] = str(e)

        # OCR
        if self.ocr_detector is not None:
            try:
                text = self.ocr_detector.detect_text(image)
                results["text"] = text
            except Exception as e:
                results["text"] = []
                results["ocr_error"] = str(e)

        # Face detection / recognition
        if self.face_detector is not None:
            try:
                faces = self.face_detector.recognize_faces(image)
                results["faces"] = faces
            except Exception as e:
                try:
                    faces = self.face_detector.detect_faces(image)
                    results["faces"] = faces
                except Exception:
                    results["faces"] = []
                    results["face_error"] = str(e)

        # Gesture detection
        if self.gesture_detector is not None:
            try:
                gestures = self.gesture_detector.detect_gestures(image)
                results["gestures"] = gestures
            except Exception as e:
                results["gestures"] = []
                results["gesture_error"] = str(e)

        # Expression detection
        if self.expression_detector is not None:
            try:
                expressions = self.expression_detector.detect_expressions(image)
                results["expressions"] = expressions
            except Exception as e:
                results["expressions"] = []
                results["expression_error"] = str(e)

        return {
            "success": True,
            "message": "Vision analysis completed.",
            "results": results
        }

    def summarize(self, analysis: Dict[str, Any]) -> str:
        """Convert raw vision results into a human-readable AKSHU response."""

        if not analysis.get("success"):
            return analysis.get("message", "Vision analysis failed.")

        results = analysis.get("results", {})
        parts = []

        # Objects
        objects = results.get("objects", [])
        if objects:
            names = []

            for obj in objects:
                if isinstance(obj, dict):
                    name = (
                        obj.get("class_name")
                        or obj.get("class")
                        or obj.get("name")
                        or obj.get("label")
                    )

                    if name:
                        names.append(str(name))

            if names:
                counts = {}
                for name in names:
                    counts[name] = counts.get(name, 0) + 1

                object_text = ", ".join(
                    f"{count} {name}" if count > 1 else f"1 {name}"
                    for name, count in counts.items()
                )

                parts.append(f"I can see {object_text}.")

        # OCR
        text_items = results.get("text", [])
        if text_items:
            detected_text = []

            for item in text_items:
                if isinstance(item, dict):
                    value = (
                        item.get("text")
                        or item.get("label")
                        or item.get("value")
                    )

                    if value:
                        detected_text.append(str(value))

                elif isinstance(item, str):
                    detected_text.append(item)

            if detected_text:
                parts.append(
                    "I detected the text: " +
                    " | ".join(detected_text[:10])
                )

        # Faces
        faces = results.get("faces", [])
        if faces:
            recognized = []

            for face in faces:
                if isinstance(face, dict):
                    name = (
                        face.get("name")
                        or face.get("person")
                        or face.get("identity")
                        or face.get("label")
                    )

                    if name:
                        recognized.append(str(name))

            if recognized:
                parts.append(
                    "I recognize: " +
                    ", ".join(recognized[:10]) + "."
                )
            else:
                parts.append(
                    f"I detect {len(faces)} face"
                    + ("s." if len(faces) != 1 else ".")
                )

        # Gestures
        gestures = results.get("gestures", [])
        if gestures:
            gesture_names = []

            for gesture in gestures:
                if isinstance(gesture, dict):
                    name = (
                        gesture.get("gesture")
                        or gesture.get("name")
                        or gesture.get("label")
                    )

                    if name:
                        gesture_names.append(str(name))

            if gesture_names:
                parts.append(
                    "Detected gesture: " +
                    ", ".join(gesture_names) + "."
                )

        # Expressions
        expressions = results.get("expressions", [])
        if expressions:
            expression_names = []

            for expression in expressions:
                if isinstance(expression, dict):
                    name = (
                        expression.get("expression")
                        or expression.get("emotion")
                        or expression.get("label")
                        or expression.get("name")
                    )

                    if name:
                        expression_names.append(str(name))

            if expression_names:
                parts.append(
                    "Detected expression: " +
                    ", ".join(expression_names) + "."
                )

        if not parts:
            return "I analyzed the image, but I couldn't identify anything confidently."

        return " ".join(parts)

    def get_status(self) -> Dict[str, bool]:
        """Return the availability of each vision module."""

        return self.status.copy()
