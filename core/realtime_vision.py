import time
import cv2

from detection.object_detector import ObjectDetector
from detection.ocr_detector import OcrDetector

from core.smart_vision import SmartVision
from core.face_intelligence import FaceIntelligence


class RealtimeVisionEngine:

    _shared_object_detector = None
    _shared_face_intelligence = None
    _shared_ocr_detector = None

    def __init__(self, process_every_n_frames=5):

        self.process_every_n_frames = max(
            1,
            process_every_n_frames
        )

        self.frame_count = 0

        self.object_detector = None
        self.face_intelligence = None
        self.ocr_detector = None

        self.smart_vision = SmartVision()

        self.last_objects = []
        self.last_faces = []
        self.last_text = []

        self.initialized = False
        self.ocr_initialized = False

        self.processing_fps = 0.0
        self.last_error = None

        self.last_summary = (
            "Waiting for vision data..."
        )

    def initialize(self):

        if self.initialized:
            return True

        try:

            # -------------------------
            # YOLO
            # -------------------------

            if (
                RealtimeVisionEngine
                ._shared_object_detector
                is None
            ):

                print(
                    "[AKSHU] Loading FAST YOLO engine..."
                )

                RealtimeVisionEngine._shared_object_detector = (
                    ObjectDetector(
                        model_path="yolov8n.pt",
                        confidence_threshold=0.30,
                        imgsz=640
                    )
                )

                print(
                    "[AKSHU] YOLO MODEL LOADED"
                )

            else:

                print(
                    "[AKSHU] Reusing existing YOLO model"
                )

            self.object_detector = (
                RealtimeVisionEngine
                ._shared_object_detector
            )

            # -------------------------
            # FACE
            # -------------------------

            if (
                RealtimeVisionEngine
                ._shared_face_intelligence
                is None
            ):

                face = FaceIntelligence()

                if face.initialize():

                    RealtimeVisionEngine._shared_face_intelligence = (
                        face
                    )

            self.face_intelligence = (
                RealtimeVisionEngine
                ._shared_face_intelligence
            )

            # -------------------------
            # ENGINE READY
            # -------------------------

            self.initialized = True

            print(
                "[AKSHU] MULTI-VISION ENGINE READY"
            )

            return True

        except Exception as e:

            self.last_error = str(e)

            print(
                "[AKSHU] ENGINE ERROR:",
                e
            )

            return False

    def initialize_ocr(self):

        if self.ocr_initialized:
            return True

        try:

            if (
                RealtimeVisionEngine
                ._shared_ocr_detector
                is None
            ):

                print(
                    "[AKSHU] Loading OCR engine..."
                )

                RealtimeVisionEngine._shared_ocr_detector = (
                    OcrDetector(
                        languages=["en", "bn"]
                    )
                )

                print(
                    "[AKSHU] OCR ENGINE READY"
                )

            else:

                print(
                    "[AKSHU] Reusing existing OCR engine"
                )

            self.ocr_detector = (
                RealtimeVisionEngine
                ._shared_ocr_detector
            )

            self.ocr_initialized = True

            return True

        except Exception as e:

            self.last_error = str(e)

            print(
                "[AKSHU] OCR ERROR:",
                e
            )

            return False

    def read_text(self, frame):

        if frame is None:

            return {
                "success": False,
                "text": [],
                "message": "No image received."
            }

        if not self.initialize_ocr():

            return {
                "success": False,
                "text": [],
                "message": "OCR engine unavailable."
            }

        try:

            self.last_text = (
                self.ocr_detector.detect_text(
                    frame
                )
            )

            texts = []

            for item in self.last_text:

                if isinstance(item, dict):

                    value = (
                        item.get("text")
                        or item.get("label")
                        or item.get("value")
                    )

                    if value:
                        texts.append(
                            str(value)
                        )

                elif isinstance(item, str):

                    texts.append(item)

            if texts:

                message = (
                    "I can read: "
                    + " | ".join(texts[:10])
                )

            else:

                message = (
                    "I couldn't detect readable text."
                )

            return {
                "success": True,
                "text": self.last_text,
                "message": message
            }

        except Exception as e:

            self.last_error = str(e)

            return {
                "success": False,
                "text": [],
                "message": f"OCR error: {e}"
            }

    def process_frame(self, frame):

        if frame is None:
            return frame

        self.frame_count += 1

        output = frame.copy()

        if not self.initialize():
            return output

        if (
            self.frame_count
            % self.process_every_n_frames
            == 0
        ):

            start = time.time()

            try:

                # OBJECTS
                self.last_objects = (
                    self.object_detector.detect_objects(
                        frame
                    )
                )

                self.smart_vision.update(
                    self.last_objects
                )

                # FACES
                if self.face_intelligence:

                    self.last_faces = (
                        self.face_intelligence.detect(
                            frame
                        )
                    )

                else:

                    self.last_faces = []

                # SUMMARY
                object_summary = (
                    self.smart_vision.summary()
                )

                face_count = len(
                    self.last_faces
                )

                if face_count == 0:

                    face_summary = (
                        "No face detected."
                    )

                elif face_count == 1:

                    face_summary = (
                        "1 face detected."
                    )

                else:

                    face_summary = (
                        f"{face_count} faces detected."
                    )

                self.last_summary = (
                    object_summary
                    + " "
                    + face_summary
                )

                elapsed = (
                    time.time() - start
                )

                if elapsed > 0:

                    self.processing_fps = (
                        1.0 / elapsed
                    )

                self.last_error = None

            except Exception as e:

                self.last_error = str(e)

                print(
                    "[AKSHU VISION ERROR]:",
                    e
                )

        # -------------------------
        # DRAW OBJECTS
        # -------------------------

        if self.last_objects:

            try:

                output = (
                    self.object_detector.draw_detections(
                        output,
                        self.last_objects,
                        language="en"
                    )
                )

            except Exception as e:

                self.last_error = str(e)

        # -------------------------
        # DRAW FACES
        # -------------------------

        if (
            self.last_faces
            and self.face_intelligence
        ):

            try:

                output = (
                    self.face_intelligence.draw(
                        output,
                        self.last_faces
                    )
                )

            except Exception as e:

                self.last_error = str(e)

        # -------------------------
        # HUD
        # -------------------------

        cv2.putText(
            output,
            "AKSHU 3.0 | MULTI-VISION",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"OBJECTS: {len(self.last_objects)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"FACES: {len(self.last_faces)}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"AI FPS: {self.processing_fps:.1f}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        summary = self.last_summary

        if len(summary) > 85:

            summary = (
                summary[:82]
                + "..."
            )

        cv2.putText(
            output,
            summary,
            (15, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        return output

    def get_latest_analysis(self):

        return {
            "success": True,
            "results": {
                "objects": self.last_objects,
                "faces": self.last_faces,
                "text": self.last_text
            },
            "summary": self.last_summary
        }

    def get_status(self):

        return {
            "frames_received": self.frame_count,
            "initialized": self.initialized,
            "ocr_initialized": self.ocr_initialized,
            "objects": len(self.last_objects),
            "faces": len(self.last_faces),
            "text_items": len(self.last_text),
            "processing_fps": round(
                self.processing_fps,
                2
            ),
            "summary": self.last_summary,
            "last_error": self.last_error
        }

    def reset(self):

        self.frame_count = 0

        self.last_objects = []
        self.last_faces = []
        self.last_text = []

        self.processing_fps = 0.0

        self.last_summary = (
            "Waiting for vision data..."
        )

        self.last_error = None
