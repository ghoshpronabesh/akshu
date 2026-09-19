import cv2
from pathlib import Path


class FaceIntelligence:

    def __init__(self):
        self.detector = None
        self.initialized = False
        self.last_faces = []

        self.cascade_path = (
            Path(__file__).resolve().parent.parent
            / "haarcascade_frontalface_default.xml"
        )

    def initialize(self):

        if self.initialized:
            return True

        try:

            if not self.cascade_path.exists():
                raise FileNotFoundError(
                    f"Face cascade not found: {self.cascade_path}"
                )

            self.detector = cv2.CascadeClassifier(
                str(self.cascade_path)
            )

            if self.detector.empty():
                raise RuntimeError(
                    "OpenCV face cascade could not be loaded."
                )

            self.initialized = True

            print("[AKSHU] FACE INTELLIGENCE READY")

            return True

        except Exception as e:

            print("[AKSHU FACE ERROR]:", e)

            return False

    def detect(self, frame):

        if frame is None:
            return []

        if not self.initialize():
            return []

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        results = []

        for x, y, w, h in faces:

            results.append({
                "bbox": (
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h)
                ),
                "confidence": None
            })

        self.last_faces = results

        return results

    def draw(self, frame, faces=None):

        if frame is None:
            return frame

        if faces is None:
            faces = self.last_faces

        output = frame.copy()

        for face in faces:

            x1, y1, x2, y2 = face["bbox"]

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                2
            )

            cv2.putText(
                output,
                "FACE",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA
            )

        return output

    def count(self):
        return len(self.last_faces)

    def summary(self):

        count = self.count()

        if count == 0:
            return "No face detected."

        if count == 1:
            return "I detect one face."

        return f"I detect {count} faces."

    def get_status(self):

        return {
            "initialized": self.initialized,
            "faces": self.count()
        }
