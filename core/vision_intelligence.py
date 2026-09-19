
class VisionIntelligence:
    """Natural-language intelligence layer for AKSHU vision."""

    def __init__(self):
        self.latest_analysis = None

    def set_analysis(self, analysis):
        self.latest_analysis = analysis

    def get_analysis(self):
        return self.latest_analysis

    def _results(self):
        if not self.latest_analysis:
            return {}
        return self.latest_analysis.get("results", {})

    def get_objects(self):
        objects = self._results().get("objects", [])
        return objects if isinstance(objects, list) else []

    def get_object_names(self):
        names = []

        for obj in self.get_objects():
            if isinstance(obj, dict):
                name = (
                    obj.get("name")
                    or obj.get("class_name")
                    or obj.get("class")
                    or obj.get("label")
                )

                if name:
                    names.append(str(name))

            elif isinstance(obj, str):
                names.append(obj)

        return names

    def count_object(self, object_name):
        target = object_name.lower().strip()

        return sum(
            1
            for name in self.get_object_names()
            if name.lower() == target
        )

    def summarize_objects(self):
        names = self.get_object_names()

        if not names:
            return "I don't see any recognizable objects."

        counts = {}

        for name in names:
            key = name.lower()
            counts[key] = counts.get(key, 0) + 1

        parts = []

        for name, count in counts.items():
            if count == 1:
                parts.append(f"1 {name}")
            else:
                parts.append(f"{count} {name}s")

        return "I can see " + ", ".join(parts) + "."

    def summarize_text(self):
        text_results = self._results().get("text", [])

        if not isinstance(text_results, list):
            return ""

        texts = []

        for item in text_results:
            if isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("value")
                    or item.get("content")
                )

                if text:
                    texts.append(str(text))

            elif isinstance(item, str):
                texts.append(item)

        return " ".join(texts).strip()

    def summarize_faces(self):
        faces = self._results().get("faces", [])

        if not isinstance(faces, list) or len(faces) == 0:
            return "I don't detect any faces."

        recognized = []

        for face in faces:
            if isinstance(face, dict):
                name = (
                    face.get("name")
                    or face.get("identity")
                    or face.get("person")
                )

                if name:
                    recognized.append(str(name))

        count = len(faces)

        message = f"I detect {count} face"
        if count != 1:
            message += "s"

        message += "."

        if recognized:
            message += " Recognized: " + ", ".join(recognized) + "."

        return message

    def summarize_gestures(self):
        gestures = self._results().get("gestures", [])

        if not isinstance(gestures, list) or not gestures:
            return "I don't detect any hand gestures."

        names = []

        for gesture in gestures:
            if isinstance(gesture, dict):
                name = (
                    gesture.get("gesture")
                    or gesture.get("name")
                    or gesture.get("label")
                )

                if name:
                    names.append(str(name))

            elif isinstance(gesture, str):
                names.append(gesture)

        if names:
            return "I detect: " + ", ".join(names) + "."

        return "I detect a hand gesture."

    def summarize_expressions(self):
        expressions = self._results().get("expressions", [])

        if not isinstance(expressions, list) or not expressions:
            return "I don't detect a clear facial expression."

        names = []

        for expression in expressions:
            if isinstance(expression, dict):
                name = (
                    expression.get("expression")
                    or expression.get("emotion")
                    or expression.get("name")
                    or expression.get("label")
                )

                if name:
                    names.append(str(name))

            elif isinstance(expression, str):
                names.append(expression)

        if names:
            return "Detected expression: " + ", ".join(names) + "."

        return "I detect a facial expression."

    def scene_summary(self):
        parts = [self.summarize_objects()]

        face = self.summarize_faces()

        if face != "I don't detect any faces.":
            parts.append(face)

        gesture = self.summarize_gestures()

        if gesture != "I don't detect any hand gestures.":
            parts.append(gesture)

        expression = self.summarize_expressions()

        if expression != "I don't detect a clear facial expression.":
            parts.append(expression)

        text = self.summarize_text()

        if text:
            parts.append("I can read: " + text)

        return " ".join(parts)

    def answer(self, question):
        if not self.latest_analysis:
            return "I don't have a recent camera scan. Please scan the scene first."

        q = question.lower().strip()

        if (
            "what do you see" in q
            or "what can you see" in q
            or "describe the scene" in q
            or "what objects" in q
        ):
            return self.scene_summary()

        if (
            "read the text" in q
            or "read text" in q
            or "what does it say" in q
            or "what is written" in q
        ):
            text = self.summarize_text()

            if text:
                return "The detected text says: " + text

            return "I don't detect readable text."

        if (
            "who is" in q
            or "who do you recognize" in q
            or "recognize" in q
            or "face" in q
            or "faces" in q
        ):
            return self.summarize_faces()

        if "gesture" in q or "hand" in q:
            return self.summarize_gestures()

        if (
            "expression" in q
            or "emotion" in q
            or "feeling" in q
        ):
            return self.summarize_expressions()

        if "how many" in q:
            words = q.split()

            if words:
                possible_object = words[-1].rstrip("?.,!")

                count = self.count_object(possible_object)

                return (
                    f"I detect {count} {possible_object}"
                    + ("." if count == 1 else "s.")
                )

        common_objects = [
            "person",
            "people",
            "laptop",
            "computer",
            "phone",
            "bottle",
            "chair",
            "table",
            "car",
            "motorcycle",
            "bicycle",
            "book",
            "backpack",
            "keyboard",
            "mouse",
        ]

        if "is there" in q or "are there" in q:
            for object_name in common_objects:
                if object_name in q:
                    search_name = (
                        "person"
                        if object_name == "people"
                        else object_name
                    )

                    count = self.count_object(search_name)

                    if count:
                        return f"Yes. I detect {count} {search_name}."

                    return f"No. I don't detect a {search_name}."

        return self.scene_summary()
