from collections import Counter


class SmartVision:

    def __init__(self):
        self.last_objects = []
        self.last_faces = []
        self.last_text = []
        self.last_gestures = []
        self.last_expressions = []

    # =====================================================
    # UPDATE LIVE VISION STATE
    # =====================================================

    def update(
        self,
        objects=None,
        faces=None,
        text=None,
        gestures=None,
        expressions=None
    ):

        if objects is not None:
            self.last_objects = objects or []

        if faces is not None:
            self.last_faces = faces or []

        if text is not None:
            self.last_text = text or []

        if gestures is not None:
            self.last_gestures = gestures or []

        if expressions is not None:
            self.last_expressions = expressions or []


    # =====================================================
    # OBJECT INTELLIGENCE
    # =====================================================

    def object_count(self):

        return len(self.last_objects)


    def names(self):

        return [
            obj.get("class_name", "unknown")
            for obj in self.last_objects
            if isinstance(obj, dict)
        ]


    def counts(self):

        return dict(
            Counter(self.names())
        )


    def count_object(self, name):

        name = name.lower()

        return sum(
            1
            for obj in self.last_objects
            if isinstance(obj, dict)
            and obj.get("class_name", "").lower() == name
        )


    def nearest_object(self):

        valid = [
            obj
            for obj in self.last_objects
            if isinstance(obj, dict)
            and isinstance(
                obj.get("distance"),
                (int, float)
            )
        ]

        if not valid:
            return None

        return min(
            valid,
            key=lambda obj: obj["distance"]
        )


    # =====================================================
    # FACE INTELLIGENCE
    # =====================================================

    def face_count(self):

        return len(self.last_faces)


    # =====================================================
    # OCR INTELLIGENCE
    # =====================================================

    def text_count(self):

        return len(self.last_text)


    def detected_text(self):

        output = []

        for item in self.last_text:

            if isinstance(item, dict):

                value = (
                    item.get("text")
                    or item.get("label")
                    or item.get("value")
                )

                if value:
                    output.append(
                        str(value)
                    )

            elif isinstance(item, str):

                output.append(item)

        return output


    # =====================================================
    # SCENE SUMMARY
    # =====================================================

    def summary(self):

        parts = []

        # Objects

        counts = self.counts()

        if counts:

            object_parts = []

            for name, count in counts.items():

                if count == 1:
                    object_parts.append(
                        f"1 {name}"
                    )

                else:
                    object_parts.append(
                        f"{count} {name}"
                    )

            parts.append(
                "I can see "
                + ", ".join(object_parts)
                + "."
            )


        # Faces

        if self.face_count() == 1:

            parts.append(
                "I detect one face."
            )

        elif self.face_count() > 1:

            parts.append(
                f"I detect {self.face_count()} faces."
            )


        # OCR

        text = self.detected_text()

        if text:

            parts.append(
                "I can read: "
                + " | ".join(text[:5])
                + "."
            )


        # Nothing

        if not parts:

            return (
                "I don't currently have "
                "enough visual information."
            )


        # Nearest object

        nearest = self.nearest_object()

        if nearest:

            name = nearest.get(
                "class_name",
                "object"
            )

            distance = nearest.get(
                "distance"
            )

            if isinstance(
                distance,
                (int, float)
            ):

                parts.append(
                    f"The nearest detected "
                    f"{name} is approximately "
                    f"{distance:.1f} meters away."
                )


        return " ".join(parts)


    # =====================================================
    # NATURAL LANGUAGE VISION COMMANDS
    # =====================================================

    def answer(self, question):

        question = (
            question
            .lower()
            .strip()
        )


        # ---------------------------------------------
        # SCENE
        # ---------------------------------------------

        if (
            "what do you see" in question
            or "what can you see" in question
            or "describe the scene" in question
            or "describe what you see" in question
        ):

            return self.summary()


        # ---------------------------------------------
        # PEOPLE
        # ---------------------------------------------

        if (
            "how many people" in question
            or "how many persons" in question
            or "number of people" in question
        ):

            people = self.count_object(
                "person"
            )

            faces = self.face_count()

            return (
                f"I detect {people} person(s) "
                f"and {faces} face(s)."
            )


        # ---------------------------------------------
        # FACES
        # ---------------------------------------------

        if (
            "how many faces" in question
            or "are there any faces" in question
            or "do you see any faces" in question
            or "any faces" in question
        ):

            count = self.face_count()

            if count == 0:
                return "I don't currently detect any faces."

            return (
                f"Yes. I detect {count} face(s)."
            )


        # ---------------------------------------------
        # OBJECT COUNT
        # ---------------------------------------------

        if (
            "how many objects" in question
            or "object count" in question
            or "count objects" in question
        ):

            return (
                f"I currently detect "
                f"{self.object_count()} object(s)."
            )


        # ---------------------------------------------
        # NEAREST
        # ---------------------------------------------

        if (
            "nearest" in question
            or "closest" in question
            or "what is near" in question
        ):

            nearest = self.nearest_object()

            if nearest:

                name = nearest.get(
                    "class_name",
                    "unknown"
                )

                distance = nearest.get(
                    "distance"
                )

                if isinstance(
                    distance,
                    (int, float)
                ):

                    return (
                        f"The nearest detected "
                        f"object is {name}, "
                        f"approximately "
                        f"{distance:.1f} meters away."
                    )

                return (
                    f"The nearest detected "
                    f"object is {name}."
                )

            return (
                "I don't currently have "
                "a measurable nearby object."
            )


        # ---------------------------------------------
        # OCR
        # ---------------------------------------------

        if (
            "what text" in question
            or "read the text" in question
            or "read this" in question
            or "what does it say" in question
            or "can you read" in question
        ):

            text = self.detected_text()

            if text:

                return (
                    "I can read: "
                    + " | ".join(text[:10])
                    + "."
                )

            return (
                "I don't currently have "
                "any readable text."
            )


        # ---------------------------------------------
        # SPECIFIC OBJECT
        # ---------------------------------------------

        words = question.split()

        known_objects = self.counts()

        for object_name in known_objects:

            if object_name.lower() in words:

                count = self.count_object(
                    object_name
                )

                return (
                    f"I detect {count} "
                    f"{object_name}."
                )


        # ---------------------------------------------
        # FALLBACK
        # ---------------------------------------------

        return self.summary()
