from core.router import AkshuRouter
from core.vision_bridge import VisionBridge
from core.vision_intelligence import VisionIntelligence


class AkshuBrain:
    """Central intelligence layer for AKSHU."""

    def __init__(self):

        self.router = AkshuRouter()

        self.vision = None
        self.vision_intelligence = VisionIntelligence()

        self.last_vision_analysis = None

    def _get_vision(self):

        if self.vision is None:

            self.vision = VisionBridge()
            self.vision.initialize()

        return self.vision

    def analyze_image(self, image):

        """Run the complete vision pipeline."""

        if image is None:

            return {
                "message": "No image received.",
                "type": "vision_error"
            }

        try:

            vision = self._get_vision()

            analysis = vision.analyze(image)

            self.last_vision_analysis = analysis

            self.vision_intelligence.set_analysis(
                analysis
            )

            response = vision.summarize(
                analysis
            )

            return {
                "message": response,
                "type": "vision",
                "analysis": analysis
            }

        except Exception as e:

            return {
                "message": f"Vision system error: {e}",
                "type": "error"
            }

    def think(self, user_input, image=None):

        if not user_input:

            return {
                "message": "Please give me a command.",
                "type": "system"
            }

        command = user_input.strip().lower()

        # ==========================================
        # VISION QUESTION
        # ==========================================

        vision_question_words = [
            "what do you see",
            "what can you see",
            "what objects",
            "how many",
            "is there",
            "are there",
            "do you see",
            "can you see",
            "read the text",
            "read this",
            "read text",
            "who is",
            "who do you",
            "recognize",
            "what gesture",
            "what expression",
            "what emotion",
            "describe the scene",
        ]

        is_vision_question = any(
            phrase in command
            for phrase in vision_question_words
        )

        # If a new image is supplied, analyze it.
        if image is not None:

            return self.analyze_image(image)

        # If user asks a vision question and we already
        # have a previous scan, answer from that scan.
        if is_vision_question:

            return {
                "message": self.vision_intelligence.answer(
                    user_input
                ),
                "type": "vision_query"
            }

        # ==========================================
        # NORMAL TEXT COMMAND
        # ==========================================

        result = self.router.process(
            user_input
        )

        return {
            "message": result["message"],
            "type": "assistant",
            "assistant": result["assistant"],
            "version": result["version"]
        }
