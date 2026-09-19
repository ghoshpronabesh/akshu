import speech_recognition as sr
import pyttsx3


class AkshuVoice:

    def __init__(self):

        self.recognizer = sr.Recognizer()

        self.engine = pyttsx3.init()

        self.engine.setProperty(
            "rate",
            175
        )

        self.engine.setProperty(
            "volume",
            1.0
        )

        self.listening = False


    def speak(self, text):

        if not text:
            return

        print(
            f"[AKSHU VOICE] {text}"
        )

        self.engine.say(text)

        self.engine.runAndWait()


    def listen(self, timeout=5, phrase_time_limit=8):

        try:

            with sr.Microphone() as source:

                print(
                    "[AKSHU] Listening..."
                )

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.5
                )

                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )


            print(
                "[AKSHU] Processing speech..."
            )


            text = self.recognizer.recognize_google(
                audio
            )


            print(
                f"[YOU] {text}"
            )

            return text


        except sr.WaitTimeoutError:

            return ""


        except sr.UnknownValueError:

            return ""


        except sr.RequestError as e:

            print(
                f"[AKSHU SPEECH ERROR] {e}"
            )

            return ""


        except Exception as e:

            print(
                f"[AKSHU MICROPHONE ERROR] {e}"
            )

            return ""


    def listen_and_respond(
        self,
        brain,
        image=None
    ):

        command = self.listen()

        if not command:

            self.speak(
                "I didn't catch that."
            )

            return None


        result = brain.think(
            command,
            image=image
        )


        response = result.get(
            "message",
            "I have nothing to report."
        )


        self.speak(
            response
        )


        return {
            "command": command,
            "response": response,
            "result": result
        }
