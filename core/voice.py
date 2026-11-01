import json
import queue
import subprocess
import threading
import time
import tempfile
from pathlib import Path

try:
    import sounddevice as sd
    import speech_recognition as sr
    from scipy.io.wavfile import write
except ImportError:
    sd = None
    sr = None
    write = None


class AkshuVoice:
    """
    AKSHU voice module.

    Speech output uses Windows System.Speech (the same Windows speech engine
    verified by the direct PowerShell audio test). This avoids relying on
    pyttsx3's SAPI5 audio-device/COM behaviour inside the navigation thread.
    """

    POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Volume = 100
$speaker.Rate = 0
[Console]::Out.WriteLine('AKSHU_READY')
[Console]::Out.Flush()

try {
    while ($true) {
        $line = [Console]::In.ReadLine()
        if ($null -eq $line) { break }

        try {
            $item = $line | ConvertFrom-Json
            if ($item.command -eq 'STOP') { break }

            $message = [string]$item.text
            if (-not [string]::IsNullOrWhiteSpace($message)) {
                $speaker.Speak($message)
            }

            [Console]::Out.WriteLine('AKSHU_DONE')
            [Console]::Out.Flush()
        }
        catch {
            [Console]::Error.WriteLine($_.ToString())
            [Console]::Out.WriteLine('AKSHU_ERROR')
            [Console]::Out.Flush()
        }
    }
}
finally {
    $speaker.Dispose()
}
"""

    def __init__(self):
        self.recognizer = sr.Recognizer() if sr else None
        self.sample_rate = 16000
        self.record_seconds = 5

        self._speech_queue = queue.Queue(maxsize=8)
        self._closed = threading.Event()
        self._muted = threading.Event()
        self._ready = threading.Event()
        self._worker = threading.Thread(
            target=self._speech_worker,
            name="AKSHU-Voice",
            daemon=True,
        )

        self._last_message = ""
        self._last_message_time = 0.0
        self._dedupe_lock = threading.Lock()
        self._worker.start()

    def _start_windows_speech(self):
        """Start a persistent Windows System.Speech process."""
        command = [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            self.POWERSHELL_SCRIPT,
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        ready_line = process.stdout.readline().strip()
        if ready_line != "AKSHU_READY":
            try:
                error_text = process.stderr.read(2000)
            except Exception:
                error_text = ""
            process.terminate()
            raise RuntimeError(
                "Windows speech process did not become ready. "
                f"Received: {ready_line!r}. Details: {error_text}"
            )

        return process

    def _speak_windows(self, process, message):
        payload = json.dumps({"text": message}, ensure_ascii=True)
        process.stdin.write(payload + "\n")
        process.stdin.flush()

        result = process.stdout.readline().strip()
        if result == "AKSHU_DONE":
            return
        if result == "AKSHU_ERROR":
            error_text = process.stderr.readline().strip()
            raise RuntimeError(error_text or "Windows System.Speech failed.")

        raise RuntimeError(
            "Unexpected response from Windows speech process: "
            f"{result!r}; return code={process.poll()}"
        )

    def _speech_worker(self):
        process = None
        try:
            process = self._start_windows_speech()
            self._ready.set()
            print(
                "[AKSHU VOICE] Windows System.Speech engine ready.",
                flush=True,
            )

            while True:
                try:
                    message = self._speech_queue.get(timeout=0.25)
                except queue.Empty:
                    if self._closed.is_set():
                        break
                    continue

                try:
                    if message is None:
                        break
                    if self._muted.is_set():
                        print("[AKSHU VOICE] Speech skipped: muted.", flush=True)
                        continue

                    print("[AKSHU VOICE] Speaking through Windows audio:", message, flush=True)
                    self._speak_windows(process, message)
                    print("[AKSHU VOICE] Audio playback completed.", flush=True)

                except Exception as exc:
                    print(f"[AKSHU TTS ERROR] {exc}", flush=True)

                finally:
                    self._speech_queue.task_done()

        except Exception as exc:
            print(f"[AKSHU VOICE ERROR] {exc}", flush=True)

        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.stdin.write(json.dumps({"command": "STOP"}) + "\n")
                        process.stdin.flush()
                        process.wait(timeout=5)
                except Exception:
                    try:
                        process.terminate()
                    except Exception:
                        pass

            print("[AKSHU VOICE] Speech worker stopped.", flush=True)

    def speak(self, text):
        if not text or self._closed.is_set() or self._muted.is_set():
            return False

        message = str(text).strip()
        if not message:
            return False

        now = time.monotonic()
        with self._dedupe_lock:
            if (
                message == self._last_message
                and now - self._last_message_time < 3.0
            ):
                print("[AKSHU VOICE] Duplicate announcement suppressed.", flush=True)
                return False

            try:
                self._speech_queue.put_nowait(message)
            except queue.Full:
                try:
                    self._speech_queue.get_nowait()
                    self._speech_queue.task_done()
                except queue.Empty:
                    pass

                try:
                    self._speech_queue.put_nowait(message)
                except queue.Full:
                    print("[AKSHU VOICE] Queue full; announcement skipped.", flush=True)
                    return False

            self._last_message = message
            self._last_message_time = now

        print("[AKSHU VOICE QUEUED]", message, flush=True)
        return True

    def mute(self):
        self._muted.set()

    def unmute(self):
        self._muted.clear()

    @property
    def is_muted(self):
        return self._muted.is_set()

    def listen(self, timeout=5, phrase_time_limit=8):
        if sd is None or sr is None or write is None:
            print(
                "[AKSHU MIC ERROR] Install sounddevice, SpeechRecognition and scipy.",
                flush=True,
            )
            return ""

        try:
            print("[AKSHU] Listening. Speak now...", flush=True)
            recording = sd.rec(
                int(self.record_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                write(str(temp_path), self.sample_rate, recording)
                with sr.AudioFile(str(temp_path)) as source:
                    audio = self.recognizer.record(source)
                result = self.recognizer.recognize_google(audio)
                print("[YOU]", result, flush=True)
                return result
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        except sr.UnknownValueError:
            print("[AKSHU] I could not understand that.", flush=True)
            return ""
        except sr.RequestError as exc:
            print(f"[AKSHU MIC ERROR] {exc}", flush=True)
            return ""
        except Exception as exc:
            print(f"[AKSHU MIC ERROR] {exc}", flush=True)
            return ""

    def close(self):
        if self._closed.is_set():
            return

        self._closed.set()
        try:
            self._speech_queue.put(None, timeout=5)
        except queue.Full:
            print("[AKSHU VOICE WARNING] Could not queue stop signal.", flush=True)

        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=20)
            if self._worker.is_alive():
                print("[AKSHU VOICE WARNING] Speech worker is still running.", flush=True)
