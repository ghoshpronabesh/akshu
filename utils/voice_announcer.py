"""
Voice Announcement Module for Text-to-Speech Detection Results.
"""

import threading
import time
from typing import List, Dict, Set, Optional
from queue import Queue
import pyttsx3
import asyncio
import os
import subprocess
import numpy as np
import cv2

try:
    import ollama
except ImportError:
    ollama = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

try:
    import edge_tts
except ImportError:
    edge_tts = None



# Complete dictionary of Bengali translations for all 80 COCO classes
BENGALI_CLASS_NAMES = {
    'person': 'মানুষ',
    'bicycle': 'সাইকেল',
    'car': 'গাড়ি',
    'motorcycle': 'মোটরসাইকেল',
    'airplane': 'উড়োজাহাজ',
    'bus': 'বাস',
    'train': 'ট্রেন',
    'truck': 'ট্রাক',
    'boat': 'নৌকা',
    'traffic light': 'ট্রাফিক সিগন্যাল',
    'fire hydrant': 'অগ্নিনির্বাপক কল',
    'stop sign': 'থামার নির্দেশক',
    'parking meter': 'পার্কিং মিটার',
    'bench': 'বেঞ্চ',
    'bird': 'পাখি',
    'cat': 'বিড়াল',
    'dog': 'কুকুর',
    'horse': 'ঘোড়া',
    'sheep': 'ভেড়া',
    'cow': 'গরু',
    'elephant': 'হাতি',
    'bear': 'ভাল্লুক',
    'zebra': 'জেব্রা',
    'giraffe': 'জিরাফ',
    'backpack': 'ব্যাগ',
    'umbrella': 'ছাতা',
    'handbag': 'হ্যান্ডব্যাগ',
    'tie': 'টাই',
    'suitcase': 'সুটকেস',
    'frisbee': 'ফ্রিজবি',
    'skis': 'স্কি',
    'snowboard': 'স্নোবোর্ড',
    'sports ball': 'খেলার বল',
    'kite': 'ঘুড়ি',
    'baseball bat': 'বেসবল ব্যাট',
    'baseball glove': 'বেসবল গ্লাভস',
    'skateboard': 'স্কেটবোর্ড',
    'surfboard': 'সার্ফবোর্ড',
    'tennis racket': 'টেনিস র‍্যাকেট',
    'bottle': 'বোতল',
    'wine glass': 'গ্লাস',
    'cup': 'কাপ',
    'fork': 'কাঁটাচামচ',
    'knife': 'ছুরি',
    'spoon': 'চামচ',
    'bowl': 'বাটি',
    'banana': 'কলা',
    'apple': 'আপেল',
    'sandwich': 'স্যান্ডউইচ',
    'orange': 'কমলালেবু',
    'broccoli': 'ব্রকলি',
    'carrot': 'গাজর',
    'hot dog': 'হট ডগ',
    'pizza': 'পিজ্জা',
    'donut': 'ডোনাট',
    'cake': 'কেক',
    'chair': 'চেয়ার',
    'couch': 'সোফা',
    'potted plant': 'টবগাছ',
    'bed': 'বিছানা',
    'dining table': 'খাবার টেবিল',
    'toilet': 'টয়লেট',
    'tv': 'টিভি',
    'laptop': 'ল্যাপটপ',
    'mouse': 'মাউস',
    'remote': 'রিমোট',
    'keyboard': 'কীবোর্ড',
    'cell phone': 'মোবাইল ফোন',
    'microwave': 'মাইক্রোওয়েভ',
    'oven': '오븐',  # oven
    'toaster': 'টোস্টার',
    'sink': 'বেসিন',
    'refrigerator': 'ফ্রিজ',
    'book': 'বই',
    'clock': 'ঘড়ি',
    'vase': 'ফুলদানি',
    'scissors': 'কাঁচি',
    'teddy bear': 'টেডি বিয়ার',
    'hair drier': 'হেয়ার ড্রায়ার',
    'toothbrush': 'টুথব্রাশ'
}


class VoiceAnnouncer:
    """Voice announcer for detection results with text-to-speech."""
    
    def __init__(self, announcement_cooldown: float = 2.0, language: str = 'en'):
        """
        Initialize the voice announcer.
        
        Args:
            announcement_cooldown: Minimum seconds between announcements of same item
            language: Language for TTS ('en' for English, 'bn' for Bengali)
        """
        self.announcement_cooldown = announcement_cooldown
        self.language = language
        self.enabled = False
        
        # Track last announcement times to avoid spam
        self.last_announced = {}
        
        # Queue for announcements
        self.announcement_queue = Queue()
        
        # Threading for non-blocking announcements
        self.announcement_thread = None
        self.stop_thread = False
        
        # Ollama configurations
        self.ollama_enabled = False
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "llama3.2"
        self.ollama_personality = "Friendly"
        self.ollama_vision_enabled = False
        
        # Gemini configurations
        self.gemini_enabled = False
        self.gemini_api_key = ""
        self.gemini_model = "gemini-3.1-flash-lite"
        self.last_gemini_time = 0.0
        
        # Audio output routing setting
        self.selected_device_index = None
        
        # Face tracking state for real-time distance movements
        self.face_tracks = {}  # key: track_id, value: tracking state dict
        self.next_track_id = 0
        
        # Initialize TTS engine
        self.tts_engine = None
        self._init_tts_engine()
        
        # Language-specific phrases
        self.phrases = {
            'en': {
                'person_detected': "Person detected",
                'known_person': "Hello",
                'unknown_person': "Unknown person",
                'object_detected': "detected",
                'objects_detected': "detected",
                'face_detected': "Face detected",
                'faces_detected': "faces detected",
                'and': "and",
                'i_see': "I see"
            },
            'bn': {
                'person_detected': "ব্যক্তি দেখা যাচ্ছে",
                'known_person': "হ্যালো",
                'unknown_person': "অজানা ব্যক্তি", 
                'object_detected': "দেখা যাচ্ছে",
                'objects_detected': "দেখা যাচ্ছে",
                'face_detected': "মুখ দেখা যাচ্ছে",
                'faces_detected': "মুখ দেখা যাচ্ছে",
                'and': "এবং",
                'i_see': "আমি দেখতে পাচ্ছি"
            }
        }
        
    def _init_tts_engine(self):
        """Initialize text-to-speech engine."""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            self.tts_engine.setProperty('rate', 160)  # Moderate, highly clear rate for visually impaired
            self.tts_engine.setProperty('volume', 0.9)  # Volume level
            
            # Try to set premium system voice for language
            self._update_voice_for_language()
            
            print("✅ TTS Engine initialized successfully")
            
        except ImportError:
            print("⚠️ pyttsx3 not available. Using fallback announcement method.")
            self.tts_engine = None
        except Exception as e:
            print(f"⚠️ TTS Engine initialization failed: {e}")
            self.tts_engine = None

    def _update_voice_for_language(self):
        """Find and select the highest-quality premium system voice for the active language."""
        if not self.tts_engine:
            return
            
        try:
            voices = self.tts_engine.getProperty('voices')
            target_voice_id = None
            
            if self.language == 'bn':
                # Explicitly search for macOS high-quality Bengali voice 'Paya' (Piya) or 'bn_IN'
                for voice in voices:
                    if 'bn' in voice.languages[0] or 'Paya' in voice.name or 'Piya' in voice.name:
                        target_voice_id = voice.id
                        break
            else:
                # Explicitly search for macOS famous premium English voice 'Samantha' or 'en_US'
                for voice in voices:
                    if 'Samantha' in voice.name or 'en_US' in voice.languages[0]:
                        target_voice_id = voice.id
                        break
                        
            # Secondary check for Indian English accent 'Rishi' or 'Tara' if US English Samantha not found
            if not target_voice_id and self.language != 'bn':
                for voice in voices:
                    if 'Rishi' in voice.name or 'Tara' in voice.name or 'en_IN' in voice.languages[0]:
                        target_voice_id = voice.id
                        break
                        
            # General fallback to any English voice
            if not target_voice_id and self.language != 'bn':
                for voice in voices:
                    if 'en' in voice.languages[0]:
                        target_voice_id = voice.id
                        break
            
            if target_voice_id:
                self.tts_engine.setProperty('voice', target_voice_id)
                print(f"🔊 Loaded premium voice: {target_voice_id}")
            else:
                # Absolute fallback to first system voice
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
                    print(f"⚠️ Premium voice for '{self.language}' not found. Defaulting to system voice.")
                    
        except Exception as e:
            print(f"Error updating system voice: {e}")
    
    def enable_voice(self, enable: bool):
        """
        Enable or disable voice announcements.
        
        Args:
            enable: Whether to enable voice mode
        """
        self.enabled = enable
        
        if enable and not self.announcement_thread:
            self.start_announcement_thread()
        elif not enable:
            self.stop_speaking()  # Instantly silence and clear queue!
            if self.announcement_thread:
                self.stop_announcement_thread()
    
    def start_announcement_thread(self):
        """Start the announcement worker thread."""
        self.stop_thread = False
        self.announcement_thread = threading.Thread(
            target=self._announcement_worker, 
            daemon=True
        )
        self.announcement_thread.start()
        print("🔊 Voice announcement thread started")
    
    def stop_announcement_thread(self):
        """Stop the announcement worker thread."""
        self.stop_thread = True
        self.stop_speaking()  # Instantly silence!
        if self.announcement_thread:
            self.announcement_thread.join(timeout=1.0)
            self.announcement_thread = None
        print("🔇 Voice announcement thread stopped")

    def stop_speaking(self):
        """Instantly stop any active speech playback and clear the queue."""
        # Empty the queue entirely to discard pending announcements
        while not self.announcement_queue.empty():
            try:
                self.announcement_queue.get_nowait()
            except Exception:
                break

        # Instantly silence pyttsx3 offline speech engine if it is active
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                print(f"Error stopping pyttsx3 engine: {e}")
                
        # Terminate any active afplay process instantly on macOS
        try:
            os.system("killall afplay 2>/dev/null")
        except Exception:
            pass

        # Clean up any leftover temp speech files in the directory
        try:
            for file in os.listdir(os.getcwd()):
                if file.startswith("temp_speech_") and file.endswith(".mp3"):
                    try:
                        os.remove(os.path.join(os.getcwd(), file))
                    except Exception:
                        pass
        except Exception:
            pass
    
    def _announcement_worker(self):
        """Worker thread for processing announcements."""
        while not self.stop_thread:
            try:
                if not self.announcement_queue.empty():
                    item = self.announcement_queue.get(timeout=0.1)
                    if item and self.enabled:
                        if isinstance(item, dict) and item.get('type') == 'detection':
                            text = self._generate_ai_response(item)
                        else:
                            text = item
                            
                        if text:
                            self._speak_text(text)
                time.sleep(0.1)
            except Exception as e:
                print(f"Announcement worker error: {e}")
                
    def _generate_ai_response(self, payload: dict) -> str:
        """
        Generate a conversational AI response using cloud Gemini model or local Ollama model.
        Falls back to standard templated voice string on error or if disabled.
        """
        if not self.ollama_enabled and not self.gemini_enabled:
            return payload.get('raw_text', '')
            
        # Format detection info for LLM
        faces = payload.get('faces', [])
        objects = payload.get('objects', [])
        expressions = payload.get('expressions', [])
        gestures = payload.get('gestures', [])
        
        description_parts = []
        
        # Formulate faces description with matched expressions
        if faces:
            face_desc = []
            for f in faces:
                name = f.get('name', 'Unknown')
                is_known = f.get('is_known', False)
                bbox = f.get('bbox', (0,0,0,0))
                xf, yf, wf, hf = bbox
                
                # Match expression
                expr = "Neutral"
                if expressions:
                    face_center = np.array([xf + wf/2, yf + hf/2])
                    best_dist = float('inf')
                    for e in expressions:
                        e_box = e['bbox']
                        e_center = np.array([(e_box[0] + e_box[2])/2, (e_box[1] + e_box[3])/2])
                        dist = np.linalg.norm(face_center - e_center)
                        if dist < best_dist and dist < (wf * 0.8):
                              best_dist = dist
                              expr = e['expression']
                              
                expr_str = f" showing {expr.lower()} expression" if expr != "Neutral" else ""
                
                # Add movement direction to description if present
                movement = f.get('movement', '')
                movement_desc = f" and is {movement}" if movement else ""
                
                if is_known and name != 'Unknown':
                    face_desc.append(f"{name}{expr_str}{movement_desc}")
                else:
                    face_desc.append(f"an unknown person{expr_str}{movement_desc}")
            description_parts.append(f"People: {', and '.join(face_desc)}")
            
        # Formulate hand gestures / sign language
        if gestures:
            hand_desc = []
            for g in gestures:
                gesture = g.get('gesture', 'Unknown Sign')
                if gesture != 'Unknown Sign':
                    hand_desc.append(f"{g.get('hand_type', 'hand')} hand showing {gesture}")
            if hand_desc:
                description_parts.append(f"Hand signs: {', '.join(hand_desc)}")
            
        # Formulate objects description
        if objects:
            obj_desc = []
            for obj in objects:
                name = obj.get('class_name', 'object')
                dist = obj.get('distance', 0.0)
                obj_desc.append(f"{name} at {dist:.1f} meters")
            description_parts.append(f"Objects: {', '.join(obj_desc)}")
            
        if not description_parts:
            return ""
            
        detections_summary = "; ".join(description_parts)
        language_name = "English" if self.language == 'en' else "Bengali"
        
        # --- Option B: Cloud Gemini API ---
        if self.gemini_enabled and self.gemini_api_key:
            current_time = time.time()
            if current_time - getattr(self, 'last_gemini_time', 0.0) < 5.0:
                print("⏳ Throttling Gemini API call to respect Free Tier rate limits (max 12 RPM). Falling back to local/raw text.")
            else:
                # First try using system curl because Python's built-in SSL/urllib3/requests
                # might have LibreSSL handshake issues on macOS.
                try:
                    self.last_gemini_time = current_time
                    import base64
                    import json
                    import subprocess
                    
                    print(f"♊ Preparing Gemini API request ({self.gemini_model}) via system curl...")
                    
                    contents = []
                    frame = payload.get('frame')
                    if frame is not None:
                        try:
                            # Resize image to maximum 640 width to reduce base64 size and speed up curl transmission
                            h, w = frame.shape[:2]
                            if w > 640:
                                resized_frame = cv2.resize(frame, (640, int(h * 640 / w)))
                            else:
                                resized_frame = frame
                            
                            success, encoded_image = cv2.imencode('.jpg', resized_frame)
                            if success:
                                image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                                contents.append({
                                    "inlineData": {
                                        "mimeType": "image/jpeg",
                                        "data": image_base64
                                    }
                                })
                                print(f"📸 Encoded frame base64 (size: {len(image_base64)} chars) for curl.")
                        except Exception as enc_err:
                            print(f"⚠️ Failed to encode frame for curl: {enc_err}")
                    
                    system_prompt = (
                        f"You are a helpful AI assistant for a visually impaired user. "
                        f"Your task is to analyze the image frame and describe what is in front of the user in a single natural, conversational sentence in {language_name}. "
                        f"Use the detection summary as a guide: {detections_summary}. "
                        f"Rely on the image to verify detections and resolve any false positives (e.g. if the detections show 'tie' but you see a window/grates, do not mention a tie). "
                        f"Keep the description extremely brief (under 15 words) and highly relevant for navigation/awareness. "
                        f"Tone/Personality: {self.ollama_personality}.\n"
                        f"Rules:\n"
                        f"1. Rely on the image. Do NOT mention false positive object detections (like window grates identified as a tie).\n"
                        f"2. Respond ONLY with the final output sentence in {language_name}.\n"
                        f"3. Do not include any explanations, introduction, markdown, or quotes."
                    )
                    
                    user_prompt = f"Describe the scene briefly and naturally in {language_name} based on the rules. Detection guide: {detections_summary}."
                    contents.append({"text": user_prompt})
                    
                    post_data = {
                        "contents": [{"parts": contents}],
                        "systemInstruction": {
                            "parts": [{"text": system_prompt}]
                        },
                        "generationConfig": {
                            "temperature": 0.3
                        }
                    }
                    
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
                    
                    start_time = time.time()
                    res = subprocess.run([
                        "curl", "-s", "-w", "\nHTTP_STATUS_CODE:%{http_code}\n",
                        "-X", "POST",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps(post_data),
                        url
                    ], capture_output=True, text=True, timeout=10)
                    
                    elapsed = time.time() - start_time
                    stdout = res.stdout
                    
                    # find HTTP status
                    status_code = "unknown"
                    for line in stdout.split('\n'):
                        if "HTTP_STATUS_CODE:" in line:
                            status_code = line.split(':')[-1].strip()
                            break
                    
                    if status_code == "200":
                        clean_stdout = '\n'.join([line for line in stdout.split('\n') if "HTTP_STATUS_CODE:" not in line])
                        res_json = json.loads(clean_stdout)
                        ai_text = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                        
                        if ai_text.startswith('"') and ai_text.endswith('"'):
                            ai_text = ai_text[1:-1].strip()
                        if ai_text.startswith("'") and ai_text.endswith("'"):
                            ai_text = ai_text[1:-1].strip()
                            
                        if ai_text:
                            if not self._check_hallucination(ai_text, objects, faces):
                                print(f"✨ Gemini AI Response Generated via curl ({self.gemini_model}) in {elapsed:.2f}s: {ai_text}")
                                return ai_text
                            else:
                                print(f"🔄 Gemini response failed hallucination guardrail. Falling back.")
                    else:
                        print(f"⚠️ Gemini curl request failed with HTTP status {status_code}: {stdout[:300]}")
                except Exception as curl_err:
                    print(f"⚠️ Gemini curl request exception: {curl_err}")

        # --- Option A: Local Ollama Model ---
        if self.ollama_enabled and ollama is not None:
            try:
                # Create local client
                client = ollama.Client(host=self.ollama_url)
                
                # Few-shot examples based on language to steer small models
                if self.language == 'bn':
                    few_shot_examples = (
                        "Here are examples of how you must rephrase the input:\n"
                        "Input: Detections: Objects: bottle at 0.3 meters. Raw: bottle at 0.3m.\n"
                        "Output: আপনার খুব কাছে, মাত্র ৩০ সেন্টিমিটার দূরে একটি বোতল রয়েছে।\n"
                        "Input: Detections: People: pronabesh showing happy expression; Objects: cell phone at 0.4 meters. Raw: Hello pronabesh, cell phone at 0.4m.\n"
                        "Output: হ্যালো pronabesh! আপনাকে দেখে খুশি মনে হচ্ছে, আর কাছেই একটি মোবাইল ফোন আছে।\n"
                        "Input: Detections: Objects: chair at 2.5 meters. Raw: chair at 2.5m.\n"
                        "Output: সামনে প্রায় আড়াই মিটার দূরে একটি চেয়ার আছে।"
                    )
                else:
                    few_shot_examples = (
                        "Here are examples of how you must rephrase the input:\n"
                        "Input: Detections: Objects: bottle at 0.3 meters. Raw: bottle at 0.3m.\n"
                        "Output: There is a bottle very close to you, about 30 centimeters away.\n"
                        "Input: Detections: People: pronabesh showing happy expression; Objects: cell phone at 0.4 meters. Raw: Hello pronabesh, cell phone at 0.4m.\n"
                        "Output: Hey pronabesh! You are smiling, and there is a phone nearby.\n"
                        "Input: Detections: Objects: chair at 2.5 meters. Raw: chair at 2.5m.\n"
                        "Output: A chair is located about two and a half meters ahead."
                    )
                # Encode frame if vision mode is enabled
                image_bytes = None
                if getattr(self, 'ollama_vision_enabled', False) and payload.get('frame') is not None:
                    try:
                        success, encoded_image = cv2.imencode('.jpg', payload['frame'])
                        if success:
                            image_bytes = encoded_image.tobytes()
                            print(f"📸 Frame encoded successfully for Ollama Vision API (Model: {self.ollama_model}).")
                    except Exception as img_err:
                        print(f"⚠️ Error encoding frame for Ollama Vision: {img_err}")

                if image_bytes:
                    # Prompt optimized for Vision-Language Models (e.g. llava, moondream)
                    system_prompt = (
                        f"You are a helpful AI assistant for a visually impaired user. "
                        f"Your task is to analyze the image frame and describe what is in front of the user in a single natural, conversational sentence in {language_name}. "
                        f"Use the detection summary as a guide: {detections_summary}. "
                        f"Rely on the image to verify detections and resolve any false positives (e.g. if the detections show 'tie' but you see a window/grates, do not mention a tie). "
                        f"Keep the description extremely brief (under 15 words) and highly relevant for navigation/awareness. "
                        f"Tone/Personality: {self.ollama_personality}.\n"
                        f"Rules:\n"
                        f"1. Rely on the image. Do NOT mention false positive object detections (like window grates identified as a tie).\n"
                        f"2. Respond ONLY with the final output sentence in {language_name}.\n"
                        f"3. Do not include any explanations, introduction, markdown, or quotes."
                    )
                    user_prompt = f"Describe the image briefly and naturally in {language_name} based on the rules. Detection guide: {detections_summary}."
                else:
                    # Text-only LLM prompt
                    system_prompt = (
                        f"You are a helpful AI assistant for a visually impaired user. "
                        f"Your task is to convert the raw object/face detections into a natural, conversational sentence in {language_name}. "
                        f"Keep the description extremely brief (under 15 words) and highly relevant for navigation/awareness. "
                        f"Tone/Personality: {self.ollama_personality}.\n"
                        f"{few_shot_examples}\n"
                        f"Rules:\n"
                        f"1. Describe ONLY the objects/people listed in the input detections summary. Do NOT assume, suggest, or add any other objects, background details, or furniture not present in the input. This is critical for the safety of a blind user.\n"
                        f"2. Respond ONLY with the final output sentence in {language_name}.\n"
                        f"3. Do not include any explanations, introduction, markdown, or quotes."
                    )
                    user_prompt = f"Input: Detections: {detections_summary}. Raw: {payload.get('raw_text', '')}.\nOutput:"
                
                # Run generate
                response = client.generate(
                    model=self.ollama_model,
                    system=system_prompt,
                    prompt=user_prompt,
                    images=[image_bytes] if image_bytes else None,
                    options={
                        'temperature': 0.3,
                        'num_predict': 50
                    }
                )
                
                ai_text = response.get('response', '').strip()
                if ai_text.startswith('"') and ai_text.endswith('"'):
                    ai_text = ai_text[1:-1].strip()
                if ai_text.startswith("'") and ai_text.endswith("'"):
                    ai_text = ai_text[1:-1].strip()
                    
                if ai_text:
                    if not self._check_hallucination(ai_text, objects, faces):
                        print(f"🤖 Ollama AI Response Generated ({self.ollama_model}): {ai_text}")
                        return ai_text
            except Exception as e:
                print(f"⚠️ Ollama generation failed: {e}. Falling back to standard raw text.")

        return payload.get('raw_text', '')

    def _check_hallucination(self, ai_text: str, objects: list, faces: list) -> bool:
        """
        Verify if the generated text contains objects/names that were not detected.
        Returns True if hallucination was found, False otherwise.
        """
        ai_text_lower = ai_text.lower()
        
        # Gather all detected items
        detected_names = set()
        for obj in objects:
            c_name = obj.get('class_name', '').lower()
            if c_name:
                detected_names.add(c_name)
                # Add Bengali translation for check
                bn_trans = BENGALI_CLASS_NAMES.get(c_name, '')
                if bn_trans:
                    detected_names.add(bn_trans.lower())
                    
        for f in faces:
            f_name = f.get('name', '').lower()
            if f_name and f_name != 'unknown' and f_name != 'face':
                detected_names.add(f_name)
                
        # Check for mention of standard classes that were NOT detected
        for cls_en, cls_bn in BENGALI_CLASS_NAMES.items():
            if cls_en not in detected_names and cls_bn.lower() not in detected_names:
                # Simple word boundary check
                has_en_word = f" {cls_en} " in f" {ai_text_lower} " or ai_text_lower.startswith(f"{cls_en} ") or ai_text_lower.endswith(f" {cls_en}")
                has_bn_word = cls_bn.lower() in ai_text_lower
                
                if has_en_word or has_bn_word:
                    # Skip if "person" / "মানুষ" is mentioned generally and there are people
                    if (cls_en == 'person' or cls_en == 'chair') and faces:
                        continue
                    print(f"⚠️ Hallucination Guardrail triggered! AI mentioned '{cls_en}'/'{cls_bn}' which is not in detections list. Rejecting response.")
                    return True
        return False
    
    def get_audio_output_devices(self) -> List[Dict]:
        """Get list of available audio output devices."""
        devices = []
        if sd is None:
            return devices
        try:
            for idx, d in enumerate(sd.query_devices()):
                if d['max_output_channels'] > 0:
                    devices.append({
                        'index': idx,
                        'name': d['name'],
                        'hostapi': d['hostapi']
                    })
        except Exception as e:
            print(f"Error querying audio devices: {e}")
        return devices

    def _speak_text(self, text: str):
        """
        Speak the given text using modern Microsoft Edge Neural AI TTS.
        Falls back to native system pyttsx3 offline synthesizer if network is unavailable.
        
        Args:
            text: Text to speak
        """
        if not self.enabled or self.stop_thread:
            return

        try:
            if edge_tts is None:
                raise ImportError("edge_tts module is not installed")
                
            # 1. Attempt to use premium Edge Neural AI TTS
            # Choose highly realistic, human-like neural voices
            if self.language == 'bn':
                # 'Tanisha' is a highly premium, natural, clear female Bengali voice
                edge_voice = 'bn-IN-TanishaaNeural'
            else:
                # 'Aria' is a highly premium, natural, friendly female US English voice
                edge_voice = 'en-US-AriaNeural'
                
            # Define a safe temporary audio file path in the workspace
            temp_file = os.path.join(os.getcwd(), f"temp_speech_{int(time.time())}.mp3")
            
            async def generate_speech_audio():
                communicate = edge_tts.Communicate(text, edge_voice)
                await communicate.save(temp_file)
                
            # Check before generating
            if not self.enabled or self.stop_thread:
                return

            # Run the asynchronous speech synthesis synchronously inside the background thread
            asyncio.run(generate_speech_audio())
            
            # Check after generating but before playing
            if not self.enabled or self.stop_thread:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
                return

            if os.path.exists(temp_file):
                print(f"🔊 AI Speaking ({self.language}): {text}")
                
                # If a specific Bluetooth or output device is selected, route there
                if self.selected_device_index is not None and sd is not None and sf is not None:
                    try:
                        data, samplerate = sf.read(temp_file)
                        sd.play(data, samplerate, device=self.selected_device_index)
                        sd.wait()  # Block queue worker until speech is finished playing
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass
                        return
                    except Exception as play_err:
                        print(f"⚠️ Playback via sounddevice failed: {play_err}. Falling back to default system output.")
                
                # Play using macOS native high-performance audio player (afplay)
                # Subprocess blocks the queue worker during playback, keeping announcements perfectly sequenced!
                subprocess.run(["afplay", temp_file])
                
                # Clean up temporary file
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
                return  # Success, exit!
                
        except Exception as e:
            print(f"⚠️ Edge Neural TTS failed: {e}. Falling back to macOS native say offline system voice.")
            
        # 2. Offline Fallback using macOS native 'say' command (thread-safe and routes correctly to custom device)
        if not self.enabled or self.stop_thread:
            return

        try:
            # Select appropriate voice
            voice_name = None
            if self.language == 'bn':
                voice_name = 'Piya'
            else:
                voice_name = 'Daniel'
                
            temp_aiff = os.path.join(os.getcwd(), f"temp_speech_{int(time.time())}.aiff")
            
            # Execute say command in shell
            cmd = ["say"]
            if voice_name:
                cmd.extend(["-v", voice_name])
            cmd.extend(["-o", temp_aiff, text])
            
            print(f"🔊 Running offline say command: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0:
                print(f"⚠️ Voice '{voice_name}' not available or failed. Retrying with default system voice.")
                cmd = ["say", "-o", temp_aiff, text]
                subprocess.run(cmd, check=True)
            
            if os.path.exists(temp_aiff):
                if self.selected_device_index is not None and sd is not None and sf is not None:
                    try:
                        data, samplerate = sf.read(temp_aiff)
                        sd.play(data, samplerate, device=self.selected_device_index)
                        sd.wait()
                        try:
                            os.remove(temp_aiff)
                        except Exception:
                            pass
                        return
                    except Exception as play_err:
                        print(f"⚠️ sounddevice offline playback failed: {play_err}. Falling back to default afplay.")
                
                # Default afplay routing
                subprocess.run(["afplay", temp_aiff])
                try:
                    os.remove(temp_aiff)
                except Exception:
                    pass
                return
                
        except Exception as say_err:
            print(f"⚠️ macOS say command offline fallback failed: {say_err}. Trying pyttsx3.")

        # 3. Last resort fallback using pyttsx3
        try:
            if self.tts_engine:
                print(f"🔊 System Fallback Speaking (pyttsx3): {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        except Exception as pyttsx3_err:
            print(f"❌ pyttsx3 last resort fallback failed: {pyttsx3_err}")
    
    def announce_detections(self, face_results: List[Dict], object_results: List[Dict],
                           expression_results: Optional[List[Dict]] = None,
                           gesture_results: Optional[List[Dict]] = None,
                           frame: Optional[np.ndarray] = None):
        """
        Announce detection results, including facial expressions and hand signs.
        
        Args:
            face_results: List of face detection/recognition results
            object_results: List of object detection results
            expression_results: List of facial expression results
            gesture_results: List of hand sign language results
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        announcements = []
        
        # Process face detections (matched with expressions)
        if face_results:
            face_announcements = self._process_face_detections(face_results, expression_results, current_time)
            announcements.extend(face_announcements)
        
        # Process object detections
        if object_results:
            object_announcements = self._process_object_detections(object_results, current_time)
            announcements.extend(object_announcements)
            
        # Process hand gestures (sign language)
        if gesture_results:
            gesture_announcements = self._process_gesture_detections(gesture_results, current_time)
            announcements.extend(gesture_announcements)
        
        # Queue announcements
        if announcements:
            combined_announcement = self._combine_announcements(announcements)
            # Empty the queue first to discard old backlog and ensure real-time responsiveness
            while self.announcement_queue.qsize() > 0:
                try:
                    self.announcement_queue.get_nowait()
                except Exception:
                    break
            
            if self.ollama_enabled or self.gemini_enabled:
                payload = {
                    'type': 'detection',
                    'faces': face_results,
                    'objects': object_results,
                    'expressions': expression_results,
                    'gestures': gesture_results,
                    'raw_text': combined_announcement,
                    'frame': frame
                }
                self.announcement_queue.put(payload)
            else:
                self.announcement_queue.put(combined_announcement)
    
    def _update_face_tracking(self, face_results: List[Dict], current_time: float) -> List[str]:
        """
        Track faces across frames and detect if they are moving closer or farther away.
        
        Returns:
            List of movement announcement strings.
        """
        movement_announcements = []
        
        # 1. Clean up old tracks (not seen for > 5.0 seconds)
        inactive_ids = [tid for tid, track in self.face_tracks.items() if current_time - track['last_seen'] > 5.0]
        for tid in inactive_ids:
            del self.face_tracks[tid]
            
        # 2. Match current face detections to existing tracks
        unmatched_tracks = list(self.face_tracks.keys())
        
        for result in face_results:
            name = result.get('name', 'Unknown')
            is_known = result.get('is_known', False)
            bbox = result.get('bbox', (0,0,0,0))
            xf, yf, wf, hf = bbox
            cx = xf + wf / 2.0
            cy = yf + hf / 2.0
            distance = result.get('distance', 0.0)
            
            best_match_id = None
            best_match_score = float('inf') # lower is better
            
            # Match by name first if known
            if is_known and name != 'Unknown' and name != 'Face':
                if name in self.face_tracks:
                    best_match_id = name
                    best_match_score = 0.0
                    
            # Match by spatial centroid distance if name match is not possible or found
            if best_match_id is None:
                for tid in unmatched_tracks:
                    track = self.face_tracks[tid]
                    tcx, tcy = track['last_centroid']
                    spatial_dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                    if spatial_dist < 500.0 and spatial_dist < best_match_score:
                        best_match_score = spatial_dist
                        best_match_id = tid
                        
            # Update or create track
            if best_match_id is not None:
                if best_match_id in unmatched_tracks:
                    unmatched_tracks.remove(best_match_id)
                    
                track = self.face_tracks[best_match_id]
                track['last_centroid'] = (cx, cy)
                track['last_seen'] = current_time
                
                # Smooth the distance using a moving average
                track['distance_history'].append(distance)
                if len(track['distance_history']) > 5:
                    track['distance_history'].pop(0)
                smoothed_dist = sum(track['distance_history']) / len(track['distance_history'])
                track['last_distance'] = smoothed_dist
                
                # Check movement direction relative to reference_distance
                ref_dist = track['reference_distance']
                dist_change = smoothed_dist - ref_dist
                
                print(f"👤 Track '{best_match_id}' update: dist={smoothed_dist:.2f}m, ref={ref_dist:.2f}m, diff={dist_change:.2f}m")
                
                if abs(dist_change) >= 0.25 and current_time - track['last_announce_time'] >= 3.0:
                    display_name = name if (is_known and name != 'Unknown') else ('ব্যক্তিটি' if self.language == 'bn' else 'The person')
                    
                    if dist_change > 0:
                        # Moving away
                        result['movement'] = 'moving away'
                        if self.language == 'bn':
                            msg = f"{display_name} দূরে সরে যাচ্ছে"
                        else:
                            msg = f"{display_name} is moving away"
                    else:
                        # Coming closer
                        result['movement'] = 'coming closer'
                        if self.language == 'bn':
                            msg = f"{display_name} কাছে এগিয়ে আসছে"
                        else:
                            msg = f"{display_name} is coming closer"
                            
                    print(f"🔊 Movement Announcement triggered: '{msg}'")
                    movement_announcements.append(msg)
                    track['reference_distance'] = smoothed_dist
                    track['last_announce_time'] = current_time
            else:
                # Create a new track
                new_id = name if (is_known and name != 'Unknown' and name != 'Face') else f"unknown_{self.next_track_id}"
                if not is_known or name == 'Unknown' or name == 'Face':
                    self.next_track_id += 1
                    
                self.face_tracks[new_id] = {
                    'name': name,
                    'is_known': is_known,
                    'last_centroid': (cx, cy),
                    'last_distance': distance,
                    'reference_distance': distance,
                    'distance_history': [distance],
                    'last_seen': current_time,
                    'last_announce_time': 0.0
                }
                print(f"👤 Track '{new_id}' created: dist={distance:.2f}m")
                
        return movement_announcements

    def _process_face_detections(self, face_results: List[Dict], expression_results: Optional[List[Dict]], current_time: float) -> List[str]:
        """Process face detection results and match with expressions for announcements."""
        # Update face tracking for distance movement announcements
        announcements = self._update_face_tracking(face_results, current_time)
        phrases = self.phrases[self.language]
        
        unknown_count = 0
        
        for result in face_results:
            name = result.get('name', 'Unknown')
            is_known = result.get('is_known', False)
            bbox = result.get('bbox', (0,0,0,0))
            xf, yf, wf, hf = bbox
            
            # Match expression
            expr = "Neutral"
            if expression_results:
                face_center = np.array([xf + wf/2, yf + hf/2])
                best_dist = float('inf')
                for e in expression_results:
                    e_box = e['bbox']
                    e_center = np.array([(e_box[0] + e_box[2])/2, (e_box[1] + e_box[3])/2])
                    dist = np.linalg.norm(face_center - e_center)
                    if dist < best_dist and dist < (wf * 0.8):
                        best_dist = dist
                        expr = e['expression']
            
            # Formulate description with expression
            expr_desc_en = f" looking {expr.lower()}" if expr != "Neutral" else ""
            expr_bn_map = {'Happy': 'খুশি', 'Sad': 'দুঃখিত', 'Angry': 'রাগান্বিত', 'Surprised': 'অবাক'}
            expr_desc_bn = f", যাকে দেখে {expr_bn_map.get(expr, '')} মনে হচ্ছে" if expr in expr_bn_map else ""
            
            # If this person moved on this frame, we suppress the standard static greeting
            # to prevent voice clutter and double announcements.
            if 'movement' in result:
                continue
                
            if is_known and name != 'Unknown':
                # Check cooldown for this specific person
                last_time = self.last_announced.get(f"person_{name}", 0)
                if current_time - last_time >= self.announcement_cooldown:
                    if self.language == 'bn':
                        announcements.append(f"{phrases['known_person']} {name}{expr_desc_bn}")
                    else:
                        announcements.append(f"{phrases['known_person']} {name}{expr_desc_en}")
                    self.last_announced[f"person_{name}"] = current_time
            else:
                unknown_count += 1
                # If unknown, we can announce the expression if not neutral
                if expr != "Neutral":
                    last_time = self.last_announced.get(f"unknown_expression_{expr}", 0)
                    if current_time - last_time >= self.announcement_cooldown:
                        if self.language == 'bn':
                            announcements.append(f"একজন {expr_bn_map.get(expr, '')} ব্যক্তি")
                        else:
                            announcements.append(f"A {expr.lower()} unknown person")
                        self.last_announced[f"unknown_expression_{expr}"] = current_time
        
        # Standard unknown count announcement (if no expressions were announced or in general)
        if unknown_count > 0:
            last_time = self.last_announced.get("unknown_person", 0)
            if current_time - last_time >= self.announcement_cooldown:
                if unknown_count == 1:
                    announcements.append(phrases['unknown_person'])
                else:
                    announcements.append(f"{unknown_count} {phrases['unknown_person']}")
                self.last_announced["unknown_person"] = current_time
                
        return announcements
        
    def _process_gesture_detections(self, gesture_results: List[Dict], current_time: float) -> List[str]:
        """Process hand gesture detection results for announcements."""
        announcements = []
        
        for det in gesture_results:
            gesture = det.get('gesture', 'Unknown Sign')
            if gesture == 'Unknown Sign':
                continue
                
            last_time = self.last_announced.get(f"gesture_{gesture}", 0)
            # Gestures have a cooldown of 3 seconds to avoid repeating "OK Sign" constantly
            if current_time - last_time >= 3.0:
                if self.language == 'bn':
                    gesture_bn_map = {
                        'Open Palm (Hello)': 'হ্যালো বা থামার ইশারা',
                        'Fist (Yes)': 'মুষ্টিবদ্ধ হাত বা হ্যাঁ সূচক ইশারা',
                        'Thumbs Up (Like)': 'লাইক বা ভালো ইশারা',
                        'Thumbs Down (Dislike)': 'ডিসলাইক বা মন্দ ইশারা',
                        'Peace Sign (V)': 'বিজয় চিহ্ন',
                        'OK Sign': 'ওকে বা ঠিক আছে ইশারা',
                        'Call Me Sign': 'কল মি বা ফোন করার ইশারা',
                        'I Love You Sign': 'আই লাভ ইউ ইশারা',
                        'Pointing Sign': 'ইঙ্গিত বা নির্দেশক ইশারা'
                    }
                    announcements.append(f"হাত দিয়ে {gesture_bn_map.get(gesture, gesture)} করা হচ্ছে")
                else:
                    announcements.append(f"Hand sign {gesture}")
                self.last_announced[f"gesture_{gesture}"] = current_time
                
        return announcements
    
    def _process_object_detections(self, object_results: List[Dict], current_time: float) -> List[str]:
        """
        Process object detection results for announcements with count and distance tracking.
        
        Args:
            object_results: List of detected objects
            current_time: Current timestamp
            
        Returns:
            List of announcement strings
        """
        announcements = []
        
        # Group objects by class and find the minimum distance for each
        groups = {}
        for result in object_results:
            class_name = result.get('class_name', 'object')
            dist = result.get('distance', 0.0)
            
            if class_name not in groups:
                groups[class_name] = {'count': 0, 'min_distance': float('inf')}
            
            groups[class_name]['count'] += 1
            if dist > 0:
                groups[class_name]['min_distance'] = min(groups[class_name]['min_distance'], dist)
        
        # Build description for each object class
        for object_name, info in groups.items():
            count = info['count']
            min_dist = info['min_distance']
            
            # Use 100 meters if no distance recorded
            has_distance = min_dist != float('inf')
            
            # Check cooldown for this object name
            last_time = self.last_announced.get(f"object_{object_name}", 0)
            
            if current_time - last_time >= self.announcement_cooldown:
                # 1. Announce in Bengali
                if self.language == 'bn':
                    translated = BENGALI_CLASS_NAMES.get(object_name, object_name)
                    
                    if has_distance:
                        # Translate distance number representation (use English/Standard numerals for pyttsx3 support)
                        dist_str = f"{min_dist:.1f}"
                        if count == 1:
                            text = f"{dist_str} মিটার দূরত্বে {translated}"
                        else:
                            text = f"{count}টি {translated}, সবচেয়ে কাছে {dist_str} মিটার দূরত্বে"
                    else:
                        if count == 1:
                            text = f"{translated} দেখা যাচ্ছে"
                        else:
                            text = f"{count}টি {translated} দেখা যাচ্ছে"
                            
                # 2. Announce in English
                else:
                    display_name = object_name
                    # Simple pluralization
                    if count > 1 and not display_name.endswith('s'):
                        display_name += 's'
                        
                    if has_distance:
                        dist_str = f"{min_dist:.1f}"
                        if count == 1:
                            text = f"{display_name} at {dist_str} meters"
                        else:
                            text = f"{count} {display_name}, closest at {dist_str} meters"
                    else:
                        if count == 1:
                            text = f"{display_name} detected"
                        else:
                            text = f"{count} {display_name} detected"
                
                announcements.append(text)
                self.last_announced[f"object_{object_name}"] = current_time
        
        return announcements
    
    def _combine_announcements(self, announcements: List[str]) -> str:
        """
        Combine multiple announcements into a single sentence.
        
        Args:
            announcements: List of announcement strings
            
        Returns:
            Combined announcement text
        """
        if not announcements:
            return ""
        
        if len(announcements) == 1:
            return announcements[0]
        
        phrases = self.phrases[self.language]
        
        # Combine with 'and'
        if len(announcements) == 2:
            return f"{announcements[0]} {phrases['and']} {announcements[1]}"
        else:
            # For more than 2, use commas and 'and' for the last one
            return ", ".join(announcements[:-1]) + f" {phrases['and']} " + announcements[-1]
    def announce_ocr(self, ocr_results: List[Dict]):
        """
        Announce recognized text from OCR results with an extended cooldown to prevent overlap.
        
        Args:
            ocr_results: List of OCR detection dictionaries
        """
        if not self.enabled or not ocr_results:
            return
            
        current_time = time.time()
        
        # Extract unique text strings
        texts = []
        for res in ocr_results:
            txt = res.get('text', '').strip()
            if txt and txt not in texts:
                texts.append(txt)
                
        if not texts:
            return
            
        combined_text = ", ".join(texts)
        
        # Check cooldown (minimum 6 seconds for OCR readings to avoid voice clutter)
        last_time = self.last_announced.get("ocr_reading", 0)
        if current_time - last_time >= max(6.0, self.announcement_cooldown * 2.5):
            if self.language == 'bn':
                prefix = "লেখা রয়েছে"
            else:
                prefix = "I read"
                
            announcement = f"{prefix}: {combined_text}"
            # Empty the queue first to discard old backlog and ensure real-time responsiveness
            while self.announcement_queue.qsize() > 0:
                try:
                    self.announcement_queue.get_nowait()
                except Exception:
                    break
            self.announcement_queue.put(announcement)
            self.last_announced["ocr_reading"] = current_time

    def announce_custom(self, text: str):
        """
        Announce custom text.
        
        Args:
            text: Custom text to announce
        """
        if self.enabled:
            self.announcement_queue.put(text)
    
    def set_language(self, language: str):
        """
        Set announcement language.
        
        Args:
            language: Language code ('en' or 'bn')
        """
        if language in self.phrases:
            self.language = language
            self._update_voice_for_language()
            print(f"🌐 Voice language set to: {language}")
    
    def set_speech_rate(self, rate: int):
        """
        Set TTS speech rate.
        
        Args:
            rate: Speech rate (words per minute)
        """
        if self.tts_engine:
            try:
                self.tts_engine.setProperty('rate', rate)
                print(f"⚡ Speech rate set to: {rate} WPM")
            except Exception as e:
                print(f"Error setting speech rate: {e}")
    
    def set_volume(self, volume: float):
        """
        Set TTS volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.tts_engine:
            try:
                volume = max(0.0, min(1.0, volume))
                self.tts_engine.setProperty('volume', volume)
                print(f"🔊 Volume set to: {volume}")
            except Exception as e:
                print(f"Error setting volume: {e}")
    
    def get_status(self) -> Dict:
        """Get current voice announcer status."""
        return {
            'enabled': self.enabled,
            'language': self.language,
            'tts_available': self.tts_engine is not None,
            'queue_size': self.announcement_queue.qsize() if hasattr(self.announcement_queue, 'qsize') else 0,
            'thread_active': self.announcement_thread is not None and self.announcement_thread.is_alive()
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_announcement_thread()