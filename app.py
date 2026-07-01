"""
Main Streamlit Application for Face and Object Detection System with Voice Announcements.
"""

import streamlit as st
import cv2
import numpy as np
import time
from detection.face_detector import FaceDetector
from detection.object_detector import ObjectDetector
from detection.ocr_detector import OcrDetector
from utils.camera import StreamlitVideoProcessor


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Face & Object Detection with Voice",
        page_icon="🎯🔊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("🎯🔊 AI Detection System with Voice Announcements")
    st.markdown("Built with Streamlit, OpenCV, YOLO v8, and Text-to-Speech")
    
    # Initialize session state
    if 'detectors_loaded' not in st.session_state:
        st.session_state.detectors_loaded = False
        st.session_state.face_detector = None
        st.session_state.object_detector = None
        st.session_state.ocr_detector = None
        st.session_state.video_processor = None
        st.session_state.camera_started = False
        st.session_state.voice_enabled = False
        st.session_state.face_recognition_enabled = False
        st.session_state.ocr_enabled = False
        st.session_state.run_video = False
        st.session_state.ollama_enabled = False
        st.session_state.ollama_url = "http://localhost:11434"
        st.session_state.ollama_model = "llama3.2"
        st.session_state.ollama_personality = "Friendly"
        st.session_state.ollama_vision_enabled = False
        st.session_state.gemini_enabled = False
        st.session_state.gemini_api_key = ""
        st.session_state.gemini_model = "gemini-3.1-flash-lite"
        st.session_state.expression_enabled = True
        st.session_state.gesture_enabled = True
        st.session_state.yolo_imgsz = 960
        st.session_state.camera_source = "Local Webcam"
        st.session_state.rpi_stream_url = "http://192.168.1.100:5000/video_feed"
        st.session_state.audio_device = "System Default"
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Detection Controls")
        
        # Detection toggles
        enable_face_detection = st.checkbox("Enable Face Detection", value=True)
        enable_object_detection = st.checkbox("Enable Object Detection", value=True)
        enable_ocr = st.checkbox("Enable OCR (Text Reading)", value=st.session_state.ocr_enabled, help="Read written text in English and Bengali")
        
        # Face recognition toggle
        enable_face_recognition = st.checkbox(
            "Enable Face Recognition", 
            value=st.session_state.face_recognition_enabled,
            help="Recognize known people from the known_faces folder"
        )
        
        # Expression and Gesture detection toggles
        enable_expression = st.checkbox(
            "Enable Expression Detection", 
            value=st.session_state.expression_enabled,
            help="Detect facial expressions (Happy, Sad, Angry, Surprised)"
        )
        enable_gesture = st.checkbox(
            "Enable Sign Language", 
            value=st.session_state.gesture_enabled,
            help="Detect sign language gestures and hand signals"
        )
        
        # Save to session state
        st.session_state.expression_enabled = enable_expression
        st.session_state.gesture_enabled = enable_gesture
        
        st.divider()
        
        # Voice Controls Section
        st.header("🔊 Voice Announcements")
        
        enable_voice = st.checkbox(
            "Enable Voice Mode", 
            value=st.session_state.voice_enabled,
            help="Turn on voice announcements for detections"
        )
        
        # Initialize default values in case voice is disabled
        ollama_enabled = st.session_state.ollama_enabled
        ollama_url = st.session_state.ollama_url
        ollama_model = st.session_state.ollama_model
        ollama_personality = st.session_state.ollama_personality
        ollama_vision_enabled = st.session_state.ollama_vision_enabled
        gemini_enabled = st.session_state.gemini_enabled
        gemini_api_key = st.session_state.gemini_api_key
        gemini_model = st.session_state.gemini_model
        
        if enable_voice:
            # Query audio output devices dynamically
            audio_devices = ["System Default"]
            try:
                import sounddevice as sd
                for idx, d in enumerate(sd.query_devices()):
                    if d['max_output_channels'] > 0:
                        audio_devices.append(f"{idx}: {d['name']}")
            except Exception:
                pass
            
            saved_device = st.session_state.get('audio_device', 'System Default')
            default_device_idx = 0
            if saved_device in audio_devices:
                default_device_idx = audio_devices.index(saved_device)
                
            selected_audio_device = st.selectbox(
                "Audio Output Device",
                options=audio_devices,
                index=default_device_idx,
                help="Select where to route the voice announcements (e.g., Bluetooth earbuds)"
            )
            st.session_state.audio_device = selected_audio_device
            
            # Voice language selection
            voice_language = st.selectbox(
                "Voice Language",
                options=['en', 'bn'],
                format_func=lambda x: {'en': 'English', 'bn': 'Bengali'}[x],
                index=0,
                help="Select announcement language"
            )
            
            st.divider()
            st.subheader("🧠 Ollama AI Responses")
            
            ollama_enabled = st.checkbox(
                "Enable Ollama AI", 
                value=st.session_state.ollama_enabled,
                help="Use local self-hosted Ollama LLM to generate conversational announcements"
            )
            
            if ollama_enabled:
                ollama_url = st.text_input(
                    "Ollama Server URL", 
                    value=st.session_state.ollama_url,
                    help="Enter your local or remote Ollama server API URL"
                )
                
                # Try to fetch available models
                models = []
                connected = False
                try:
                    import requests
                    r = requests.get(f"{ollama_url}/api/tags", timeout=1.0)
                    if r.status_code == 200:
                        data = r.json()
                        models = [m['name'] for m in data.get('models', [])]
                        connected = True
                except Exception:
                    pass
                
                if connected:
                    st.success("🟢 Connected to Ollama!")
                    if models:
                        default_idx = 0
                        saved_model = st.session_state.ollama_model
                        for idx, m in enumerate(models):
                            if saved_model in m or m in saved_model:
                                default_idx = idx
                                break
                        ollama_model = st.selectbox("Select AI Model", options=models, index=default_idx)
                    else:
                        st.warning("⚠️ Connected, but no models found. Pull one via CLI (e.g. `ollama pull llama3.2`).")
                        ollama_model = st.text_input("AI Model", value=st.session_state.ollama_model)
                else:
                    st.info("ℹ️ Ollama server offline. Run `ollama serve` or check service status.")
                    ollama_model = st.text_input("AI Model", value=st.session_state.ollama_model)
                
                personalities = ["Friendly", "Precise Guide", "Sarcastic Robot", "Excited Companion"]
                try:
                    default_pers_idx = personalities.index(st.session_state.ollama_personality)
                except ValueError:
                    default_pers_idx = 0
                    
                ollama_personality = st.selectbox(
                    "AI Personality",
                    options=personalities,
                    index=default_pers_idx
                )
                
                ollama_vision_enabled = st.checkbox(
                    "Enable Local Vision Model (Option A)",
                    value=st.session_state.get('ollama_vision_enabled', False),
                    help="Enable image-based context. You MUST select or type a vision-capable model (like 'llava', 'moondream' or 'minicpm-v') above. Note: Vision inference is slower than text-only."
                )
            
            st.divider()
            st.subheader("♊ Google Gemini AI (Cloud)")
            
            gemini_enabled = st.checkbox(
                "Enable Gemini AI (Cloud)",
                value=st.session_state.get('gemini_enabled', False),
                help="Use Google Gemini API for cloud-based multimodal scene descriptions"
            )
            
            if gemini_enabled:
                import os
                default_key = os.environ.get("GEMINI_API_KEY", "")
                saved_key = st.session_state.get('gemini_api_key', default_key)
                
                gemini_api_key = st.text_input(
                    "Gemini API Key",
                    value=saved_key,
                    type="password",
                    help="Enter your API Key from Google AI Studio. Stored strictly in local memory."
                )
                
                gemini_model = st.selectbox(
                    "Select Gemini Model",
                    options=["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                    index=0,
                    help="Gemini 3.1 Flash Lite is recommended (works best with free-tier quotas)."
                )
            
            # Sync settings directly with initialized announcer
            if st.session_state.video_processor and hasattr(st.session_state.video_processor, 'voice_announcer'):
                st.session_state.video_processor.voice_announcer.ollama_enabled = ollama_enabled
                st.session_state.video_processor.voice_announcer.ollama_url = ollama_url
                st.session_state.video_processor.voice_announcer.ollama_model = ollama_model
                st.session_state.video_processor.voice_announcer.ollama_personality = ollama_personality
                st.session_state.video_processor.voice_announcer.ollama_vision_enabled = ollama_vision_enabled
                st.session_state.video_processor.voice_announcer.gemini_enabled = gemini_enabled
                st.session_state.video_processor.voice_announcer.gemini_api_key = gemini_api_key
                st.session_state.video_processor.voice_announcer.gemini_model = gemini_model
                
                # Parse device index
                device_idx = None
                if selected_audio_device != "System Default":
                    try:
                        device_idx = int(selected_audio_device.split(":")[0])
                    except Exception:
                        pass
                st.session_state.video_processor.voice_announcer.selected_device_index = device_idx
        else:
            voice_language = 'en'
            
        # Update session state values
        st.session_state.ollama_enabled = ollama_enabled
        st.session_state.ollama_url = ollama_url
        st.session_state.ollama_model = ollama_model
        st.session_state.ollama_personality = ollama_personality
        st.session_state.ollama_vision_enabled = ollama_vision_enabled
        st.session_state.gemini_enabled = gemini_enabled
        st.session_state.gemini_api_key = gemini_api_key
        st.session_state.gemini_model = gemini_model
        
        st.divider()
        
        # Object Detection Settings
        st.header("⚙️ Detection Settings")
        
        yolo_model_choice = st.selectbox(
            "AI Model Range (YOLOv8)",
            options=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
            format_func=lambda x: {
                'yolov8n.pt': 'YOLOv8 Nano (Fast)',
                'yolov8s.pt': 'YOLOv8 Small (Accurate)',
                'yolov8m.pt': 'YOLOv8 Medium (High Accuracy)',
                'yolov8l.pt': 'YOLOv8 Large (Maximum Accuracy)'
            }[x],
            index=0,
            help="Larger models have significantly higher accuracy but require more processing."
        )
        
        yolo_imgsz = st.selectbox(
            "YOLO Inference Resolution",
            options=[640, 960, 1280],
            format_func=lambda x: {
                640: '640px (Fast / Standard)',
                960: '960px (High Detail / Recommended)',
                1280: '1280px (Ultra Detail / Small Objects)'
            }[x],
            index=1,
            help="Higher resolutions preserve fine details for small or held objects."
        )
        st.session_state.yolo_imgsz = yolo_imgsz
        
        confidence_threshold = st.slider(
            "Object Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Increase confidence to eliminate false positives; lower to detect far-away objects."
        )
        
        st.divider()
        
        # Camera source settings
        st.header("📹 Camera Source")
        camera_source = st.selectbox(
            "Camera Input Source",
            options=["Local Webcam", "Raspberry Pi (WiFi Stream)"],
            index=0 if st.session_state.get('camera_source', 'Local Webcam') == 'Local Webcam' else 1,
            help="Choose between your local webcam or a remote Raspberry Pi stream over WiFi"
        )
        st.session_state.camera_source = camera_source
        
        rpi_url = st.session_state.get('rpi_stream_url', 'http://192.168.1.100:5000/video_feed')
        if camera_source == "Raspberry Pi (WiFi Stream)":
            rpi_url = st.text_input(
                "Raspberry Pi Stream URL",
                value=rpi_url,
                help="Enter the URL of your Raspberry Pi MJPEG or RTSP video stream"
            )
            st.session_state.rpi_stream_url = rpi_url
            st.caption("Example: http://192.168.1.100:5000/video_feed. Use the Raspberry Pi IP address, not localhost.")
            if "localhost" in rpi_url.lower() or "127.0.0.1" in rpi_url:
                st.warning("localhost points to this laptop. For a Raspberry Pi camera, use the Pi's LAN IP address.")

        # Camera controls
        st.header("📹 Camera Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶️ Start", type="primary", use_container_width=True)
        with col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        
        # Initialize system button
        init_btn = st.button("🚀 Initialize System", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📺 Live Video Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
    with col2:
        st.header("📊 Detection Statistics")
        
        # Statistics placeholders
        face_stats = st.empty()
        object_stats = st.empty()
        fps_stats = st.empty()
        
        st.divider()
        
        # OCR Read Text Panel
        st.subheader("📝 Read Text (OCR)")
        ocr_placeholder = st.empty()
        
        st.divider()
        
        # System information
        st.subheader("ℹ️ System Status")
        detector_status = st.empty()
    
    # Handle Initialize System
    if init_btn:
        with st.spinner("Loading AI detection models..."):
            try:
                # Initialize face detector
                face_detector = FaceDetector(enable_recognition=enable_face_recognition)
                
                # Initialize object detector  
                object_detector = ObjectDetector(model_path=yolo_model_choice, confidence_threshold=confidence_threshold, imgsz=st.session_state.yolo_imgsz)
                
                # Initialize OCR only when requested; EasyOCR downloads/models are heavy.
                ocr_detector = OcrDetector() if enable_ocr else None
                
                # Initialize video processor
                video_processor = StreamlitVideoProcessor()
                
                # Sync initial voice announcer Ollama settings
                if hasattr(video_processor, 'voice_announcer'):
                    video_processor.voice_announcer.ollama_enabled = st.session_state.ollama_enabled
                    video_processor.voice_announcer.ollama_url = st.session_state.ollama_url
                    video_processor.voice_announcer.ollama_model = st.session_state.ollama_model
                    video_processor.voice_announcer.ollama_personality = st.session_state.ollama_personality
                    video_processor.voice_announcer.ollama_vision_enabled = st.session_state.ollama_vision_enabled
                    video_processor.voice_announcer.gemini_enabled = st.session_state.gemini_enabled
                    video_processor.voice_announcer.gemini_api_key = st.session_state.gemini_api_key
                    video_processor.voice_announcer.gemini_model = st.session_state.gemini_model
                
                st.session_state.face_detector = face_detector
                st.session_state.object_detector = object_detector
                st.session_state.ocr_detector = ocr_detector
                st.session_state.video_processor = video_processor
                st.session_state.detectors_loaded = True
                
                st.success("✅ AI detection system loaded!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error loading system: {e}")
    
    # Handle Start Camera
    if start_btn:
        if not st.session_state.detectors_loaded:
            status_placeholder.warning("⚠️ Please click 'Initialize System' first!")
        else:
            st.session_state.run_video = True
            st.session_state.camera_started = True
    
    # Handle Stop Camera
    if stop_btn:
        st.session_state.run_video = False
        st.session_state.camera_started = False
        if st.session_state.video_processor:
            st.session_state.video_processor.stop_video_stream()
        status_placeholder.info("📷 Camera stopped")
    
    # Update detector status display
    if st.session_state.detectors_loaded:
        detector_status.success("✅ Detection System Ready")
        if st.session_state.object_detector:
            st.session_state.object_detector.set_confidence_threshold(confidence_threshold)
            st.session_state.object_detector.set_imgsz(st.session_state.yolo_imgsz)
    else:
        detector_status.warning("⚠️ Click 'Initialize System' to load models")
    
    # Main video processing loop
    if st.session_state.run_video and st.session_state.detectors_loaded:
        run_video_stream(
            video_placeholder,
            status_placeholder,
            face_stats,
            object_stats,
            fps_stats,
            ocr_placeholder,
            enable_face_detection,
            enable_object_detection,
            enable_ocr,
            enable_voice,
            voice_language,
            confidence_threshold,
            enable_expression,
            enable_gesture
        )
    else:
        if not st.session_state.camera_started:
            video_placeholder.info("👆 Click 'Start' to begin video detection")


def run_video_stream(video_placeholder, status_placeholder, face_stats, object_stats, 
                    fps_stats, ocr_placeholder, enable_face, enable_objects, enable_ocr,
                    enable_voice, voice_language, confidence_threshold,
                    enable_expression=True, enable_gesture=True):
    """Run the video stream with continuous updates."""
    
    if enable_ocr and st.session_state.ocr_detector is None:
        with st.spinner("Loading OCR models..."):
            st.session_state.ocr_detector = OcrDetector()
    
    # Set camera source dynamically before opening stream
    if st.session_state.get('camera_source', 'Local Webcam') == "Raspberry Pi (WiFi Stream)":
        st.session_state.video_processor.camera.camera_id = st.session_state.get('rpi_stream_url', '')
    else:
        st.session_state.video_processor.camera.camera_id = 0

    # Start camera if not already started
    if not st.session_state.video_processor.camera.is_opened:
        status_placeholder.info("🔄 Starting camera...")
        show_diag = (st.session_state.get('camera_source', 'Local Webcam') == 'Local Webcam')
        if not st.session_state.video_processor.start_video_stream(show_diagnostics=show_diag):
            status_placeholder.error("❌ Failed to start camera. Check permissions or stream URL.")
            return
    
    # Configure voice if enabled
    if enable_voice:
        # Sync initial voice announcer Ollama settings
        if hasattr(st.session_state.video_processor, 'voice_announcer'):
            st.session_state.video_processor.voice_announcer.ollama_enabled = st.session_state.ollama_enabled
            st.session_state.video_processor.voice_announcer.ollama_url = st.session_state.ollama_url
            st.session_state.video_processor.voice_announcer.ollama_model = st.session_state.ollama_model
            st.session_state.video_processor.voice_announcer.ollama_personality = st.session_state.ollama_personality
            st.session_state.video_processor.voice_announcer.ollama_vision_enabled = st.session_state.ollama_vision_enabled
            st.session_state.video_processor.voice_announcer.gemini_enabled = st.session_state.gemini_enabled
            st.session_state.video_processor.voice_announcer.gemini_api_key = st.session_state.gemini_api_key
            st.session_state.video_processor.voice_announcer.gemini_model = st.session_state.gemini_model
        st.session_state.video_processor.enable_voice_mode(True, voice_language)
    
    status_placeholder.success("✅ Camera active - streaming...")
    
    # Create a stop button in the main area
    stop_stream = st.button("🛑 Stop Stream", key="stop_stream_btn")
    
    if stop_stream:
        st.session_state.run_video = False
        st.session_state.video_processor.stop_video_stream()
        st.rerun()
    
    # Video streaming loop
    frame_count = 0
    
    while st.session_state.run_video:
        frame_start_time = time.time()
        try:
            # Sync session state values inside the loop to ensure they are updated in real-time
            if hasattr(st.session_state.video_processor, 'voice_announcer'):
                st.session_state.video_processor.voice_announcer.ollama_enabled = st.session_state.ollama_enabled
                st.session_state.video_processor.voice_announcer.ollama_url = st.session_state.ollama_url
                st.session_state.video_processor.voice_announcer.ollama_model = st.session_state.ollama_model
                st.session_state.video_processor.voice_announcer.ollama_personality = st.session_state.ollama_personality
                st.session_state.video_processor.voice_announcer.ollama_vision_enabled = st.session_state.ollama_vision_enabled
                st.session_state.video_processor.voice_announcer.gemini_enabled = st.session_state.gemini_enabled
                st.session_state.video_processor.voice_announcer.gemini_api_key = st.session_state.gemini_api_key
                st.session_state.video_processor.voice_announcer.gemini_model = st.session_state.gemini_model
                
            # Read frame from camera
            success, frame = st.session_state.video_processor.camera.read_frame()
            
            if not success or frame is None:
                status_placeholder.warning("⚠️ Frame read failed, retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Process frame
            try:
                processed_frame, face_count, object_count, ocr_count, fps, recognition_stats, ocr_results = \
                    st.session_state.video_processor.process_frame_pipeline(
                        frame,
                        st.session_state.face_detector if enable_face else None,
                        st.session_state.object_detector if enable_objects else None,
                        st.session_state.ocr_detector if enable_ocr else None,
                        enable_face,
                        enable_objects,
                        enable_ocr,
                        enable_voice,
                        enable_expression,
                        enable_gesture
                    )
                
                # Display the frame (Resized to 960x540 to dramatically speed up Streamlit's WebSocket serialization and rendering)
                h_f, w_f = processed_frame.shape[:2]
                if w_f > 960:
                    display_frame = cv2.resize(processed_frame, (960, int(h_f * 960 / w_f)))
                else:
                    display_frame = processed_frame
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
                # Update statistics
                face_stats.metric("👤 Faces", face_count)
                object_stats.metric("🎯 Objects & Text", object_count + ocr_count)
                fps_stats.metric("⚡ FPS", f"{fps:.1f}")
                
                # Display OCR results in real-time in the sidebar/dashboard
                if enable_ocr and ocr_results:
                    ocr_html = "<div style='background-color:#1E1E2E; padding:12px; border-radius:8px; border-left: 4px solid #8200FA; max-height:220px; overflow-y:auto; font-family:sans-serif; color:#E0E0E0; font-size: 0.95em; line-height: 1.4;'>"
                    for res in ocr_results:
                        ocr_html += f"<div style='margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #2D2D3E;'>➡️ <b>{res['text']}</b> <span style='color:#A0A0A0; font-size:0.8em;'>({res['confidence']:.2f})</span></div>"
                    ocr_html += "</div>"
                    ocr_placeholder.markdown(ocr_html, unsafe_allow_html=True)
                else:
                    ocr_placeholder.info("No text detected. Try holding a book or page in front of the camera.")
                
            except Exception as proc_error:
                # Show raw frame if processing fails
                raw_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(raw_frame_rgb, channels="RGB", use_container_width=True)
                status_placeholder.warning(f"⚠️ Processing issue: {proc_error}")
            
            # Calculate dynamic delay to maintain target 30 FPS (33ms total frame time)
            elapsed = time.time() - frame_start_time
            delay = max(0.001, 0.033 - elapsed)
            time.sleep(delay)
            
        except Exception as e:
            status_placeholder.error(f"❌ Stream error: {e}")
            break
    
    # Cleanup when loop ends
    if st.session_state.video_processor:
        st.session_state.video_processor.stop_video_stream()
    status_placeholder.info("📷 Stream ended")


if __name__ == "__main__":
    main()
