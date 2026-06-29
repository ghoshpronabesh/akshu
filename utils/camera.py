"""
Camera utilities for video capture and frame processing with enhanced error handling.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List
import streamlit as st
from .voice_announcer import VoiceAnnouncer


class CameraCapture:
    """Handles camera capture and frame processing with robust error handling."""
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.last_successful_frame = None
        
    def detect_available_cameras(self) -> List[int]:
        """
        Detect available camera devices.
        
        Returns:
            List of available camera IDs
        """
        available_cameras = []
        
        # Test camera IDs 0-4
        for camera_id in range(5):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(camera_id)
                    print(f"✅ Found camera at ID {camera_id}")
                cap.release()
            else:
                print(f"❌ No camera at ID {camera_id}")
        
        return available_cameras
    
    def start_capture(self, try_all_cameras: bool = True) -> bool:
        """
        Start video capture from camera with fallback options.
        
        Args:
            try_all_cameras: Whether to try other camera IDs if the primary fails
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            # First try the specified camera ID
            if self._try_camera(self.camera_id):
                return True
            
            # If that fails and try_all_cameras is True, try other IDs (only for local webcam IDs)
            if try_all_cameras and not isinstance(self.camera_id, str):
                st.warning(f"Camera ID {self.camera_id} failed, trying other cameras...")
                
                available_cameras = self.detect_available_cameras()
                
                if not available_cameras:
                    st.error("No cameras detected on your system")
                    return False
                
                for camera_id in available_cameras:
                    if camera_id != self.camera_id:
                        st.info(f"Trying camera ID {camera_id}...")
                        if self._try_camera(camera_id):
                            self.camera_id = camera_id
                            st.success(f"Successfully connected to camera ID {camera_id}")
                            return True
            
            return False
            
        except Exception as e:
            st.error(f"Camera initialization error: {e}")
            return False
    
    def _try_camera(self, camera_id) -> bool:
        """
        Try to open a specific camera ID or network stream URL.
        
        Args:
            camera_id: Camera device ID (int) or network stream URL (str) to try
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Release any existing camera
            if self.cap is not None:
                self.cap.release()
            
            # If camera_id is a string, it represents a network stream URL (WiFi camera from Raspberry Pi)
            if isinstance(camera_id, str):
                print(f"Connecting to WiFi Stream: {camera_id}")
                try:
                    self.cap = cv2.VideoCapture(camera_id)
                    if self.cap.isOpened():
                        # Test if we can read a frame
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.is_opened = True
                            self.frame_count = 0
                            print(f"✅ WiFi Stream {camera_id} opened successfully!")
                            return True
                        else:
                            self.cap.release()
                except Exception as stream_err:
                    print(f"WiFi Stream connection failed: {stream_err}")
                return False
            
            # Try different camera backends for local cameras
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(camera_id, backend)
                    
                    if self.cap.isOpened():
                        # Test if we can actually read frames
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            # Set camera properties for better performance (720p HD for Apple Silicon long range)
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            
                            # Set buffer size to reduce latency
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            self.is_opened = True
                            self.frame_count = 0
                            print(f"✅ Camera {camera_id} opened with backend {backend}")
                            return True
                        else:
                            self.cap.release()
                    
                except Exception as backend_error:
                    print(f"Backend {backend} failed: {backend_error}")
                    if self.cap:
                        self.cap.release()
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error trying camera {camera_id}: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera with retry logic.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            for attempt in range(3):  # Try up to 3 times
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Validate frame
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        self.frame_count += 1
                        self.last_successful_frame = frame.copy()
                        return True, frame
                
                # If frame read failed, try to restart camera
                if attempt < 2:  # Don't restart on the last attempt
                    print(f"Frame read failed, attempt {attempt + 1}/3")
                    time.sleep(0.1)
            
            # If all attempts failed, return last successful frame if available
            if self.last_successful_frame is not None:
                st.warning("Camera unstable, using last successful frame")
                return True, self.last_successful_frame
            
            return False, None
            
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def restart_camera(self) -> bool:
        """
        Restart the camera connection.
        
        Returns:
            True if restart successful, False otherwise
        """
        print("Attempting to restart camera...")
        self.stop_capture()
        time.sleep(1)  # Give the camera time to release
        return self.start_capture()
    
    def get_camera_info(self) -> dict:
        """Get camera information and properties."""
        if not self.is_opened or self.cap is None:
            return {"status": "Not connected"}
        
        try:
            info = {
                "status": "Connected",
                "camera_id": self.camera_id,
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
                "frame_count": self.frame_count,
                "backend": self.cap.getBackendName() if hasattr(self.cap, 'getBackendName') else "Unknown"
            }
            return info
        except Exception as e:
            return {"status": f"Error getting info: {e}"}
    
    def stop_capture(self):
        """Stop video capture and release camera."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            self.cap = None
            print("Camera released")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_capture()


class FrameProcessor:
    """Handles frame processing and performance monitoring."""
    
    def __init__(self):
        """Initialize frame processor."""
        self.frame_times = []
        self.max_frame_history = 30  # Keep last 30 frame times for FPS calculation
        
    def calculate_fps(self) -> float:
        """
        Calculate current FPS based on recent frame processing times.
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times = self.frame_times[-self.max_frame_history:]
        
        # Calculate FPS
        if len(self.frame_times) >= 2:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                fps = (len(self.frame_times) - 1) / time_diff
                return fps
        
        return 0.0
    
    def resize_frame(self, frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Resize frame to specified dimensions.
        
        Args:
            frame: Input frame
            width: Target width
            height: Target height
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, (width, height))
    
    def add_fps_overlay(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Add FPS overlay to frame.
        
        Args:
            frame: Input frame
            fps: Current FPS value
            
        Returns:
            Frame with FPS overlay
        """
        result_frame = frame.copy()
        
        # Add FPS text
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            result_frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return result_frame
    
    def add_detection_stats(self, frame: np.ndarray, faces: int, objects: int, 
                           voice_status: str = "", camera_info: str = "") -> np.ndarray:
        """
        Add detection statistics overlay to frame.
        
        Args:
            frame: Input frame
            faces: Number of detected faces
            objects: Number of detected objects
            voice_status: Voice mode status
            camera_info: Camera information
            
        Returns:
            Frame with detection stats overlay
        """
        result_frame = frame.copy()
        
        # Add detection stats
        stats_text = f"Faces: {faces} | Objects: {objects}"
        cv2.putText(
            result_frame,
            stats_text,
            (10, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Add voice status if provided
        if voice_status:
            cv2.putText(
                result_frame,
                voice_status,
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
        
        # Add camera info if provided
        if camera_info:
            cv2.putText(
                result_frame,
                camera_info,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
        
        return result_frame
    
    def convert_frame_for_streamlit(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame from BGR to RGB for Streamlit display.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Frame in RGB format
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class StreamlitVideoProcessor:
    """Integrates camera processing with Streamlit interface and voice announcements."""
    
    def __init__(self):
        """Initialize Streamlit video processor with voice capabilities."""
        self.camera = CameraCapture()
        self.processor = FrameProcessor()
        self.voice_announcer = VoiceAnnouncer()
        self.camera_restart_attempts = 0
        self.max_restart_attempts = 3
        self.frame_counter = 0
        self.last_ocr_results = []
        
        # Initialize gesture and expression detectors safely
        try:
            from detection.gesture_detector import GestureDetector
            from detection.expression_detector import ExpressionDetector
            self.gesture_detector = GestureDetector()
            self.expression_detector = ExpressionDetector()
            print("✅ Hand Gesture & Face Expression detectors loaded")
        except Exception as ex_det:
            print(f"⚠️ Failed to load MediaPipe detectors: {ex_det}")
            self.gesture_detector = None
            self.expression_detector = None
        
    def diagnose_camera_issues(self):
        """Diagnose and display camera issues."""
        st.subheader("🔍 Camera Diagnostics")
        
        with st.expander("Camera Information", expanded=True):
            available_cameras = self.camera.detect_available_cameras()
            
            if available_cameras:
                st.success(f"Found {len(available_cameras)} available camera(s): {available_cameras}")
                
                for cam_id in available_cameras:
                    with st.container():
                        st.write(f"**Camera {cam_id}:**")
                        
                        # Test this camera
                        test_cap = cv2.VideoCapture(cam_id)
                        if test_cap.isOpened():
                            ret, frame = test_cap.read()
                            if ret and frame is not None:
                                width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = int(test_cap.get(cv2.CAP_PROP_FPS))
                                st.write(f"  - Resolution: {width}x{height}")
                                st.write(f"  - FPS: {fps}")
                                st.write(f"  - Status: ✅ Working")
                            else:
                                st.write(f"  - Status: ❌ Cannot read frames")
                        test_cap.release()
            else:
                st.error("No cameras detected")
                
                st.subheader("🛠️ Troubleshooting Steps")
                st.markdown("""
                1. **Check camera permissions:**
                   - macOS: System Preferences > Security & Privacy > Camera
                   - Windows: Settings > Privacy > Camera
                   - Linux: Check if user is in video group
                
                2. **Check if camera is in use:**
                   - Close other applications that might be using the camera
                   - Restart your browser/Streamlit app
                
                3. **Try different browsers:**
                   - Chrome, Firefox, Safari, Edge
                
                4. **System checks:**
                   - Restart your computer
                   - Update camera drivers
                   - Check USB connections (for external cameras)
                """)
    
    def start_video_stream(self, show_diagnostics: bool = True):
        """Start the video stream for Streamlit with enhanced error handling."""
        if show_diagnostics:
            self.diagnose_camera_issues()
        
        success = self.camera.start_capture(try_all_cameras=True)
        
        if not success:
            st.error("Could not access any camera. Please check camera permissions and connections.")
            return False
        
        # Display successful connection info
        camera_info = self.camera.get_camera_info()
        st.success(f"📹 Camera {camera_info.get('camera_id')} connected successfully!")
        st.info(f"Resolution: {camera_info.get('width')}x{camera_info.get('height')}, FPS: {camera_info.get('fps')}")
        
        return True
    
    def stop_video_stream(self):
        """Stop the video stream."""
        self.camera.stop_capture()
        self.camera_restart_attempts = 0
        if hasattr(self, 'voice_announcer'):
            self.voice_announcer.enable_voice(False)  # Disable voice mode and stop worker thread instantly!
    
    def enable_voice_mode(self, enable: bool, language: str = 'en'):
        """
        Enable or disable voice announcements.
        
        Args:
            enable: Whether to enable voice mode
            language: Language for announcements ('en' or 'bn')
        """
        self.voice_announcer.enable_voice(enable)
        self.voice_announcer.set_language(language)
        
        if enable:
            self.voice_announcer.announce_custom("Voice mode enabled")
        
    def configure_voice_settings(self, speech_rate: int = 150, volume: float = 0.8):
        """
        Configure voice announcement settings.
        
        Args:
            speech_rate: Speech rate in words per minute
            volume: Volume level (0.0 to 1.0)
        """
        self.voice_announcer.set_speech_rate(speech_rate)
        self.voice_announcer.set_volume(volume)
    
    def process_frame_pipeline(self, frame: np.ndarray, face_detector=None, 
                             object_detector=None, ocr_detector=None,
                             enable_face=True, enable_objects=True, enable_ocr=False,
                             enable_voice=False, enable_expression=False, enable_gesture=False) -> Tuple[np.ndarray, int, int, int, float, dict, list]:
        """
        Complete frame processing pipeline with face detection, object detection, distance estimation,
        OCR text reading, voice announcements, and error recovery.
        
        Args:
            frame: Input frame
            face_detector: Face detection instance
            object_detector: Object detection instance
            ocr_detector: OCR detection instance
            enable_face: Whether to perform face detection
            enable_objects: Whether to perform object detection
            enable_ocr: Whether to perform OCR text reading
            enable_voice: Whether to use voice announcements
            enable_expression: Whether to perform facial expression detection
            enable_gesture: Whether to perform sign language gesture detection
            
        Returns:
            Tuple of (processed_frame, face_count, object_count, ocr_count, fps, recognition_stats, ocr_results)
        """
        processed_frame = frame.copy()
        face_count = 0
        object_count = 0
        ocr_count = 0
        face_results = []
        object_results = []
        ocr_results = []
        gesture_results = []
        expression_results = []
        recognition_stats = {}
        
        # Get active announcer language
        language = self.voice_announcer.language
        
        self.frame_counter += 1
        
        # Dynamically sync the voice announcer enabled state on every frame
        if hasattr(self, 'voice_announcer'):
            if self.voice_announcer.enabled != enable_voice:
                self.voice_announcer.enable_voice(enable_voice)
        
        try:
            # Face detection/recognition
            if enable_face and face_detector is not None:
                if hasattr(face_detector, 'process_frame') and callable(face_detector.process_frame):
                    # New face detector with recognition
                    processed_frame, face_count, recognition_stats = face_detector.process_frame(processed_frame)
                    
                    # Extract face results for voice announcements from cached results
                    face_results = getattr(face_detector, 'last_results', [])
                else:
                    # Fallback to old interface
                    processed_frame, face_count = face_detector.process_frame(processed_frame)
            
            # Object detection with Distance & Language Translations
            if enable_objects and object_detector is not None:
                if hasattr(object_detector, 'detect_objects'):
                    object_results = object_detector.detect_objects(frame)
                    # Pass the selected language for drawing BBoxes and labels!
                    processed_frame = object_detector.draw_detections(processed_frame, object_results, language=language)
                    object_count = len(object_results)
                else:
                    processed_frame, object_count = object_detector.process_frame(processed_frame)
            
            # OCR Text reading (Optimized with Frame Skipping - run once every 12 frames to maintain smooth 30 FPS feed)
            if enable_ocr and ocr_detector is not None:
                if hasattr(ocr_detector, 'detect_text'):
                    # Perform heavy OCR inference only once every 12 frames (~400ms)
                    if self.frame_counter % 12 == 0 or self.frame_counter <= 1 or not self.last_ocr_results:
                        ocr_results = ocr_detector.detect_text(frame)
                        self.last_ocr_results = ocr_results
                    else:
                        # Draw cached text bounding boxes on intermediate frames
                        ocr_results = self.last_ocr_results
                        
                    processed_frame = ocr_detector.draw_detections(processed_frame, ocr_results)
                    ocr_count = len(ocr_results)
                    
            # Facial expression detection
            if enable_expression and getattr(self, 'expression_detector', None) is not None:
                expression_results = self.expression_detector.detect_expressions(frame)
                processed_frame = self.expression_detector.draw_detections(processed_frame, expression_results)
                
            # Hand gesture / Sign language detection
            if enable_gesture and getattr(self, 'gesture_detector', None) is not None:
                gesture_results = self.gesture_detector.detect_gestures(frame)
                processed_frame = self.gesture_detector.draw_detections(processed_frame, gesture_results)
            
            # Voice announcements (for face, object, expression and gesture detections)
            if enable_voice and (face_results or object_results or gesture_results):
                # Pass expression and gesture results to announce_detections
                self.voice_announcer.announce_detections(
                    face_results, 
                    object_results, 
                    expression_results=expression_results, 
                    gesture_results=gesture_results,
                    frame=frame
                )
                
                # Also announce OCR text if available (throttled inside announcer)
                if ocr_results and hasattr(self.voice_announcer, 'announce_ocr'):
                    self.voice_announcer.announce_ocr(ocr_results)
            
        except Exception as e:
            st.error(f"Detection processing error: {e}")
        
        # Calculate FPS
        fps = self.processor.calculate_fps()
        
        # Get status information for overlay
        voice_status = ""
        if enable_voice:
            try:
                voice_info = self.voice_announcer.get_status()
                if voice_info.get('enabled', False):
                    language_code = voice_info.get('language', 'en').upper()
                    voice_status = f"🔊 Voice: {language_code}"
                else:
                    voice_status = "🔇 Voice: OFF"
            except Exception as e:
                print(f"Voice status error: {e}")
                voice_status = "⚠️ Voice: ERR"
        else:
            voice_status = "🔇 Voice: OFF"
        
        # Camera info for overlay
        camera_info = ""
        cam_info = self.camera.get_camera_info()
        if cam_info.get('status') == 'Connected':
            camera_info = f"Cam{cam_info.get('camera_id')}: {cam_info.get('width')}x{cam_info.get('height')}"
        
        # Add overlays
        processed_frame = self.processor.add_fps_overlay(processed_frame, fps)
        processed_frame = self.processor.add_detection_stats(
            processed_frame, 
            face_count, 
            object_count + ocr_count,  # Include OCR count in total objects metric overlay
            voice_status, 
            camera_info
        )
        
        # Convert for Streamlit display
        processed_frame = self.processor.convert_frame_for_streamlit(processed_frame)
        
        return processed_frame, face_count, object_count, ocr_count, fps, recognition_stats, ocr_results
    
    def handle_camera_error(self) -> bool:
        """
        Handle camera errors with restart logic.
        
        Returns:
            True if recovery successful, False otherwise
        """
        if self.camera_restart_attempts < self.max_restart_attempts:
            self.camera_restart_attempts += 1
            st.warning(f"Camera error detected. Attempting restart ({self.camera_restart_attempts}/{self.max_restart_attempts})...")
            
            if self.camera.restart_camera():
                st.success("Camera restarted successfully!")
                self.camera_restart_attempts = 0
                return True
            else:
                st.error(f"Camera restart attempt {self.camera_restart_attempts} failed")
                return False
        else:
            st.error("Maximum camera restart attempts exceeded. Please check your camera manually.")
            return False
    
    def get_voice_status(self) -> dict:
        """Get current voice announcer status."""
        return self.voice_announcer.get_status()
    
    def announce_custom_message(self, message: str):
        """Announce a custom message."""
        self.voice_announcer.announce_custom(message)
    
    def get_camera_info(self) -> dict:
        """Get current camera information."""
        return self.camera.get_camera_info()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_video_stream()
        if hasattr(self, 'voice_announcer'):
            self.voice_announcer.stop_announcement_thread()