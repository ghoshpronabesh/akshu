"""
Face Detection Module using OpenCV Haar Cascades with Recognition capabilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from .face_recognizer import FaceRecognizer


class FaceDetector:
    """Face detection with optional recognition capabilities."""
    
    def __init__(self, scale_factor: float = 1.15, min_neighbors: int = 7, enable_recognition: bool = False):
        """
        Initialize the face detector.
        
        Args:
            scale_factor: Parameter specifying how much the image size is reduced at each scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
            enable_recognition: Whether to enable face recognition
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.enable_recognition = enable_recognition
        
        # Load the pre-trained Haar cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            raise RuntimeError("Could not load face cascade classifier")
        
        # Initialize face recognizer if enabled
        self.face_recognizer = None
        if enable_recognition:
            try:
                self.face_recognizer = FaceRecognizer()
                print("✅ Face recognition enabled")
            except Exception as e:
                print(f"⚠️ Face recognition initialization failed: {e}")
                self.enable_recognition = False
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of tuples containing (x, y, width, height) for each detected face
        """
        h, w = image.shape[:2]
        # Downscale for much faster detection
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            
        if scale != 1.0:
            resized_gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale, fy=scale)
        else:
            resized_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces on downscaled grayscale image
        faces = self.face_cascade.detectMultiScale(
            resized_gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(int(30 * scale), int(30 * scale))
        )
        
        # Upscale coordinates back to original image size
        scaled_faces = []
        for x, y, w_box, h_box in faces:
            scaled_faces.append((
                int(x / scale),
                int(y / scale),
                int(w_box / scale),
                int(h_box / scale)
            ))
        
        # Debug output
        if len(scaled_faces) > 0:
            print(f"🔍 Face detection: found {len(scaled_faces)} faces (optimized scale {scale:.2f})")
        
        return scaled_faces
    
    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize faces in the image if recognition is enabled.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of recognition results
        """
        if not self.enable_recognition or self.face_recognizer is None:
            # Fall back to basic detection
            faces = self.detect_faces(image)
            return [{'bbox': (x, y, w, h), 'name': 'Face', 'confidence': 1.0, 'is_known': False} 
                   for x, y, w, h in faces]
        
        return self.face_recognizer.recognize_faces(image)
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], face_results: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Draw bounding boxes around detected faces (basic detection mode).
        
        Args:
            image: Input image as numpy array
            faces: List of face coordinates (x, y, width, height)
            face_results: List of dictionaries containing face results with distance
            
        Returns:
            Image with face bounding boxes drawn
        """
        result_image = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Prepare label with distance if available
            dist_str = ""
            if face_results and i < len(face_results) and 'distance' in face_results[i]:
                dist_str = f" ({face_results[i]['distance']:.1f}m)"
                
            label = f"Face{dist_str}"
            
            # Add label
            cv2.putText(
                result_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )
        
        return result_image
    
    def draw_recognitions(self, image: np.ndarray, recognitions: List[Dict]) -> np.ndarray:
        """
        Draw recognition results on image.
        
        Args:
            image: Input image as numpy array
            recognitions: List of recognition results
            
        Returns:
            Image with recognition annotations drawn
        """
        if self.enable_recognition and self.face_recognizer is not None:
            return self.face_recognizer.draw_recognitions(image, recognitions)
        else:
            # Fall back to basic face drawing
            faces = [(r['bbox'][0], r['bbox'][1], r['bbox'][2], r['bbox'][3]) for r in recognitions]
            return self.draw_faces(image, faces)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, Dict]:
        """
        Process a single frame to detect/recognize faces.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (processed_frame, number_of_faces, recognition_stats)
        """
        if self.enable_recognition and self.face_recognizer is not None:
            # Use recognition mode
            recognitions = self.recognize_faces(frame)
            
            # Add distance to recognitions
            h_img, w_img = frame.shape[:2]
            focal_length = w_img * 0.9375
            for r in recognitions:
                xf, yf, wf, hf = r['bbox']
                distance = (0.15 * focal_length) / max(1.0, float(wf))
                r['distance'] = max(0.1, min(40.0, float(distance)))
                
            self.last_results = recognitions
            annotated_frame = self.draw_recognitions(frame, recognitions)
            
            # Calculate recognition stats
            known_count = sum(1 for r in recognitions if r.get('is_known', False))
            unknown_count = len(recognitions) - known_count
            
            stats = {
                'total_faces': len(recognitions),
                'known_faces': known_count,
                'unknown_faces': unknown_count,
                'recognized_names': [r['name'] for r in recognitions if r.get('is_known', False)]
            }
            
            return annotated_frame, len(recognitions), stats
        else:
            # Use basic detection mode
            faces = self.detect_faces(frame)
            
            h_img, w_img = frame.shape[:2]
            focal_length = w_img * 0.9375
            self.last_results = []
            for x, y, w, h in faces:
                distance = (0.15 * focal_length) / max(1.0, float(w))
                self.last_results.append({
                    'bbox': (x, y, w, h),
                    'name': 'Face',
                    'confidence': 1.0,
                    'is_known': False,
                    'distance': max(0.1, min(40.0, float(distance)))
                })
                
            annotated_frame = self.draw_faces(frame, faces, self.last_results)
            
            stats = {
                'total_faces': len(faces),
                'known_faces': 0,
                'unknown_faces': len(faces),
                'recognized_names': []
            }
            
            return annotated_frame, len(faces), stats
    
    def toggle_recognition(self, enable: bool):
        """
        Enable or disable face recognition.
        
        Args:
            enable: Whether to enable recognition
        """
        if enable and not self.enable_recognition:
            try:
                self.face_recognizer = FaceRecognizer()
                self.enable_recognition = True
                print("✅ Face recognition enabled")
            except Exception as e:
                print(f"⚠️ Failed to enable face recognition: {e}")
                self.enable_recognition = False
        elif not enable:
            self.enable_recognition = False
            self.face_recognizer = None
            print("ℹ️ Face recognition disabled")
    
    def get_face_recognizer(self) -> Optional[FaceRecognizer]:
        """Get the face recognizer instance if available."""
        return self.face_recognizer if self.enable_recognition else None