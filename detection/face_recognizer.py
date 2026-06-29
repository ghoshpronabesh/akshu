"""
Face Recognition Module using OpenCV and advanced face matching techniques.
"""

import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional
import streamlit as st
from PIL import Image
import hashlib


class FaceRecognizer:
    """Face recognition system for identifying known individuals."""
    
    def __init__(self, known_faces_dir: str = "known_faces", similarity_threshold: float = 0.6):
        """
        Initialize the face recognizer.
        
        Args:
            known_faces_dir: Directory containing known face images
            similarity_threshold: Threshold for face matching (0.0 to 1.0)
        """
        self.known_faces_dir = known_faces_dir
        self.similarity_threshold = similarity_threshold
        self.encodings_file = os.path.join(known_faces_dir, "encodings.pkl")
        
        # Initialize face detection and recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage for known faces
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_labels = []
        self.label_to_name = {}
        
        # Load or create face encodings
        self.load_known_faces()
    
    def extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face features using OpenCV's built-in methods.
        
        Args:
            image: Input image
            
        Returns:
            Face features as numpy array or None if no face found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Use the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to standard size for consistency
        face_roi = cv2.resize(face_roi, (100, 100))
        
        return face_roi
    
    def load_known_faces(self):
        """Load known faces from directory and create encodings."""
        
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            return
        
        # Check if encodings file exists and is up to date
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # Verify if encodings are still valid (check modification times)
                if self._encodings_up_to_date(data.get('file_hashes', {})):
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    self.known_face_labels = data['labels']
                    self.label_to_name = data['label_to_name']
                    
                    # Train recognizer with loaded data
                    if len(self.known_face_encodings) > 0:
                        self.face_recognizer.train(self.known_face_encodings, np.array(self.known_face_labels))
                    
                    print(f"Loaded {len(self.known_face_names)} known faces from cache")
                    return
            except Exception as e:
                print(f"Error loading cached encodings: {e}")
        
        # Build encodings from scratch
        self._build_encodings()
    
    def _encodings_up_to_date(self, cached_hashes: Dict[str, str]) -> bool:
        """Check if cached encodings are up to date."""
        current_hashes = self._get_file_hashes()
        return cached_hashes == current_hashes
    
    def _get_file_hashes(self) -> Dict[str, str]:
        """Get hash of all image files for cache validation."""
        file_hashes = {}
        
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_dir) or person_name.startswith('.'):
                continue
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(person_dir, filename)
                    
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        file_hashes[file_path] = file_hash
        
        return file_hashes
    
    def _build_encodings(self):
        """Build face encodings from known faces directory."""
        print("Building face encodings...")
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_labels = []
        self.label_to_name = {}
        
        label_counter = 0
        
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            # Skip files and hidden directories
            if not os.path.isdir(person_dir) or person_name.startswith('.'):
                continue
            
            person_encodings = []
            person_labels = []
            
            print(f"Processing {person_name}...")
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, filename)
                    
                    try:
                        # Load image
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        
                        # Extract face features
                        face_features = self.extract_face_features(image)
                        
                        if face_features is not None:
                            person_encodings.append(face_features)
                            person_labels.append(label_counter)
                            
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
            
            if len(person_encodings) > 0:
                self.known_face_encodings.extend(person_encodings)
                self.known_face_labels.extend(person_labels)
                self.label_to_name[label_counter] = person_name
                
                # Add person name for each encoding
                for _ in person_encodings:
                    self.known_face_names.append(person_name)
                
                label_counter += 1
                print(f"Added {len(person_encodings)} encodings for {person_name}")
        
        # Train the recognizer
        if len(self.known_face_encodings) > 0:
            self.face_recognizer.train(self.known_face_encodings, np.array(self.known_face_labels))
            
            # Save encodings to cache
            self._save_encodings()
            
            print(f"Trained recognizer with {len(self.known_face_encodings)} face encodings for {len(set(self.known_face_names))} people")
        else:
            print("No valid face encodings found")
    
    def _save_encodings(self):
        """Save face encodings to cache file."""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'labels': self.known_face_labels,
                'label_to_name': self.label_to_name,
                'file_hashes': self._get_file_hashes()
            }
            
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"Saved encodings to {self.encodings_file}")
            
        except Exception as e:
            print(f"Error saving encodings: {e}")
    
    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize faces in the given image.
        
        Args:
            image: Input image
            
        Returns:
            List of dictionaries with recognition results
        """
        h, w = image.shape[:2]
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if scale != 1.0:
            resized_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        else:
            resized_gray = gray
        
        # Detect faces on downscaled grayscale image
        faces = self.face_cascade.detectMultiScale(resized_gray, 1.3, 7)
        
        results = []
        
        for (x, y, w_box, h_box) in faces:
            # Scale coordinates back to original size to extract high-quality face ROI
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(w_box / scale)
            orig_h = int(h_box / scale)
            
            # Bound check to avoid out of bounds slicing
            orig_x = max(0, min(orig_x, w - 1))
            orig_y = max(0, min(orig_y, h - 1))
            orig_w = max(1, min(orig_w, w - orig_x))
            orig_h = max(1, min(orig_h, h - orig_y))
            
            # Extract face region from the original gray image
            face_roi = gray[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            
            # Recognize face
            name = "Unknown"
            confidence = 0.0
            
            if len(self.known_face_encodings) > 0:
                try:
                    label, confidence_score = self.face_recognizer.predict(face_roi_resized)
                    
                    # Convert confidence to similarity (lower distance = higher similarity)
                    # LBPH returns confidence as distance, so we invert it
                    similarity = max(0, 1 - (confidence_score / 100))
                    
                    if similarity >= self.similarity_threshold:
                        name = self.label_to_name.get(label, "Unknown")
                        confidence = similarity
                    
                except Exception as e:
                    print(f"Recognition error: {e}")
            
            result = {
                'bbox': (orig_x, orig_y, orig_w, orig_h),
                'name': name,
                'confidence': confidence,
                'is_known': name != "Unknown"
            }
            
            results.append(result)
        
        return results
    
    def draw_recognitions(self, image: np.ndarray, recognitions: List[Dict]) -> np.ndarray:
        """
        Draw recognition results on image.
        
        Args:
            image: Input image
            recognitions: List of recognition results
            
        Returns:
            Image with recognition annotations
        """
        result_image = image.copy()
        
        for recognition in recognitions:
            x, y, w, h = recognition['bbox']
            name = recognition['name']
            confidence = recognition['confidence']
            is_known = recognition['is_known']
            
            # Choose colors
            color = (0, 255, 0) if is_known else (0, 0, 255)  # Green for known, Red for unknown
            
            # Draw rectangle
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            dist_str = f" {recognition['distance']:.1f}m" if 'distance' in recognition else ""
            if is_known:
                label = f"{name} ({confidence:.2f}){dist_str}"
            else:
                label = f"Unknown{dist_str}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                result_image,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return result_image
    
    def add_person(self, person_name: str, images: List[np.ndarray]) -> bool:
        """
        Add a new person to the known faces database.
        
        Args:
            person_name: Name of the person
            images: List of face images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create person directory
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save images
            for i, image in enumerate(images):
                filename = f"{person_name}_{i+1}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, image)
            
            # Rebuild encodings
            self._build_encodings()
            
            return True
            
        except Exception as e:
            print(f"Error adding person {person_name}: {e}")
            return False
    
    def get_known_people(self) -> List[str]:
        """Get list of known people."""
        return list(set(self.known_face_names))
    
    def remove_person(self, person_name: str) -> bool:
        """
        Remove a person from known faces database.
        
        Args:
            person_name: Name of person to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if os.path.exists(person_dir):
                import shutil
                shutil.rmtree(person_dir)
                
                # Rebuild encodings
                self._build_encodings()
                return True
            
            return False
            
        except Exception as e:
            print(f"Error removing person {person_name}: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a frame for face recognition.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, recognition_results)
        """
        recognitions = self.recognize_faces(frame)
        annotated_frame = self.draw_recognitions(frame, recognitions)
        return annotated_frame, recognitions