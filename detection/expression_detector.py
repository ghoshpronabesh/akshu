"""
Facial Expression and Emotion Detector using MediaPipe Face Mesh.
Extracts 3D facial landmarks and calculates geometric ratios to classify expressions.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple


class ExpressionDetector:
    """Classifies facial expressions using MediaPipe Face Mesh landmarks."""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_expressions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect face landmarks and classify expressions.
        
        Returns:
            List of dictionaries containing classification, landmarks, and bboxes.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        detections = []
        
        if results.multi_face_landmarks:
            h, w, c = image.shape
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Key landmarks for ratio calculation
                # 61: Left mouth corner, 291: Right mouth corner
                # 13: Upper lip center, 14: Lower lip center
                # 70: Left inner eyebrow, 300: Right inner eyebrow
                # 159: Left eye top, 145: Left eye bottom
                # 386: Right eye top, 374: Right eye bottom
                
                def get_pt(idx):
                    lm = landmarks[idx]
                    return np.array([lm.x * w, lm.y * h])
                
                m_left = get_pt(61)
                m_right = get_pt(291)
                m_top = get_pt(13)
                m_bottom = get_pt(14)
                
                eb_left = get_pt(70)
                eb_right = get_pt(300)
                
                left_eye_center = (get_pt(159) + get_pt(145)) / 2.0
                right_eye_center = (get_pt(386) + get_pt(374)) / 2.0
                eye_dist = np.linalg.norm(left_eye_center - right_eye_center)
                
                # 1. Mouth Aspect Ratio (MAR)
                mouth_height = np.linalg.norm(m_top - m_bottom)
                mouth_width = np.linalg.norm(m_left - m_right)
                mar = mouth_height / max(1.0, mouth_width)
                
                # 2. Smile score (corners height vs mouth vertical center)
                mouth_y_center = (m_top[1] + m_bottom[1]) / 2.0
                corners_y_avg = (m_left[1] + m_right[1]) / 2.0
                # Higher corners mean smaller y coordinate
                smile_score = (mouth_y_center - corners_y_avg) / max(1.0, mouth_width)
                
                # 3. Eyebrow distance score (Anger/Focus)
                eb_dist_score = np.linalg.norm(eb_left - eb_right) / max(1.0, eye_dist)
                
                # Classification Logic
                expression = "Neutral"
                if mar > 0.45:
                    expression = "Surprised"
                elif smile_score > 0.05:
                    expression = "Happy"
                elif smile_score < -0.06:
                    expression = "Sad"
                elif eb_dist_score < 0.22:
                    expression = "Angry"
                    
                # Calculate bounding box of the face
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                bbox = (max(0, int(min(xs))), max(0, int(min(ys))), min(w, int(max(xs))), min(h, int(max(ys))))
                
                detections.append({
                    'landmarks': face_landmarks,
                    'expression': expression,
                    'bbox': bbox
                })
                
        return detections
        
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw light facial contours and expression label on the face."""
        result_image = image.copy()
        
        for det in detections:
            # Draw Face Mesh contour connections
            self.mp_draw.draw_landmarks(
                image=result_image,
                landmark_list=det['landmarks'],
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Draw bbox and label
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (46, 204, 113), 2)
            
            label = f"Expression: {det['expression']}"
            cv2.putText(
                result_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (46, 204, 113),
                2,
                cv2.LINE_AA
            )
            
        return result_image
