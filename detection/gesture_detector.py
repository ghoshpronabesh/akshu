"""
Hand Gesture and Sign Language Detector using MediaPipe Hands.
Calculates finger extensions and coordinates to recognize signs.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple


class GestureDetector:
    """Detects and interprets hand gestures for Sign Language."""
    
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.7):
        """Initialize MediaPipe Hands."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_gestures(self, image: np.ndarray) -> List[Dict]:
        """
        Detect hand landmarks and classify gestures.
        
        Returns:
            List of dictionaries containing detected gestures, landmarks, and bboxes.
        """
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        detections = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label  # 'Left' or 'Right'
                landmarks = hand_landmarks.landmark
                
                h, w, c = image.shape
                coords = []
                for lm in landmarks:
                    coords.append((int(lm.x * w), int(lm.y * h), lm.z))
                
                wrist = np.array(coords[0][:2])
                index_mcp = np.array(coords[5][:2])
                
                # Bounding box calculation
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                bbox = (max(0, min(xs) - 15), max(0, min(ys) - 15), min(w, max(xs) + 15), min(h, max(ys) + 15))
                
                # Check if a finger is extended relative to hand wrist distance
                def is_extended(tip_idx, pip_idx):
                    tip = np.array(coords[tip_idx][:2])
                    pip = np.array(coords[pip_idx][:2])
                    return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist)
                
                index_ext = is_extended(8, 6)
                middle_ext = is_extended(12, 10)
                ring_ext = is_extended(16, 14)
                pinky_ext = is_extended(20, 18)
                
                # Thumb logic: check displacement relative to the wrist-index palm vector
                thumb_tip = np.array(coords[4][:2])
                thumb_mcp = np.array(coords[2][:2])
                palm_center = (wrist + index_mcp) / 2
                thumb_ext = np.linalg.norm(thumb_tip - palm_center) > np.linalg.norm(thumb_mcp - palm_center)
                
                gesture = "Unknown Sign"
                
                # Classification Rules:
                # 1. Open Palm (Hello/Stop)
                if index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext:
                    gesture = "Open Palm (Hello)"
                # 2. Fist (Rock)
                elif not (index_ext or middle_ext or ring_ext or pinky_ext or thumb_ext):
                    gesture = "Fist (Yes)"
                # 3. Thumbs Up / Down
                elif thumb_ext and not (index_ext or middle_ext or ring_ext or pinky_ext):
                    if thumb_tip[1] < thumb_mcp[1]:
                        gesture = "Thumbs Up (Like)"
                    else:
                        gesture = "Thumbs Down (Dislike)"
                # 4. Victory / Peace (V Sign)
                elif index_ext and middle_ext and not (ring_ext or pinky_ext):
                    gesture = "Peace Sign (V)"
                # 5. OK Sign: Thumb and Index touch, others extended
                elif middle_ext and ring_ext and pinky_ext:
                    dist_thumb_index = np.linalg.norm(np.array(coords[4][:2]) - np.array(coords[8][:2]))
                    hand_size = np.linalg.norm(wrist - index_mcp)
                    if dist_thumb_index < (0.28 * hand_size):
                        gesture = "OK Sign"
                    else:
                        gesture = "Open Palm (Hello)"
                # 6. Call Me: Thumb and Pinky extended, others folded
                elif thumb_ext and pinky_ext and not (index_ext or middle_ext or ring_ext):
                    gesture = "Call Me Sign"
                # 7. I Love You: Thumb, Index, Pinky extended, Middle/Ring folded
                elif thumb_ext and index_ext and pinky_ext and not (middle_ext or ring_ext):
                    gesture = "I Love You Sign"
                # 8. Pointing: Index extended, others folded
                elif index_ext and not (middle_ext or ring_ext or pinky_ext):
                    gesture = "Pointing Sign"
                
                detections.append({
                    'landmarks': hand_landmarks,
                    'hand_type': hand_type,
                    'gesture': gesture,
                    'bbox': bbox
                })
                
        return detections
        
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw hand skeleton connections and gesture labels on image."""
        result_image = image.copy()
        
        for det in detections:
            # Draw skeleton landmarks
            self.mp_draw.draw_landmarks(
                result_image,
                det['landmarks'],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw label box
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (130, 0, 250), 2)
            
            label = f"{det['hand_type']} Hand: {det['gesture']}"
            cv2.putText(
                result_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (130, 0, 250),
                2,
                cv2.LINE_AA
            )
            
        return result_image
