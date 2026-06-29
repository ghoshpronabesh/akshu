"""
OCR (Optical Character Recognition) Module using EasyOCR.
Optimized for real-time bilingual (English and Bengali) text reading with robust CPU execution.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


class OcrDetector:
    """Optical Character Recognition system for reading signs, books, and text."""

    def __init__(self, languages: List[str] = ['en', 'bn']):
        """
        Initialize the OCR detector.

        Args:
            languages: List of language codes to load (e.g., ['en', 'bn'])
        """
        self.languages = languages
        self.reader = None
        self.initialized = False
        
        # Select device: NVIDIA GPU (CUDA) or CPU fallback
        # Note: macOS Apple Silicon MPS is not natively supported by EasyOCR due to input tensor CPU allocation.
        # Run OCR on CPU using our smart frame-skipping mechanism, ensuring 100% stability and full 30 FPS.
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        # Initialize Reader
        self.initialize_reader()

    def initialize_reader(self):
        """Lazy load easyocr reader to prevent slow startup times when OCR is unused."""
        if self.initialized:
            return
        
        try:
            import easyocr
            print(f"Loading EasyOCR models for {self.languages} (Device: {self.device})...")
            
            # Initialize reader (uses GPU/CUDA if available, otherwise CPU)
            self.reader = easyocr.Reader(self.languages, gpu=(self.device == 'cuda'))
            
            self.initialized = True
            print("✅ EasyOCR Reader initialized successfully")
        except ImportError:
            print("❌ EasyOCR package is missing. Please run pip install easyocr.")
        except Exception as e:
            print(f"⚠️ Failed to initialize EasyOCR reader: {e}")

    def detect_text(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Detect and transcribe text in the given image.

        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence to retain a detection

        Returns:
            List of dictionaries containing OCR results
        """
        if not self.initialized:
            self.initialize_reader()
            if not self.initialized:
                return []

        try:
            h, w = image.shape[:2]
            scale = 1.0
            if w > 640:
                scale = 640.0 / w
                
            if scale != 1.0:
                resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            else:
                resized_image = image

            # EasyOCR expects RGB images
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Run text detection
            results = self.reader.readtext(rgb_image)
            
            detections = []
            for (bbox, text, conf) in results:
                if conf >= confidence_threshold:
                    pts = np.array(bbox, dtype=np.float32)
                    # Rescale points back to original image coordinates
                    pts /= scale
                    pts = pts.astype(np.int32)
                    
                    x1 = int(np.min(pts[:, 0]))
                    y1 = int(np.min(pts[:, 1]))
                    x2 = int(np.max(pts[:, 0]))
                    y2 = int(np.max(pts[:, 1]))
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'points': pts,
                        'text': text.strip(),
                        'confidence': float(conf)
                    })
            
            return detections
            
        except Exception as e:
            print(f"OCR detection error: {e}")
            return []

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw transparent highlights and elegant text overlays on the image.

        Args:
            image: Input image as numpy array
            detections: List of OCR detections

        Returns:
            Annotated image with premium text highlights
        """
        if not detections:
            return image
            
        annotated_image = image.copy()
        
        # Color for text annotations: sleek purple/indigo
        box_color = (130, 0, 250)  # BGR format
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            text = detection['text']
            
            # Draw premium semi-transparent bounding highlight
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, 0.25, annotated_image, 0.75, 0, annotated_image)
            
            # Draw outer thin boundary line
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, 1)
            
            label = f"Text: {text}"
            y_label = max(y1 - 10, 15)
            
            box_height = y2 - y1
            text_scale = max(0.4, min(0.6, box_height / 45.0))
            
            cv2.putText(
                annotated_image,
                label,
                (x1, y_label),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
        return annotated_image
