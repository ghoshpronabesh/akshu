"""
Object Detection Module using YOLO v8 with Distance Estimation, Bengali Translations, and GPU Acceleration.
Optimized for long-range detection (up to 40 meters) and high-resolution feeds.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO


class ObjectDetector:
    """Object detection using YOLO v8 with monocular distance estimation and GPU acceleration."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.3, imgsz: int = 640):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence threshold for detections
            imgsz: Inference image size (resolution)
        """
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        
        # Select device: Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU fallback
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        print(f"Initializing YOLOv8 ({model_path}) on device: {self.device}")
        
        # Real world widths (in meters) for standard YOLO objects for distance estimation
        self.REAL_WORLD_WIDTHS = {
            'person': 0.45,       # shoulder width
            'bicycle': 0.60,      # handlebar width
            'car': 1.80,          # car width
            'motorcycle': 0.70,   # width
            'bus': 2.50,
            'train': 3.00,
            'truck': 2.50,
            'bench': 1.20,
            'bird': 0.15,
            'cat': 0.25,
            'dog': 0.35,
            'horse': 0.60,
            'sheep': 0.50,
            'cow': 0.65,
            'backpack': 0.30,
            'umbrella': 0.80,
            'handbag': 0.30,
            'tie': 0.10,
            'suitcase': 0.45,
            'sports ball': 0.22,
            'bottle': 0.08,
            'wine glass': 0.08,
            'cup': 0.08,
            'fork': 0.03,
            'knife': 0.03,
            'spoon': 0.03,
            'bowl': 0.15,
            'banana': 0.15,
            'apple': 0.08,
            'orange': 0.08,
            'chair': 0.50,
            'couch': 1.60,
            'potted plant': 0.30,
            'bed': 1.20,
            'dining table': 1.00,
            'toilet': 0.50,
            'tv': 0.90,
            'laptop': 0.35,
            'mouse': 0.06,
            'remote': 0.05,
            'keyboard': 0.40,
            'cell phone': 0.08,
            'microwave': 0.50,
            'oven': 0.60,
            'refrigerator': 0.80,
            'book': 0.15,
            'clock': 0.25,
            'vase': 0.15,
            'scissors': 0.10,
            'teddy bear': 0.30,
            'toothbrush': 0.02,
            'default': 0.30
        }
        
        # Complete dictionary of Bengali translations for all 80 COCO classes
        self.BENGALI_CLASS_NAMES = {
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
            'oven': 'ওভেন',
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

        try:
            # Load YOLO model and move it to specified acceleration device
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"YOLO model loaded successfully from {model_path} on {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the given image and calculate estimated distances.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing detection information
        """
        # Run inference using GPU device acceleration with the configured confidence threshold and image resolution
        results = self.model(image, verbose=False, device=self.device, conf=self.confidence_threshold, imgsz=self.imgsz)
        
        detections = []
        
        # Get actual image height and width to scale focal length dynamically
        h_img, w_img = image.shape[:2]
        
        # Scale focal length: standard 60-deg FOV lens has focal length of ~0.9375 * image_width in pixels
        focal_length = w_img * 0.9375
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Copy all boxes to CPU at once to avoid multiple GPU-CPU synchronization barriers in the loop!
                # This is a major speedup for Apple Silicon MPS.
                boxes_cpu = boxes.cpu()
                
                for box in boxes_cpu:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Determine minimum confidence required for this specific class
                    required_confidence = self.confidence_threshold
                    
                    # Background furniture/items prone to background clutter require higher confidence
                    if class_name in ['chair', 'couch', 'potted plant', 'bed', 'dining table']:
                        required_confidence = max(required_confidence, 0.45)
                    elif class_name == 'tie':
                        # Ties are extremely prone to false positives on vertical patterns (like windows/grates)
                        required_confidence = max(required_confidence, 0.60)
                    
                    if confidence >= required_confidence:
                        x1, y1, x2, y2 = box.xyxy[0].numpy()
                        
                        # Apply spatial filter to eliminate false-positive tie detections on background elements
                        if class_name == 'tie':
                            box_w = float(x2 - x1)
                            box_h = float(y2 - y1)
                            # A real necktie is a small accessory; it should not occupy a major portion of the image height or width.
                            if box_h > h_img * 0.40 or box_w > w_img * 0.15:
                                continue
                        
                        # Estimate Monocular Distance dynamically based on width in pixels
                        box_width_px = max(1.0, float(x2 - x1))
                        real_width_m = self.REAL_WORLD_WIDTHS.get(class_name, self.REAL_WORLD_WIDTHS['default'])
                        
                        # Distance formula: (Real Width * Focal Length) / Pixel Width
                        estimated_distance = (real_width_m * focal_length) / box_width_px
                        
                        # Expanded range cap up to 40 meters (standard daily vision requirement for blind navigation)
                        estimated_distance = max(0.1, min(40.0, float(estimated_distance)))
                        
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'distance': estimated_distance
                        }
                        detections.append(detection)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], language: str = 'en') -> np.ndarray:
        """
        Draw bounding boxes, distances, and translated names for detected objects.
        
        Args:
            image: Input image as numpy array
            detections: List of detection dictionaries
            language: UI/Voice language selection ('en' or 'bn')
            
        Returns:
            Image with detection annotations
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            distance = detection['distance']
            
            # Determine display name based on language selection
            if language == 'bn':
                display_name = self.BENGALI_CLASS_NAMES.get(class_name, class_name)
                label = f"{display_name} ({distance:.1f}m)"
            else:
                display_name = class_name.capitalize()
                label = f"{display_name} ({distance:.1f}m)"
            
            # Choose color based on class
            color = self._get_color_for_class(detection['class_id'])
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Get text size for background label box
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw solid label background rectangle
            y_label_start = max(y1 - text_height - 10, 0)
            cv2.rectangle(
                result_image,
                (x1, y_label_start),
                (x1 + text_width + 6, y_label_start + text_height + 10),
                color,
                -1
            )
            
            # Draw white text label inside the background rectangle
            cv2.putText(
                result_image,
                label,
                (x1 + 3, y_label_start + text_height + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return result_image
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Generate a consistent color for each class ID."""
        colors = [
            (46, 204, 113),   # Premium Emerald Green
            (231, 76, 60),    # Premium Coral Red
            (52, 152, 219),   # Premium Peter River Blue
            (241, 196, 15),   # Premium Sun Flower Yellow
            (155, 89, 182),   # Premium Amethyst Purple
            (26, 188, 156),   # Premium Turquoise Cyan
            (230, 126, 34),   # Premium Carrot Orange
            (52, 73, 94),     # Premium Dark Slate
        ]
        return colors[class_id % len(colors)]
    
    def process_frame(self, frame: np.ndarray, language: str = 'en') -> Tuple[np.ndarray, int]:
        """
        Process a single frame to detect, estimate distance, and annotate objects.
        
        Args:
            frame: Input frame as numpy array
            language: Selected language
            
        Returns:
            Tuple of (processed_frame, number_of_objects)
        """
        detections = self.detect_objects(frame)
        annotated_frame = self.draw_detections(frame, detections, language)
        return annotated_frame, len(detections)
    
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.model.overrides['conf'] = self.confidence_threshold
        
    def set_imgsz(self, imgsz: int):
        """Update the inference image size resolution."""
        self.imgsz = imgsz