# Face & Object Detection System with Voice Announcements

A real-time AI-powered face, object, gesture, expression, and OCR text detection system featuring intelligent voice announcements. Optimized for accessibility, security monitoring, and interactive demonstrations. Built with Streamlit, OpenCV, YOLOv8, and custom voice routing.

---

## 🆕 ADVANCED FEATURES

### 🔊 Bluetooth Audio Selection & Routing
- **Earbuds & Headphones Ready**: Direct audio routing to your selected Bluetooth earbuds (e.g., AirPods, Bluetooth headphones) or system speakers.
- **Dynamic Device List**: Sidebar selector lists all connected output devices dynamically.
- **Smart Cooldown & De-cluttering**: Prevents announcement spam using configurable cooldown periods.
- **Bilingual Announcements**: High-quality spoken alerts in both English and Bengali.

### 📶 Raspberry Pi Camera Stream (Wi-Fi)
- **Wireless Input**: Toggle between your local webcam or a remote Raspberry Pi camera stream over Wi-Fi.
- **Flexible Formats**: Supports standard MJPEG streams (e.g., from `mjpg-streamer` or Flask servers) and RTSP streams.
- **Robust Connection Handling**: Dynamic connection checks and clear UI status warnings.

### 🔌 High-Quality Offline Voice Fallback (macOS `say`)
- **DNS & Offline Resilient**: If the cloud/Edge TTS voice engine fails due to offline environments (DNS/SSL lookup errors), the system seamlessly transitions to macOS's native `say` command.
- **Thread-safe Output**: Audio generated offline is rendered to temporary AIFF files and streamed through `sounddevice`, preserving your selected Bluetooth output routing.
- **Bilingual Offline Voices**: Uses macOS's native `Piya` voice for Bengali and `Daniel` for English, with a generic system default voice recovery.

---

## Features

- **Live Video Feed**: High-resolution processing with dynamic FPS display overlays.
- **Face Detection & Recognition**: OpenCV Haar Cascades for detection, combined with an LBPH recognizer to identify registered individuals from `known_faces/`.
- **YOLOv8 Object Detection**: High-performance multi-class detection (up to 80 COCO classes) with GPU/MPS hardware acceleration.
- **Hand Gesture Recognition**: MediaPipe Hands classifies sign language gestures (Thumbs Up/Down, Peace, OK, Open Palm, Fist, etc.).
- **Facial Expression Detection**: MediaPipe Face Mesh classifies expressions (Happy, Sad, Angry, Surprised, Neutral).
- **OCR Text Reading**: Frame-skipped EasyOCR parses printed English and Bengali text on the fly.

---

## Installation

1. Install Python 3.8 or higher.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Set Up Known Faces (Optional)
Create folders in the `known_faces/` directory using names (use underscores instead of spaces):
```bash
known_faces/
├── john_doe/
│   ├── image1.jpg
│   └── image2.jpg
└── jane_smith/
    ├── image1.jpg
    └── image2.jpg
```
The system will automatically generate face encodings on startup.

### 2. Run the Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

### 3. Using the Sidebar Controls

#### 📹 Camera Source Settings
1. **Camera Input Source**:
   - Select **Local Webcam** for your integrated camera.
   - Select **Raspberry Pi (WiFi Stream)** to stream from a remote Pi.
2. **Raspberry Pi Stream URL**: Input the MJPEG stream path (e.g., `http://192.168.1.100:8000/stream.mjpg`) or RTSP URL.

#### 🚀 System Initialization
- Click **Initialize System** to compile weights, build face encodings, and set up detectors. 

#### 🔊 Voice Announcements
1. **Enable Voice Mode**: Route speech announcements to your preferred output.
2. **Audio Output Device**: Route announcements directly to your connected earbuds/headphones or leave on "System Default".
3. **Voice Language**: Toggle between English and Bengali.

---

## Setting up Raspberry Pi Camera Stream
To stream from your Raspberry Pi to the app over Wi-Fi:

### Option A: Using MJPEG-Streamer (Recommended)
1. Run `mjpg-streamer` on the Raspberry Pi:
   ```bash
   mjpeg_streamer -i "input_uvc.so -d /dev/video0 -r 1280x720 -f 30" -o "output_http.so -w ./www -p 8000"
   ```
2. Enter the URL in the app: `http://<YOUR_PI_IP>:8000/?action=stream`

### Option B: Using a Flask/Python Streamer
Run a script on the Raspberry Pi that reads frames from the camera using OpenCV and streams them over a Flask route, then target `http://<YOUR_PI_IP>:5000/video_feed`.

---

## Technical File Structure

```
akshu/
├── app.py                      # Main Streamlit dashboard app
├── detection/
│   ├── face_detector.py        # Cascaded face detection and recognition handler
│   ├── face_recognizer.py      # Core LBPH training and matching engine
│   ├── object_detector.py      # Accelerated YOLOv8 object detector
│   ├── ocr_detector.py         # EasyOCR text scanner (bilingual)
│   ├── expression_detector.py  # MediaPipe facial expressions classifier
│   └── gesture_detector.py     # MediaPipe hand gesture/sign language parser
├── utils/
│   ├── camera.py              # Frame capture and stream interface
│   └── voice_announcer.py     # Multilingual TTS and audio device routing
├── known_faces/               # Reference images for face recognition
├── requirements.txt           # Package dependencies
└── README.md                 # Project documentation
```

---

## Troubleshooting

### Voice Announcements not Playing
- Ensure the correct **Audio Output Device** is chosen in the sidebar.
- Check that your headphones/earbuds are paired and connected to your Mac system bluetooth.
- Look at the console logs for edge-tts connection or sounddevice warnings.

### Slow Frame Rates
- Decrease the **YOLO Inference Resolution** in the settings.
- Lower the camera stream resolution.
- Turn off unused detectors (e.g. hand gestures or OCR).