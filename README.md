# Akshu Computer Vision Streamlit App

Akshu is a Streamlit dashboard for live face detection, OpenCV LBPH face
recognition, YOLOv8 object detection, optional EasyOCR text reading, gesture and
expression detection, and voice announcements.

This guide is written for the current laptop environment detected here:
Windows with PowerShell. The existing `.venv` in this folder points to a removed
Python 3.8 install, so recreate it before running the app.

## 1. Clone The Repository

This workspace is already cloned at `D:\AKSHU`. If you are starting on a fresh
machine, run:

```powershell
git clone https://github.com/ghoshpronabesh/akshu.git
cd akshu
```

If you are using this existing folder:

```powershell
cd /d D:\AKSHU
```

## 2. Install Python

Install 64-bit [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)
for Windows from python.org. It is a stable target for this ML/CV dependency
stack and has an official Windows installer. During installer setup, enable
"Add python.exe to PATH".

Check that Windows can see Python:

```powershell
py -0p
py -3.11 --version
```

Python 3.10 can also work. If you install Python 3.10 instead, replace
`py -3.11` with `py -3.10` in the commands below.

## 3. Recreate And Activate The Virtual Environment

The previous environment is broken, so remove it and create a fresh one:

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

Quick verification:

```powershell
python -c "import streamlit, cv2; print('streamlit ok'); print('cv2', cv2.__version__, 'face module:', hasattr(cv2, 'face'))"
python -m pip check
```

The app uses OpenCV's `cv2.face.LBPHFaceRecognizer_create()`, so
`opencv-contrib-python` is required. The current code does not import
`dlib` or `face_recognition`, so those packages are intentionally not in
`requirements.txt`.

## 5. Run The Dashboard On Your LAN

For Raspberry Pi camera use, do not rely on Streamlit's localhost-only URL.
Run the included LAN launcher:

```powershell
.\scripts\run_lan.ps1
```

It starts Streamlit with:

```powershell
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open on the laptop:

```text
http://localhost:8501
```

Open from another device on the same Wi-Fi using the laptop IP printed by the
script:

```text
http://<LAPTOP_IP>:8501
```

If Windows Firewall prompts you, allow Python/Streamlit on private networks.

## 6. Stream A Raspberry Pi Camera To The Laptop

The dashboard runs on the laptop. The camera runs on the Raspberry Pi and sends
MJPEG video over Wi-Fi.

On the Raspberry Pi:

```bash
git clone https://github.com/ghoshpronabesh/akshu.git
cd akshu
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-pi.txt
python scripts/pi_camera_stream.py --host 0.0.0.0 --port 5000 --source 0
```

For the Raspberry Pi Camera Module instead of a USB camera:

```bash
sudo apt update
sudo apt install -y python3-picamera2
python scripts/pi_camera_stream.py --host 0.0.0.0 --port 5000 --picamera2
```

The Pi script prints a URL like:

```text
http://192.168.1.100:5000/video_feed
```

On the laptop dashboard:

1. Click `Initialize System`.
2. In the sidebar, set `Camera Input Source` to `Raspberry Pi (WiFi Stream)`.
3. Paste `http://<RASPBERRY_PI_IP>:5000/video_feed`.
4. Click `Start`.

Important: `localhost` on the laptop means the laptop, not the Raspberry Pi.
Use the Pi IP address.

## Alternative Pi Stream URLs

The app also accepts common OpenCV-readable streams:

```text
http://<RASPBERRY_PI_IP>:5000/video_feed
http://<RASPBERRY_PI_IP>:5000/stream.mjpg
http://<RASPBERRY_PI_IP>:8000/?action=stream
rtsp://<RASPBERRY_PI_IP>:8554/stream
```

If you already use `mjpg-streamer` on the Pi, run it on the Pi and paste its
`?action=stream` URL into the dashboard.

## Troubleshooting

### Python Or Virtual Environment Problems

- `python` or `py` is not recognized: install 64-bit Python 3.11.9 and enable
  PATH.
- `.venv` points to old Python 3.8: delete `.venv` and recreate it.
- Activation is blocked: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Dependency Install Problems

- `cv2.face` is missing: reinstall OpenCV contrib:

  ```powershell
  pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
  pip install opencv-contrib-python
  ```

- `dlib` or `face-recognition` fails with C++ build errors: the current app does
  not need them. Remove those packages unless you are testing an older branch.
  If an old branch truly needs them, install "Visual Studio Build Tools" with
  the C++ workload, then retry.
- `torch`, `mediapipe`, or `easyocr` has no matching wheel: use 64-bit Python
  3.10 or 3.11, then upgrade pip and reinstall.
- YOLO model download fails on first run: check internet access or manually
  place the required `yolov8*.pt` file in the project folder.

### Raspberry Pi Stream Problems

- Dashboard cannot connect to Pi stream: confirm laptop and Pi are on the same
  network, use the Pi IP address, and allow port `5000` through the Pi firewall.
- Browser test first from the laptop:

  ```text
  http://<RASPBERRY_PI_IP>:5000/video_feed
  ```

- USB camera not opening on Pi: try `--source 1`, close other camera apps, and
  check `ls /dev/video*`.
- Pi Camera Module not opening: run `libcamera-hello` on the Pi and install
  `python3-picamera2`.

### Audio Problems

- No voice output: choose a valid output device in the sidebar and make sure
  Bluetooth headphones are connected before starting voice mode.
- `sounddevice` cannot find a device: reinstall `sounddevice` and `soundfile`,
  then restart the terminal.
- Cloud neural voice fails offline: the app falls back to local `pyttsx3` where
  available.

## File Map

```text
app.py                         Streamlit dashboard
detection/                     Face, object, OCR, expression, gesture detection
utils/camera.py                Camera capture and frame processing
utils/voice_announcer.py       Voice announcements and AI response routing
scripts/run_lan.ps1            Windows LAN Streamlit launcher
scripts/pi_camera_stream.py    Raspberry Pi MJPEG camera streamer
requirements.txt               Laptop/dashboard dependencies
requirements-pi.txt            Raspberry Pi camera-stream dependencies
known_faces/                   Optional registered face images
```
