# AKSHU 5.0 — Smart Vision & Continuous Voice Guidance

This deployment separates the public dashboard from the camera computer:

- **Render** hosts the lightweight Flask dashboard. It does not need a camera, YOLO, OpenCV, Windows audio, or `pyttsx3`.
- **Your Windows laptop** runs `akshu_agent.py`, opens the camera, performs YOLO inference locally, and sends detections/guidance to Render over HTTPS.
- **The browser** speaks guidance using `window.speechSynthesis`. Click **ENABLE VOICE** once after opening the page. The site never treats a queued server message as proof that audio played.
- **Demo Mode** runs in the browser without a camera or local Python agent and labels every scenario as simulated.

> Experimental prototype only. It is not a safety-rated navigation aid and must not replace a white cane, guide dog, or human assistance.

## 1. Files and preservation

Extract the ZIP, then copy the files inside its `akshu_render` folder into the root of your existing `D:\akshu` folder. Before copying, back up any existing files with the same names (especially `requirements.txt`, `README.md`, and `.gitignore`) so you can restore any project-specific settings. For example, in PowerShell:

```powershell
Expand-Archive "$env:USERPROFILE\Downloads\AKSHU_5.0_Render_Deployment_Files.zip" -DestinationPath "$env:TEMP\akshu-render" -Force
cd D:\akshu
if (Test-Path .\requirements.txt) { Copy-Item .\requirements.txt .\requirements-existing.txt -Force }
Copy-Item "$env:TEMP\akshu-render\akshu_render\*" . -Recurse -Force
```

The ZIP does not contain or overwrite `akshu_web.py`, `akshu_navigation_5_0.py`, or `core\voice.py`. If your original `requirements.txt` contained extra packages for your old local application, retain the backup and install those dependencies separately when running that old application.

Deployment files:

- `app.py`
- `akshu_agent.py`
- `requirements.txt`
- `requirements-agent.txt`
- `render.yaml`
- `.gitignore`
- `.env.example`
- `templates\index.html`
- `static\style.css`
- `static\app.js`

Keep your existing `akshu_navigation_5_0.py`, `core\voice.py`, YOLO modules, and model files. This deployment does not overwrite or delete them. The agent reuses the detection approach and YOLO positioning logic from the supplied navigation file. The existing `core.voice.AkshuVoice` remains available to your original local application, but is deliberately not started by the cloud dashboard or new agent: browser speech is the single voice path for the public site.

## 2. Create the GitHub repository (Windows PowerShell)

Open PowerShell and run:

```powershell
cd D:\akshu
```

Copy the new files/folders into this folder, preserving the existing files. Then check:

```powershell
Get-ChildItem
Get-ChildItem .\templates
Get-ChildItem .\static
```

Create a GitHub repository named `akshu-5-smart-vision` (do not add a README, license, or .gitignore if your local folder already has them). Then run:

```powershell
git init
git add .
git status
git commit -m "Prepare AKSHU 5.0 Render dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/akshu-5-smart-vision.git
git push -u origin main
```

Replace `YOUR-GITHUB-USERNAME` with your GitHub username. If `origin` already exists, use `git remote set-url origin https://github.com/YOUR-GITHUB-USERNAME/akshu-5-smart-vision.git` instead of `git remote add origin`.

Do not commit `.env`, private tokens, passwords, or large model weights. `.gitignore` excludes these by default.

## 3. Deploy on Render

1. Sign in at [Render](https://render.com/).
2. Choose **New + → Blueprint** and connect your GitHub repository.
3. Render detects `render.yaml`. Review and create the web service.
4. Wait for the first deploy to finish. Open the generated `https://...onrender.com` URL.
5. Test `https://YOUR-SERVICE.onrender.com/health`. It should return JSON containing `"ok": true`.
6. In Render → your service → **Environment**, confirm `AKSHU_AGENT_TOKEN` exists. `render.yaml` asks Render to generate it. If it is missing, add it manually as a long random secret. Never paste this token into dashboard JavaScript or a public GitHub file.
7. Save changes and redeploy if you changed environment variables.

Render free services may sleep after inactivity. The first page load after sleep can take a little longer.

## 4. Run the live Windows agent

Use a second PowerShell window on the laptop with the camera. Keep the dashboard URL and token private. In PowerShell:

```powershell
cd D:\akshu
py -3.11 -m venv .venv-agent
.\.venv-agent\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-agent.txt
```

If PowerShell blocks activation, run this once in that PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-agent\Scripts\Activate.ps1
```

Set the URL and token for this PowerShell session (replace both values):

```powershell
$env:AKSHU_DASHBOARD_URL = "https://YOUR-SERVICE.onrender.com"
$env:AKSHU_AGENT_TOKEN = "PASTE_THE_RENDER_AKSHU_AGENT_TOKEN_HERE"
```

Start the agent:

```powershell
python .\akshu_agent.py
```

Allow Python through Windows camera/privacy permissions if prompted. The first run of `yolov8n.pt` may download the YOLO model. Leave this window running while demonstrating Live Vision. Press `Ctrl+C` to stop it.

Optional: run without uploading the camera preview image (detections and announcements still work):

```powershell
python .\akshu_agent.py --no-preview
```

Optional camera/model settings:

```powershell
$env:AKSHU_CAMERA_INDEX = "0"
$env:AKSHU_MODEL = "yolov8n.pt"
$env:AKSHU_CONFIDENCE = "0.45"
```

If you have a different existing model, set `AKSHU_MODEL` to its path or filename. Keep model files on the laptop; do not upload them to the cloud repository.

## 5. Demonstrate browser voice and Demo Mode

1. Open the public Render URL in current Chrome or Edge.
2. Click **ENABLE VOICE**. This click is required by browser audio autoplay policies.
3. Click **TEST VOICE** and confirm you hear the test sentence.
4. Click **DEMO MODE**. The page cycles through person, obstacle, vehicle, and visually-clear examples. The banner says **SIMULATED DATA · NOT A REAL CAMERA**.
5. Click **STOP** to end the demo.
6. For live mode, start `akshu_agent.py` in PowerShell. The dashboard should show the agent as connected, live detections, and browser speech for new guidance announcements.

The dashboard polls the cloud API; a short delay between a local detection and its appearance on the page is normal. Browser voice only speaks when enabled and unmuted. If the page is muted or not open, the cloud event remains visible in the activity log but cannot make sound on the professor's device.

## 6. Environment variables

| Variable | Where | Purpose |
|---|---|---|
| `AKSHU_AGENT_TOKEN` | Render and Windows agent | Shared secret for authenticating local agent requests. Keep private. |
| `PORT` | Render-provided | Flask/Gunicorn listening port; do not hardcode it. |
| `PYTHON_VERSION` | Render | Python runtime selected in `render.yaml`. |
| `AKSHU_DASHBOARD_URL` | Windows agent | Public Render URL. |
| `AKSHU_CAMERA_INDEX` | Windows agent | Camera index, default `0`. |
| `AKSHU_MODEL` | Windows agent | YOLO model filename/path, default `yolov8n.pt`. |
| `AKSHU_CONFIDENCE` | Windows agent | YOLO confidence threshold, default `0.45`. |

## 7. Quick troubleshooting

- **Website does not load:** check Render deploy logs and `/health`; ensure the start command is `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120`.
- **Voice is silent:** click ENABLE VOICE, then TEST VOICE; check device volume, browser tab mute, and the red voice error message. Use Chrome or Edge. A dashboard event alone does not prove audio played.
- **Agent stays disconnected:** check the URL, token, HTTPS, and PowerShell output. The Render token must exactly match the Windows `AKSHU_AGENT_TOKEN` value.
- **Camera unavailable:** close other camera apps, check Windows Settings → Privacy & security → Camera, or try another `AKSHU_CAMERA_INDEX`.
- **YOLO model download fails:** check laptop internet access and set `AKSHU_MODEL` to a model file already on the laptop.
- **Cloud dashboard sleeps:** a free Render service may take time to wake after inactivity.

## 8. Architecture and safety notes

The agent sends object labels, approximate image-relative position, confidence, FPS, optional annotated JPEG preview, and newly generated guidance messages. The shared token is sent only from the Windows agent to Flask in an Authorization header. It is never embedded in browser JavaScript. Agent heartbeats expire after 20 seconds without contact.

This is a demonstration prototype, not a certified mobility or collision-avoidance system. YOLO object categories do not establish object distance, motion, traversability, or safety.
