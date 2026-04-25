# DualSentinel — Real-Time Multi-Camera Accident Prevention System

## Overview
DualSentinel fuses two independent AI streams — a **driver-facing webcam** and a **road-facing dashcam** — into a unified real-time risk engine that detects drowsiness and vehicle proximity simultaneously.

---

## Architecture

```
CAM 1 (Webcam)               CAM 2 (dashcam.mp4)
      ↓                              ↓
MediaPipe FaceMesh           YOLOv8n (COCO pretrained)
+ YOLOv8-cls eye model       Vehicle detection
Blink / microsleep tracker   Largest bbox → TTC
drowsy: bool                 proximity_score: 0–100
      ↓                              ↓
           ── Risk Fusion Engine ──
             risk_score = 0.5×drowsiness + 0.5×proximity
             SAFE (<40) / WARNING (40–70) / DANGER (>70)
                          ↓
             ┌────────────┼────────────┐
         pyttsx3 TTS  OpenCV HUD  incidents.csv
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the eye model
Download the eye dataset from Kaggle:
- **Eyes Dataset**: https://www.kaggle.com/datasets/charunisa/eyes-dataset
- **MRL Eye Dataset** (optional): https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset

Organize into:
```
dataset/
  train/
    open/        ← open eye images
    closed/      ← closed eye images
  val/
    open/
    closed/
```

Then run:
```bash
python train.py
```
Output: `runs/train/eye_model/weights/best.pt`

### 3. Get a dashcam video
Download a free dashcam video from [Pexels](https://www.pexels.com/search/videos/dashcam%20driving/) and save as `dashcam.mp4` in the project root.

### 4. Run the system
```bash
python fusion.py

# Options:
python fusion.py --cam 0 --video dashcam.mp4
python fusion.py --cam 1 --video 1        # dual webcam setup
```

Press `q` to quit and save `incidents.csv`.

---

## Files

| File | Purpose |
|---|---|
| `train.py` | Fine-tune YOLOv8n-cls on eye open/closed dataset |
| `driver_stream.py` | Cam 1: MediaPipe + YOLOv8 eye classifier → `drowsy` |
| `road_stream.py` | Cam 2: YOLOv8n COCO vehicle detection → TTC |
| `fusion.py` | Main pipeline: risk engine, HUD, TTS, CSV log |
| `requirements.txt` | All pip dependencies |
| `incidents.csv` | Auto-generated on quit |

---

## Risk Formula

```python
drowsiness_score = 100 if drowsy else 0
proximity_score  = clamp((1 - TTC/10) * 100, 0, 100)
risk_score       = (drowsiness_score × 0.5) + (proximity_score × 0.5)
```

| Score | Label | Action |
|---|---|---|
| > 70 | 🔴 DANGER | Voice alert + red HUD |
| 40–70 | 🟠 WARNING | Voice alert + orange HUD |
| < 40 | 🟢 SAFE | No alert + green HUD |

---

## What's Deep Learning vs Not

| Component | Type |
|---|---|
| YOLOv8n-cls eye model (fine-tuned) | ✅ Deep Learning — CNN, transfer learning |
| YOLOv8n vehicle detection (COCO) | ✅ Deep Learning — pretrained CNN |
| MediaPipe FaceMesh | ✅ Deep Learning — neural landmark model |
| TTC estimation | ❌ Physics approximation |
| Risk fusion | ❌ Weighted rule engine |

---

## Limitations
- TTC is approximated — no calibrated depth sensor
- EAR not used — relies on eye classifier confidence threshold
- Tested on recorded video, not live dual-camera hardware
- Not tested in night/rain conditions
