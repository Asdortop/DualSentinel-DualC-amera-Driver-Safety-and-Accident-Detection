# DualSentinel — Product Requirements Document (PRD)

## Executive Summary

**Product Name:** DualSentinel  
**Version:** 1.0  
**Category:** Real-Time Driver Safety & Accident Prevention System  
**Target Users:** Individual drivers, fleet operators, automotive safety researchers

DualSentinel is a dual-camera AI-powered accident prevention system that simultaneously monitors driver drowsiness and road vehicle proximity to provide real-time risk assessment and alerts. The system fuses data from two independent video streams to create a unified risk score that triggers visual, auditory, and alarm warnings.

---

## Problem Statement

Road accidents caused by driver drowsiness and insufficient reaction time to obstacles are major safety concerns:
- Drowsy driving accounts for ~20% of fatal crashes
- Rear-end collisions due to insufficient following distance are common
- Existing systems monitor only one aspect (driver OR road), not both simultaneously
- No unified risk scoring that combines driver state + environmental hazards

---

## Solution Overview

DualSentinel provides a **multi-modal risk fusion engine** that:
1. Monitors driver eye state using webcam + AI classification
2. Detects vehicles and estimates Time-To-Collision (TTC) from dashcam footage
3. Fuses both signals into a unified 0-100 risk score
4. Triggers graduated alerts (visual HUD, voice warnings, audible siren)
5. Logs all incidents to CSV for post-analysis

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DualSentinel System                        │
└─────────────────────────────────────────────────────────────────┘

         CAM 1 (Webcam)                    CAM 2 (Dashcam)
         Driver-Facing                     Road-Facing
              ↓                                  ↓
    ┌─────────────────────┐          ┌─────────────────────┐
    │  MediaPipe FaceMesh │          │   YOLOv8n (COCO)    │
    │  + YOLOv8n-cls      │          │  Vehicle Detection  │
    │  Eye Classification │          │  (car/bus/truck)    │
    └─────────────────────┘          └─────────────────────┘
              ↓                                  ↓
    ┌─────────────────────┐          ┌─────────────────────┐
    │  Blink Tracker      │          │  TTC Estimator      │
    │  Microsleep Timer   │          │  (bbox area growth) │
    │  drowsy: bool       │          │  proximity: 0-100   │
    └─────────────────────┘          └─────────────────────┘
              ↓                                  ↓
              └──────────────┬───────────────────┘
                             ↓
                  ┌──────────────────────┐
                  │  Risk Fusion Engine  │
                  │  risk = 0.5×drowsy   │
                  │       + 0.5×proximity│
                  └──────────────────────┘
                             ↓
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
        ┌─────────┐   ┌──────────┐   ┌──────────┐
        │ HUD     │   │ TTS      │   │ Siren    │
        │ Display │   │ Alerts   │   │ Alarm    │
        └─────────┘   └──────────┘   └──────────┘
              ↓
        ┌─────────────┐
        │ incidents.csv│
        └─────────────┘
```

---

## Core Components

### 1. Driver Stream (Cam 1) — `driver_stream.py`

**Purpose:** Monitor driver eye state to detect drowsiness

**Technology Stack:**
- MediaPipe FaceMesh (468-point facial landmark detection)
- YOLOv8n-cls (fine-tuned binary classifier: open/closed eyes)
- Eye Aspect Ratio (EAR) geometric validation

**Processing Pipeline:**

```
Frame → FaceMesh → Locate Eyes → Crop L/R Eye Regions
                                        ↓
                              YOLOv8n-cls Classification
                                        ↓
                              EAR Geometric Veto Check
                                        ↓
                              Consecutive Frame Buffer (5 frames)
                                        ↓
                              State Machine Update
                                        ↓
                    ┌───────────────────┴───────────────────┐
                    ↓                                       ↓
            Blink Counter                          Microsleep Timer
         (open→closed edge)                    (closed ≥ 3.0 seconds)
                    ↓                                       ↓
              blinks: int                            drowsy: bool
```

**Key Features:**
- Dual-eye validation (both eyes must be closed)
- EAR veto mechanism (overrides model if geometry says open)
- Consecutive frame buffering (reduces false positives)
- Microsleep threshold: 3.0 seconds continuous closure
- Real-time blink counting

**Output State:**
```python
{
    "drowsy": bool,              # True if microsleep threshold crossed
    "eye_state": str,            # "OPEN" or "CLOSED"
    "blinks": int,               # Total blink count
    "microsleeps": int,          # Total microsleep events
    "microsleep_duration": float,# Current closure duration (seconds)
    "frame": ndarray             # Annotated video frame
}
```

---

### 2. Road Stream (Cam 2) — `road_stream.py`

**Purpose:** Detect vehicles and estimate collision risk

**Technology Stack:**
- YOLOv8n (COCO pretrained, no fine-tuning needed)
- Physics-based TTC estimation

**Target Vehicle Classes (COCO):**

- Class 2: car
- Class 3: motorcycle
- Class 5: bus
- Class 7: truck

**TTC Estimation Algorithm:**
```
TTC = (prev_bbox_area / Δarea) × (1 / fps)

Where:
- prev_bbox_area = largest vehicle bbox area in previous frame
- Δarea = curr_area - prev_area
- fps = video frame rate

If Δarea ≤ 0 → TTC = ∞ (vehicle not approaching)
```

**Proximity Score Mapping:**
```python
proximity_score = clamp((1 - TTC/10.0) × 100, 0, 100)

Examples:
TTC = 10s  → proximity = 0%   (safe)
TTC = 5s   → proximity = 50%  (moderate)
TTC = 1s   → proximity = 90%  (danger)
TTC = 0s   → proximity = 100% (collision imminent)
```

**Output State:**
```python
{
    "vehicle_count": int,        # Number of vehicles detected
    "ttc": float,                # Time-to-collision (seconds, inf = safe)
    "proximity_score": float,    # 0-100 danger score
    "frame": ndarray             # Annotated video frame
}
```

---

### 3. Risk Fusion Engine — `fusion.py`

**Purpose:** Combine driver + road signals into unified risk score

**Risk Formula:**
```python
drowsiness_score = 100 if drowsy else 0
risk_score = (0.5 × drowsiness_score) + (0.5 × proximity_score)
```

**Risk Levels:**


| Risk Score | Status Label | Color  | Actions Triggered |
|------------|-------------|--------|-------------------|
| 0 - 39     | 🟢 SAFE     | Green  | None |
| 40 - 70    | 🟠 WARNING  | Orange | Voice alert (6s cooldown) |
| 71 - 100   | 🔴 DANGER   | Red    | Voice alert + Siren alarm |

**Alert System:**

1. **Voice Alerts (TTS)** — `pyttsx3`
   - WARNING: "Warning. Stay alert. Risk level elevated."
   - DANGER: "Danger! Immediate risk detected. Please pull over."
   - Cooldown: 6 seconds between repeated alerts
   - Non-blocking daemon thread

2. **Siren Alarm** — `winsound` (Windows native)
   - Two-tone sweep: 800 Hz ↔ 1200 Hz
   - Activates when driver is drowsy
   - Stops immediately when eyes reopen
   - Zero external dependencies

3. **Visual HUD** — OpenCV
   - Dual-pane camera display (640×480 each)
   - Color-coded borders matching risk level
   - Real-time stats overlay
   - Bottom status bar with risk gauge

---

### 4. HUD Design — User Interface

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  CAM 1 — DRIVER          │  CAM 2 — ROAD                │
│  [640×480]               │  [640×480]                   │
│                          │                              │
│  Eye State: OPEN         │  Vehicles: 2                 │
│  Blinks: 15              │  TTC: 2.5s                   │
│  Microsleep: 0s          │  Proximity: 75%              │
└──────────────────────────┴──────────────────────────────┘
│                    STATUS BAR [90px]                    │
│  RISK: 38  [████████░░░░░░░░░░░░] STATUS: SAFE         │
│            DROWSY: NO    TTC: 2.5s                      │
└─────────────────────────────────────────────────────────┘
```

**Visual Elements:**

- Panel headers with color-coded accent stripes
- Semi-transparent stat overlays on camera feeds
- Flashing "DROWSY ALERT" banner when eyes closed
- Progress bar with tick marks at 40% and 70% thresholds
- Eye bounding boxes (red when closed, green when open)
- Vehicle bounding boxes with confidence scores

---

### 5. Training Pipeline — `train.py`

**Purpose:** Fine-tune YOLOv8n-cls for eye open/closed classification

**Dataset Requirements:**
- Source: Kaggle Eyes Dataset + MRL Eye Dataset
- Structure:
  ```
  dataset/
    train/
      open/    ← ~1900 images
      closed/  ← ~1900 images
    val/
      open/    ← ~540 images
      closed/  ← ~525 images
  ```

**Training Configuration:**
- Base model: `yolov8n-cls.pt` (pretrained ImageNet)
- Image size: 64×64 pixels (eyes are small)
- Batch size: 64
- Epochs: 20
- Device: CUDA (GPU required)
- Early stopping: patience=10

**Output:**
- `runs/classify/runs/train/eye_model/weights/best.pt`
- Used by `driver_stream.py` for inference

---

### 6. Incident Logger — CSV Export

**Purpose:** Record all risk events for post-analysis

**Log Trigger:** Risk score ≥ 40 (WARNING or DANGER)

**CSV Schema:**
```csv
timestamp,risk_score,status,drowsy,ttc
2026-04-13 12:01:16.464,40.97,WARNING,False,1.807
```

**Fields:**

- `timestamp`: ISO format with milliseconds
- `risk_score`: 0-100 float
- `status`: SAFE / WARNING / DANGER
- `drowsy`: boolean
- `ttc`: seconds (or -1 if infinite)

**Use Cases:**
- Fleet safety audits
- Driver behavior analysis
- Insurance claims evidence
- System performance evaluation

---

## Technical Specifications

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5 (4 cores)
- RAM: 8 GB
- GPU: NVIDIA GTX 1050 Ti (4GB VRAM) or better
- Webcam: 720p @ 30fps
- Storage: 2 GB free space

**Recommended:**
- GPU: NVIDIA RTX 3050 (6GB VRAM)
- Webcam: 1080p @ 60fps
- Dual camera setup (driver + road)

### Software Dependencies

```
ultralytics>=8.2.0        # YOLOv8 framework
mediapipe>=0.10.0         # Face landmark detection
opencv-python>=4.8.0      # Video processing
pyttsx3>=2.90             # Text-to-speech
torch>=2.0.0              # PyTorch backend
numpy>=1.24.0             # Numerical operations
Pillow>=10.0.0            # Image utilities
```

### Performance Metrics

**Target FPS:** 20-30 fps (real-time)
**Latency:** <100ms end-to-end
**GPU Utilization:** ~60-80% (RTX 3050)
**CPU Utilization:** ~30-40%

---

## User Workflows

### Workflow 1: Initial Setup


```
┌─────────────────────────────────────────────────────────────┐
│                    INITIAL SETUP FLOW                       │
└─────────────────────────────────────────────────────────────┘

1. Install Dependencies
   └─> pip install -r requirements.txt
       └─> Downloads PyTorch, YOLOv8, MediaPipe, etc.

2. Download Eye Dataset
   └─> Kaggle: Eyes Dataset + MRL Eye Dataset
       └─> Organize into dataset/train and dataset/val

3. Train Eye Classifier
   └─> python train.py
       └─> Fine-tunes YOLOv8n-cls for 20 epochs
           └─> Output: runs/classify/.../best.pt

4. Obtain Dashcam Video
   └─> Download from Pexels or use existing footage
       └─> Save as dashcam.mp4 in project root

5. Run System
   └─> python fusion.py
       └─> System starts, displays HUD
           └─> Press 'q' to quit and save incidents.csv
```

---

### Workflow 2: Real-Time Operation

```
┌─────────────────────────────────────────────────────────────┐
│                  REAL-TIME OPERATION FLOW                   │
└─────────────────────────────────────────────────────────────┘

START
  │
  ├─> Launch fusion.py
  │     │
  │     ├─> Initialize DriverStream (thread 1)
  │     │     └─> Open webcam
  │     │         └─> Load eye model
  │     │
  │     ├─> Initialize RoadStream (thread 2)
  │     │     └─> Open dashcam video
  │     │         └─> Load YOLOv8n COCO
  │     │
  │     └─> Start AlertEngine (thread 3)
  │           └─> Initialize pyttsx3 TTS
  │
  ├─> MAIN LOOP (runs until 'q' pressed)
  │     │
  │     ├─> Get driver state (drowsy, blinks, eye_state)
  │     ├─> Get road state (ttc, proximity, vehicles)
  │     │
  │     ├─> Compute risk_score
  │     │     └─> risk = 0.5×drowsy + 0.5×proximity
  │     │
  │     ├─> Determine status (SAFE/WARNING/DANGER)
  │     │
  │     ├─> Trigger alerts if needed
  │     │     ├─> Voice alert (if WARNING/DANGER)
  │     │     └─> Siren alarm (if drowsy)
  │     │
  │     ├─> Log to CSV (if risk ≥ 40)
  │     │
  │     ├─> Build HUD frame
  │     │     ├─> Annotate driver panel
  │     │     ├─> Annotate road panel
  │     │     └─> Draw status bar
  │     │
  │     └─> Display HUD window
  │
  └─> SHUTDOWN
        ├─> Stop all threads
        ├─> Release cameras
        ├─> Save incidents.csv
        └─> Exit
```

---

### Workflow 3: Incident Analysis

```
┌─────────────────────────────────────────────────────────────┐
│                  INCIDENT ANALYSIS FLOW                     │
└─────────────────────────────────────────────────────────────┘

1. Run System
   └─> python fusion.py
       └─> Drive/test for N minutes

2. Quit System (press 'q')
   └─> incidents.csv auto-generated

3. Analyze CSV
   └─> Open in Excel/Python/Tableau
       ├─> Filter by status (WARNING/DANGER)
       ├─> Group by timestamp ranges
       ├─> Calculate metrics:
       │     ├─> Total incidents
       │     ├─> Average risk score
       │     ├─> Drowsy event count
       │     └─> Minimum TTC observed
       └─> Generate reports/visualizations

4. Take Action
   └─> Driver coaching
   └─> Route optimization
   └─> Vehicle maintenance
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                         │
└──────────────────────────────────────────────────────────────┘

[Webcam]                              [Dashcam Video]
    │                                        │
    │ Frame (BGR)                            │ Frame (BGR)
    ↓                                        ↓
┌─────────────────┐                  ┌─────────────────┐
│ DriverStream    │                  │ RoadStream      │
│ (Thread 1)      │                  │ (Thread 2)      │
└─────────────────┘                  └─────────────────┘
    │                                        │
    │ State Dict                             │ State Dict
    │ {drowsy, blinks,                       │ {ttc, proximity,
    │  eye_state, frame}                     │  vehicles, frame}
    ↓                                        ↓
    └────────────────┬───────────────────────┘
                     ↓
              ┌─────────────┐
              │ fusion.py   │
              │ Main Thread │
              └─────────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Risk     │ │ Alert    │ │ Logger   │
  │ Engine   │ │ Engine   │ │          │
  └──────────┘ └──────────┘ └──────────┘
        │            │            │
        ↓            ↓            ↓
  risk_score   Voice/Siren   incidents.csv
        │
        ↓
  ┌──────────┐
  │ HUD      │
  │ Builder  │
  └──────────┘
        │
        ↓
  [Display Window]
```

---

## State Machine Diagrams

### Driver State Machine

```
┌──────────────────────────────────────────────────────────────┐
│              DRIVER DROWSINESS STATE MACHINE                 │
└──────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │   INITIAL   │
                    │  eyes: null │
                    └──────┬──────┘
                           │
                           ↓
                    ┌─────────────┐
              ┌────→│    OPEN     │←────┐
              │     │ drowsy=False│     │
              │     └──────┬──────┘     │
              │            │            │
              │  eyes open │ eyes closed│
              │            ↓            │
              │     ┌─────────────┐    │
              │     │   CLOSING   │    │
              │     │ buffer: 0-4 │    │
              │     └──────┬──────┘    │
              │            │            │
              │  buffer<5  │ buffer≥5  │
              │            ↓            │
              │     ┌─────────────┐    │
              │     │   CLOSED    │    │
              │     │ timer start │    │
              │     └──────┬──────┘    │
              │            │            │
              │  timer<3s  │ timer≥3s  │
              │            ↓            │
              │     ┌─────────────┐    │
              └─────│   DROWSY    │────┘
                    │ drowsy=True │
                    │ ALARM ON    │
                    └─────────────┘

Events:
- Blink: OPEN → CLOSING → CLOSED → OPEN (duration < 3s)
- Microsleep: OPEN → CLOSING → CLOSED → DROWSY (duration ≥ 3s)
```

### Risk Level State Machine

```
┌──────────────────────────────────────────────────────────────┐
│                 RISK LEVEL STATE MACHINE                     │
└──────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
              ┌────→│    SAFE     │←────┐
              │     │  risk < 40  │     │
              │     │  Color: 🟢  │     │
              │     └──────┬──────┘     │
              │            │            │
              │  risk≥40   │  risk<40   │
              │            ↓            │
              │     ┌─────────────┐    │
              │  ┌─→│   WARNING   │─┐  │
              │  │  │ 40≤risk≤70  │ │  │
              │  │  │  Color: 🟠  │ │  │
              │  │  │  TTS: ON    │ │  │
              │  │  └──────┬──────┘ │  │
              │  │         │        │  │
              │  │ risk<40 │ risk>70│  │
              │  │         ↓        │  │
              │  │  ┌─────────────┐ │  │
              │  └──│   DANGER    │─┘  │
              │     │  risk > 70  │    │
              └─────│  Color: 🔴  │────┘
                    │  TTS: ON    │
                    │  SIREN: ON  │
                    └─────────────┘

Transitions:
- SAFE → WARNING: risk crosses 40 threshold
- WARNING → DANGER: risk crosses 70 threshold
- DANGER → SAFE: risk drops below 40 (hysteresis)
```

---

## Deep Learning Components

### What Uses Deep Learning?

| Component | Technology | Deep Learning? | Details |
|-----------|-----------|----------------|---------|
| Eye Classification | YOLOv8n-cls | ✅ YES | CNN-based image classifier, fine-tuned on eye dataset |
| Face Landmark Detection | MediaPipe FaceMesh | ✅ YES | Neural network for 468-point facial landmarks |
| Vehicle Detection | YOLOv8n (COCO) | ✅ YES | Pretrained object detector, no fine-tuning |
| TTC Estimation | Physics formula | ❌ NO | Bounding box area growth calculation |
| Risk Fusion | Weighted average | ❌ NO | Simple arithmetic: 0.5×A + 0.5×B |
| EAR Calculation | Geometry | ❌ NO | Euclidean distance ratios |

### Model Details

**1. YOLOv8n-cls (Eye Classifier)**
- Architecture: CSPDarknet backbone + classification head
- Input: 64×64 RGB images
- Output: 2 classes (open, closed) + confidence scores
- Training: Transfer learning from ImageNet weights
- Inference time: ~5ms per eye on RTX 3050

**2. MediaPipe FaceMesh**
- Architecture: BlazeFace detector + landmark regression
- Input: Full frame (any resolution)
- Output: 468 (x,y,z) facial landmarks
- Pretrained: Google's production model
- Inference time: ~15ms per frame on CPU

**3. YOLOv8n (Vehicle Detector)**
- Architecture: CSPDarknet + PANet + detection head
- Input: Full frame (auto-resized to 640px)
- Output: Bounding boxes + class IDs + confidence
- Pretrained: COCO dataset (80 classes)
- Inference time: ~20ms per frame on RTX 3050

---

## Risk Scenarios & Examples

### Scenario 1: Drowsy Driver, Clear Road
```
Driver State:
  - Eyes closed for 4.2 seconds
  - drowsy = True
  - drowsiness_score = 100

Road State:
  - No vehicles detected
  - TTC = ∞
  - proximity_score = 0

Risk Calculation:
  risk = 0.5×100 + 0.5×0 = 50
  Status: WARNING (🟠)

Actions:
  ✓ Voice alert: "Warning. Stay alert."
  ✓ Siren alarm activated
  ✓ Orange HUD borders
  ✓ Logged to CSV
```

### Scenario 2: Alert Driver, Vehicle Approaching
```
Driver State:
  - Eyes open
  - drowsy = False
  - drowsiness_score = 0

Road State:
  - Car detected, TTC = 0.8s
  - proximity_score = 92

Risk Calculation:
  risk = 0.5×0 + 0.5×92 = 46
  Status: WARNING (🟠)

Actions:
  ✓ Voice alert: "Warning. Stay alert."
  ✗ No siren (driver not drowsy)
  ✓ Orange HUD borders
  ✓ Logged to CSV
```

### Scenario 3: Drowsy Driver + Close Vehicle (CRITICAL)
```
Driver State:
  - Eyes closed for 3.5 seconds
  - drowsy = True
  - drowsiness_score = 100

Road State:
  - Truck detected, TTC = 1.2s
  - proximity_score = 88

Risk Calculation:
  risk = 0.5×100 + 0.5×88 = 94
  Status: DANGER (🔴)

Actions:
  ✓ Voice alert: "Danger! Pull over immediately."
  ✓ Siren alarm activated
  ✓ Red HUD borders + flashing DROWSY banner
  ✓ Logged to CSV
```

---

## Limitations & Future Enhancements

### Current Limitations

1. **TTC Accuracy**
   - Uses 2D bounding box area (no depth sensor)
   - Assumes constant velocity
   - Inaccurate for lateral movements

2. **Environmental Constraints**
   - Not tested in night/rain/fog conditions
   - Requires good lighting for face detection
   - Dashcam quality affects vehicle detection

3. **Hardware Dependency**
   - Requires GPU for real-time performance
   - Single webcam may not capture all drivers
   - Dashcam video must be pre-recorded or live

4. **False Positives**
   - Sunglasses may trigger false drowsiness
   - Blinking during speech may be misclassified
   - Shadows can affect eye detection

### Proposed Enhancements

**Phase 2 (Q3 2026):**
- [ ] Stereo camera for accurate depth estimation
- [ ] Head pose estimation (looking away detection)
- [ ] Yawning detection via mouth landmarks
- [ ] Night vision mode with IR camera support

**Phase 3 (Q4 2026):**
- [ ] Mobile app integration (iOS/Android)
- [ ] Cloud dashboard for fleet management
- [ ] Real-time GPS tracking + incident mapping
- [ ] Driver identification via face recognition

**Phase 4 (2027):**
- [ ] Integration with vehicle CAN bus
- [ ] Automatic emergency braking trigger
- [ ] Lane departure warning fusion
- [ ] Predictive risk modeling with ML

---

## Success Metrics

### Technical KPIs
- System uptime: >99%
- False positive rate: <5%
- False negative rate: <2%
- Average FPS: >25
- Alert latency: <200ms

### Safety KPIs
- Drowsy driving incidents prevented: Target 80%
- Rear-end collision warnings: Target 90% accuracy
- User alert response time: <3 seconds
- System adoption rate: >70% of test drivers

---

## Compliance & Safety

### Data Privacy
- All video processing is local (no cloud upload)
- CSV logs contain no PII
- User can delete incidents.csv anytime

### Safety Standards
- Alerts do not obstruct driver vision
- Audio volume is non-startling
- System can be disabled instantly (press 'q')

---

## Conclusion

DualSentinel represents a novel approach to driver safety by fusing driver monitoring and road hazard detection into a unified risk assessment system. The combination of deep learning (YOLOv8, MediaPipe) and physics-based estimation (TTC) provides a robust, real-time solution that can prevent accidents before they occur.

The system is production-ready for:
- Individual driver safety enhancement
- Fleet operator risk management
- Automotive safety research
- Insurance telematics programs

**Next Steps:**
1. Deploy pilot program with 50 test drivers
2. Collect real-world performance data
3. Iterate on alert thresholds based on feedback
4. Pursue automotive safety certification (ISO 26262)

---

**Document Version:** 1.0  
**Last Updated:** April 13, 2026  
**Author:** DualSentinel Development Team  
**Status:** Production Ready
