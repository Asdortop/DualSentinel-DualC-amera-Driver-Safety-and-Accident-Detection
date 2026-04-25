"""
DualSentinel — road_stream.py (Cam 2)
======================================
Uses pretrained YOLOv8n (COCO weights) to detect vehicles in a dashcam feed.
No fine-tuning required — COCO already contains all target vehicle classes.

Detects:
  • car (COCO class 2)
  • motorcycle (COCO class 3)
  • bus (COCO class 5)
  • truck (COCO class 7)

Computes per-frame TTC (Time-To-Collision) via bounding box area growth:
  TTC = (prev_area / delta_area) × (1 / fps)

Outputs proximity_score (0–100) to the fusion engine.

Runs in a background daemon thread. Call get_state() to retrieve latest results.
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
VEHICLE_CLASSES   = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF_THRESHOLD    = 0.40     # minimum confidence for a valid detection
TTC_MAX_SECONDS   = 10.0    # TTC beyond this = proximity_score of 0
TTC_MIN_SECONDS   = 0.5     # TTC below this  = proximity_score of 100 (clamped)
COCO_WEIGHTS      = "yolov8n.pt"   # auto-downloaded if not present

# Box colours (BGR)
VEHICLE_COLOR     = (0, 200, 255)   # cyan-yellow for vehicle boxes
LABEL_BG_COLOR    = (0, 140, 180)


class RoadStream:
    """
    Cam 2 road-facing stream.

    Usage:
        stream = RoadStream(video_source="dashcam.mp4")
        stream.start()
        state = stream.get_state()   # {'vehicle_count', 'ttc',
                                     #  'proximity_score', 'frame'}
        stream.stop()

    video_source can be:
        • path to an mp4 file  (e.g. "dashcam.mp4")
        • integer webcam index (e.g. 1 for a second webcam)
    """

    def __init__(self, video_source="dashcam.mp4"):
        self.model      = YOLO(COCO_WEIGHTS)   # downloads ~6 MB on first run
        self.source     = video_source

        self.cap        = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source '{video_source}'.\n"
                f"Provide a valid dashcam.mp4 path or webcam index."
            )

        detected_fps    = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps        = detected_fps if detected_fps > 0 else 30.0

        # ── Shared state ──────────────────────────────────────────────────────
        self._lock            = threading.Lock()
        self._running         = False
        self._vehicle_count   = 0
        self._ttc             = float("inf")
        self._proximity_score = 0.0
        self._annotated_frame = None

        # ── TTC rolling state ─────────────────────────────────────────────────
        self._prev_max_area   = 0.0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_ttc(self, curr_max_area: float) -> float:
        """
        Physics-based TTC approximation using bounding box area growth.
        Returns float('inf') if the vehicle is not approaching.
        """
        prev = self._prev_max_area
        self._prev_max_area = curr_max_area

        if prev <= 0 or curr_max_area <= prev:
            # No vehicle detected or vehicle moving away → no danger
            return float("inf")

        delta = curr_max_area - prev
        ttc   = (prev / delta) * (1.0 / self.fps)
        return max(0.0, ttc)

    def _area_to_proximity(self, ttc: float) -> float:
        """Map TTC → proximity_score in [0, 100]."""
        if ttc == float("inf"):
            return 0.0
        score = (1.0 - ttc / TTC_MAX_SECONDS) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLOv8 inference, compute TTC, update shared state."""
        results     = self.model(
            frame,
            verbose = False,
            classes = list(VEHICLE_CLASSES.keys()),
            conf    = CONF_THRESHOLD,
        )
        out         = frame.copy()
        boxes       = results[0].boxes
        max_area    = 0.0
        count       = 0

        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area   = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                count += 1

                # Draw bounding box
                label = f"{VEHICLE_CLASSES.get(cls_id, 'vehicle')} {conf:.2f}"
                cv2.rectangle(out, (x1, y1), (x2, y2), VEHICLE_COLOR, 2)

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), LABEL_BG_COLOR, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        ttc   = self._compute_ttc(max_area)
        prox  = self._area_to_proximity(ttc)

        # Annotate TTC on frame
        ttc_text = f"TTC: {ttc:.1f}s" if ttc != float("inf") else "TTC: --"
        cv2.putText(out, ttc_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, VEHICLE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(out, f"Vehicles: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

        with self._lock:
            self._vehicle_count   = count
            self._ttc             = ttc
            self._proximity_score = round(prox, 2)
            self._annotated_frame = out

        return out

    # ── Thread worker ─────────────────────────────────────────────────────────

    def _run(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                # Loop dashcam video from beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._prev_max_area = 0.0   # reset TTC on loop
                continue
            self._process_frame(frame)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start background video capture + inference thread."""
        self._running = True
        t = threading.Thread(target=self._run, name="RoadStream", daemon=True)
        t.start()

    def stop(self):
        """Stop stream and release video capture."""
        self._running = False
        time.sleep(0.1)
        self.cap.release()

    def get_state(self) -> dict:
        """
        Thread-safe snapshot of current road state.
        Returns:
            vehicle_count   (int)   — number of vehicles detected this frame
            ttc             (float) — time-to-collision in seconds (inf = safe)
            proximity_score (float) — 0–100 danger score from TTC
            frame           (ndarray | None) — latest annotated frame
        """
        with self._lock:
            frame = self._annotated_frame
            return {
                "vehicle_count"  : self._vehicle_count,
                "ttc"            : self._ttc,
                "proximity_score": self._proximity_score,
                "frame"          : frame.copy() if frame is not None else None,
            }
