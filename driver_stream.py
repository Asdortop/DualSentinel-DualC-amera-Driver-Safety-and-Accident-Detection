"""
DualSentinel — driver_stream.py (Cam 1)
========================================
Uses MediaPipe FaceMesh to locate eyes on the driver's face,
crops each eye region, and classifies them as open/closed using
a fine-tuned YOLOv8n-cls model.

State machine tracks:
  • blink count       (open → closed transition)
  • microsleep timer  (eyes continuously closed ≥ MICROSLEEP_THRESHOLD)
  • drowsy flag       (True when microsleep threshold is crossed)

Runs in a background daemon thread. Call get_state() to retrieve
the latest results from the main thread.

Adapted from: github.com/tyrerodr/real-time-drowsy-driving-detection
Modified for DualSentinel: removed PyQt5/winsound, outputs drowsy bool
for fusion engine.
"""

import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from pathlib import Path
from ultralytics import YOLO

# ── MediaPipe FaceMesh landmark indices for each eye ──────────────────────────
# 468-point model; these give the full eye contour
LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249,
                  263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDX = [33,  7,  163, 144, 145, 153, 154, 155,
                  133, 173, 157, 158, 159, 160, 161, 246]

# ── EAR landmark indices (6 points per eye for aspect ratio) ─────────────────
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# Left eye  → p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
# Right eye → p1=33,  p2=160, p3=158, p4=133, p5=153, p6=144
LEFT_EAR_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_IDX = [33,  160, 158, 133, 153, 144]

# ── Thresholds ────────────────────────────────────────────────────────────────
MICROSLEEP_THRESHOLD   = 3.0   # seconds of continuous eye closure → drowsy
EYE_PADDING            = 12    # pixel padding around cropped eye region
CLOSED_CLASS_IDX       = 0     # YOLOv8-cls sorted alphabetically: closed=0, open=1
EYE_CONF_THRESHOLD     = 0.72  # raised: harder to trigger 'closed' (was 0.55)
EAR_OPEN_VETO          = 0.20  # if EAR > this, eye is geometrically open → override model
CLOSED_FRAMES_REQUIRED = 5     # consecutive 'closed' frames needed before timer starts


class DriverStream:
    """
    Cam 1 driver-facing stream.

    Usage:
        stream = DriverStream(model_path="runs/train/eye_model/weights/best.pt")
        stream.start()
        state = stream.get_state()   # {'drowsy', 'eye_state', 'blinks',
                                     #  'microsleep_duration', 'frame'}
        stream.stop()
    """

    def __init__(self, model_path: str, cam_index: int = 0):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Eye model not found at '{model_path}'.\n"
                f"Run  python train.py  first to generate it."
            )

        # ── Models ────────────────────────────────────────────────────────────
        self.eye_model = YOLO(str(model_path))
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces        = 1,
            refine_landmarks     = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )

        # ── Capture ───────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {cam_index}")

        # ── Shared state (written in bg thread, read in main thread) ──────────
        self._lock               = threading.Lock()
        self._running            = False
        self._drowsy             = False
        self._eye_state          = "OPEN"
        self._blinks             = 0
        self._microsleeps        = 0
        self._microsleep_dur     = 0.0
        self._annotated_frame    = None

        # ── Internal state machine vars ───────────────────────────────────────
        self._closed_since       = None   # timestamp when eyes first closed
        self._was_closed         = False  # previous frame's state
        self._consec_closed      = 0      # consecutive 'closed' frame counter

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_ear(self, landmarks, indices: list, w: int, h: int) -> float:
        """
        Compute Eye Aspect Ratio from 6 landmark points.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        Returns 0.0 if geometry is degenerate.
        """
        pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
            dtype=np.float32
        )
        p1, p2, p3, p4, p5, p6 = pts
        vertical   = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return 0.0
        return float(vertical / (2.0 * horizontal))

    def _crop_eye(self, frame: np.ndarray, landmarks, indices: list):
        """Return cropped eye region and its bounding box (x1,y1,x2,y2)."""
        h, w = frame.shape[:2]
        pts = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        )
        x1 = max(0, pts[:, 0].min() - EYE_PADDING)
        y1 = max(0, pts[:, 1].min() - EYE_PADDING)
        x2 = min(w, pts[:, 0].max() + EYE_PADDING)
        y2 = min(h, pts[:, 1].max() + EYE_PADDING)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _classify_eye(self, crop: np.ndarray) -> tuple[str, float]:
        """Run YOLOv8-cls on a cropped eye. Returns (state, confidence)."""
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return "OPEN", 0.0
        results = self.eye_model(crop, verbose=False, imgsz=64)
        probs   = results[0].probs
        cls_idx = probs.top1
        conf    = float(probs.top1conf)
        state   = "CLOSED" if (cls_idx == CLOSED_CLASS_IDX and conf >= EYE_CONF_THRESHOLD) else "OPEN"
        return state, conf

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run full pipeline on one frame. Updates shared state. Returns annotated frame."""
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        out     = frame.copy()
        h, w    = frame.shape[:2]

        if not results.multi_face_landmarks:
            # No face detected — reset buffer, treat as open/safe
            self._consec_closed = 0
            self._update_state("OPEN", out)
            return out

        lm = results.multi_face_landmarks[0].landmark

        # ── EAR veto: compute geometric openness for each eye ──────────────────
        l_ear = self._compute_ear(lm, LEFT_EAR_IDX,  w, h)
        r_ear = self._compute_ear(lm, RIGHT_EAR_IDX, w, h)
        avg_ear = (l_ear + r_ear) / 2.0
        ear_says_open = avg_ear > EAR_OPEN_VETO   # geometrically open

        # ── Crop and classify both eyes via model ──────────────────────────────
        l_crop, l_bbox = self._crop_eye(frame, lm, LEFT_EYE_IDX)
        r_crop, r_bbox = self._crop_eye(frame, lm, RIGHT_EYE_IDX)

        l_state, l_conf = self._classify_eye(l_crop)
        r_state, r_conf = self._classify_eye(r_crop)

        # ── Decision: all three checks must agree to call 'CLOSED' ─────────────
        model_says_closed = (l_state == "CLOSED") and (r_state == "CLOSED")

        if model_says_closed and ear_says_open:
            # EAR veto: model thinks closed but geometry says open → trust geometry
            both_closed = False
        else:
            both_closed = model_says_closed

        # ── Consecutive-frame buffer ───────────────────────────────────────────
        if both_closed:
            self._consec_closed += 1
        else:
            self._consec_closed = 0

        # Only enter 'CLOSED' state after N consecutive frames
        confirmed_closed = (self._consec_closed >= CLOSED_FRAMES_REQUIRED)
        eye_state = "CLOSED" if confirmed_closed else "OPEN"

        # ── Draw eye bounding boxes + EAR overlay ─────────────────────────────
        box_color = (0, 0, 230) if confirmed_closed else (0, 230, 80)
        for bbox in [l_bbox, r_bbox]:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        # EAR readout on frame
        ear_color = (0, 200, 255) if ear_says_open else (0, 80, 200)
        cv2.putText(out, f"EAR:{avg_ear:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2, cv2.LINE_AA)
        cv2.putText(out, f"EYE:{eye_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)

        self._update_state(eye_state, out)
        return out

    def _update_state(self, eye_state: str, frame: np.ndarray):
        """Update the blink / microsleep state machine and shared state."""
        now         = time.time()
        both_closed = (eye_state == "CLOSED")
        drowsy      = False
        microsleep_dur = 0.0

        if both_closed:
            if self._closed_since is None:
                self._closed_since = now
                if not self._was_closed:
                    # Rising edge: new blink/closure started
                    with self._lock:
                        self._blinks += 1
            elapsed = now - self._closed_since
            microsleep_dur = elapsed
            drowsy = elapsed >= MICROSLEEP_THRESHOLD
        else:
            if self._closed_since is not None:
                # Falling edge: closure ended
                elapsed = now - self._closed_since
                if elapsed >= MICROSLEEP_THRESHOLD:
                    with self._lock:
                        self._microsleeps += 1
            self._closed_since = None

        self._was_closed = both_closed

        with self._lock:
            self._drowsy          = drowsy
            self._eye_state       = eye_state
            self._microsleep_dur  = round(microsleep_dur, 2)
            self._annotated_frame = frame

    # ── Thread worker ─────────────────────────────────────────────────────────

    def _run(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self._process_frame(frame)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start background capture + inference thread."""
        self._running = True
        t = threading.Thread(target=self._run, name="DriverStream", daemon=True)
        t.start()

    def stop(self):
        """Stop stream and release webcam."""
        self._running = False
        time.sleep(0.1)
        self.cap.release()

    def get_state(self) -> dict:
        """
        Thread-safe snapshot of current driver state.
        Returns:
            drowsy             (bool)  — True if microsleep threshold crossed
            eye_state          (str)   — "OPEN" or "CLOSED"
            blinks             (int)   — total blink count this session
            microsleeps        (int)   — total microsleep events
            microsleep_duration (float) — current closed duration in seconds
            frame              (ndarray | None) — latest annotated frame
        """
        with self._lock:
            frame = self._annotated_frame
            return {
                "drowsy"             : self._drowsy,
                "eye_state"          : self._eye_state,
                "blinks"             : self._blinks,
                "microsleeps"        : self._microsleeps,
                "microsleep_duration": self._microsleep_dur,
                "frame"              : frame.copy() if frame is not None else None,
            }
