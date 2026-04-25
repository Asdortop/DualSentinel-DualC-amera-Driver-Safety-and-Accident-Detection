"""
DualSentinel — fusion.py
=========================
Main real-time pipeline. Fuses Cam 1 (driver) and Cam 2 (road) streams
into a unified risk score, renders a dual-pane HUD, triggers voice alerts,
and logs incidents to CSV.

Usage:
    python fusion.py [--cam CAM_INDEX] [--video DASHCAM_PATH]

Defaults:
    --cam   0            (first webcam, driver-facing)
    --video dashcam.mp4  (road/dashcam footage)

Press 'q' to quit and save incidents.csv

Architecture:
    DriverStream (thread) ─┐
                           ├── Risk Fusion Engine ── HUD ── Display
    RoadStream   (thread) ─┘                     ├── TTS alert (thread)
                                                 └── CSV log
"""

import cv2
import numpy as np
import csv
import queue
import threading
import time
import argparse
import winsound
from datetime import datetime
from pathlib import Path

import pyttsx3

from driver_stream import DriverStream
from road_stream   import RoadStream

# ── Config ────────────────────────────────────────────────────────────────────
EYE_MODEL_PATH   = "runs/classify/runs/train/eye_model/weights/best.pt"
DASHCAM_DEFAULT  = "dashcam.mp4"
CAM_DEFAULT      = 0

FRAME_W          = 640     # width of each camera panel
FRAME_H          = 480     # height of each camera panel
STATUS_BAR_H     = 90      # height of bottom status bar

DANGER_THRESHOLD  = 70.0
WARNING_THRESHOLD = 40.0

ALERT_COOLDOWN    = 6.0    # seconds between repeated voice alerts
CSV_LOG_THRESHOLD = WARNING_THRESHOLD   # only log frames above this risk

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
CLR_BG       = (18,  18,  28)    # dark background
CLR_SAFE     = (80,  220, 100)   # green
CLR_WARNING  = (0,   165, 255)   # orange
CLR_DANGER   = (50,  50,  230)   # red
CLR_ACCENT   = (255, 200, 80)    # amber for labels
CLR_WHITE    = (240, 240, 240)
CLR_GREY     = (130, 130, 140)
CLR_PANEL_BG = (22,  22,  35)    # panel header bg

# ── Risk Engine ───────────────────────────────────────────────────────────────

def compute_risk(drowsy: bool, proximity_score: float) -> tuple[float, str, tuple]:
    """
    Returns (risk_score, label, color).
    Formula from PRD:
        drowsiness_score = 100 if drowsy else 0
        risk_score       = 0.5 * drowsiness_score + 0.5 * proximity_score
    """
    drowsiness_score = 100.0 if drowsy else 0.0
    risk_score       = 0.5 * drowsiness_score + 0.5 * proximity_score
    risk_score       = float(np.clip(risk_score, 0.0, 100.0))

    if risk_score > DANGER_THRESHOLD:
        return risk_score, "DANGER",  CLR_DANGER
    elif risk_score > WARNING_THRESHOLD:
        return risk_score, "WARNING", CLR_WARNING
    else:
        return risk_score, "SAFE",    CLR_SAFE


# ── HUD Builders ──────────────────────────────────────────────────────────────

def _make_placeholder(text: str) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W, 3), CLR_BG, dtype=np.uint8)
    cv2.putText(frame, text, (FRAME_W // 2 - 80, FRAME_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_GREY, 2, cv2.LINE_AA)
    return frame


def draw_panel_header(frame: np.ndarray, title: str, color: tuple):
    """Draw a coloured header bar with title text at top of camera panel."""
    cv2.rectangle(frame, (0, 0), (FRAME_W, 36), CLR_PANEL_BG, -1)
    cv2.rectangle(frame, (0, 0), (4, 36), color, -1)   # left accent stripe
    cv2.putText(frame, title, (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 2, cv2.LINE_AA)


def draw_stat(frame: np.ndarray, label: str, value: str,
              x: int, y: int, val_color: tuple = CLR_WHITE):
    """Draw a label: value pair with styled colours."""
    cv2.putText(frame, f"{label}:", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, value, (x + 130, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, val_color, 2, cv2.LINE_AA)


def annotate_driver_panel(frame: np.ndarray, ds: dict, risk_color: tuple) -> np.ndarray:
    """Overlay driver stats on the Cam 1 frame."""
    draw_panel_header(frame, "CAM 1 — DRIVER", risk_color)

    # Semi-transparent stats box at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, FRAME_H - 90), (FRAME_W, FRAME_H), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    eye_color = CLR_DANGER if ds["eye_state"] == "CLOSED" else CLR_SAFE
    draw_stat(frame, "Eye State",    ds["eye_state"],                   10, FRAME_H - 70, eye_color)
    draw_stat(frame, "Blinks",       str(ds["blinks"]),                 10, FRAME_H - 50)
    draw_stat(frame, "Microsleep",   f"{ds['microsleep_duration']}s",   10, FRAME_H - 30,
              CLR_DANGER if ds["drowsy"] else CLR_WHITE)

    if ds["drowsy"]:
        # Flashing DROWSY banner
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (FRAME_W // 4, FRAME_H // 2 - 30),
                          (3 * FRAME_W // 4, FRAME_H // 2 + 30), CLR_DANGER, -1)
            cv2.putText(frame, "DROWSY ALERT", (FRAME_W // 4 + 10, FRAME_H // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, CLR_WHITE, 3, cv2.LINE_AA)
    return frame


def annotate_road_panel(frame: np.ndarray, rs: dict, risk_color: tuple) -> np.ndarray:
    """Overlay road stats on the Cam 2 frame."""
    draw_panel_header(frame, "CAM 2 — ROAD", risk_color)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, FRAME_H - 90), (FRAME_W, FRAME_H), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    ttc_val   = rs["ttc"]
    ttc_str   = f"{ttc_val:.1f}s" if ttc_val != float("inf") else "Clear"
    ttc_color = CLR_DANGER if rs["proximity_score"] > 60 else (
                CLR_WARNING if rs["proximity_score"] > 30 else CLR_SAFE)

    draw_stat(frame, "Vehicles",    str(rs["vehicle_count"]),          10, FRAME_H - 70)
    draw_stat(frame, "TTC",         ttc_str,                           10, FRAME_H - 50, ttc_color)
    draw_stat(frame, "Proximity",   f"{rs['proximity_score']:.0f}%",   10, FRAME_H - 30, ttc_color)
    return frame


def build_status_bar(risk_score: float, label: str, risk_color: tuple,
                     ds: dict, rs: dict) -> np.ndarray:
    """Build the bottom status bar showing risk score and progress bar."""
    bar  = np.full((STATUS_BAR_H, FRAME_W * 2, 3), CLR_BG, dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (FRAME_W * 2, 2), risk_color, -1)   # top divider

    # ── Risk score number ──
    score_str = f"{int(risk_score):3d}"
    cv2.putText(bar, "RISK", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_GREY, 1, cv2.LINE_AA)
    cv2.putText(bar, score_str, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, risk_color, 3, cv2.LINE_AA)

    # ── Progress bar ──
    bar_x, bar_y     = 120, 25
    bar_w, bar_h_px  = 650, 28
    fill_w           = int(bar_w * risk_score / 100)

    cv2.rectangle(bar, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (40, 40, 55), -1)
    if fill_w > 0:
        cv2.rectangle(bar, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h_px), risk_color, -1)
    cv2.rectangle(bar, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (60, 60, 80), 1)

    # Tick marks at 40 and 70
    for pct in [40, 70]:
        tx = bar_x + int(bar_w * pct / 100)
        cv2.line(bar, (tx, bar_y - 4), (tx, bar_y + bar_h_px + 4), (80, 80, 100), 1)

    # ── Status label ──
    lbl_x = bar_x + bar_w + 20
    cv2.putText(bar, "STATUS", (lbl_x, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_GREY, 1, cv2.LINE_AA)
    cv2.putText(bar, label, (lbl_x, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, risk_color, 3, cv2.LINE_AA)

    # ── Drowsy + TTC quick stats on right ──
    r_x = lbl_x + 200
    drowsy_str = "YES" if ds["drowsy"] else "NO"
    drowsy_clr = CLR_DANGER if ds["drowsy"] else CLR_SAFE
    ttc_str    = f"{rs['ttc']:.1f}s" if rs["ttc"] != float("inf") else "--"

    cv2.putText(bar, f"DROWSY: {drowsy_str}", (r_x, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, drowsy_clr, 2, cv2.LINE_AA)
    cv2.putText(bar, f"TTC: {ttc_str}", (r_x, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_ACCENT, 2, cv2.LINE_AA)

    return bar


def build_hud(ds: dict, rs: dict, risk_score: float, label: str,
              risk_color: tuple) -> np.ndarray:
    """Compose the full HUD frame from both camera feeds + status bar."""
    # ── Grab and resize frames ──
    driver_frame = ds.get("frame")
    road_frame   = rs.get("frame")

    if driver_frame is None:
        driver_frame = _make_placeholder("Waiting for Cam 1...")
    else:
        driver_frame = cv2.resize(driver_frame, (FRAME_W, FRAME_H))

    if road_frame is None:
        road_frame = _make_placeholder("Waiting for Cam 2...")
    else:
        road_frame = cv2.resize(road_frame, (FRAME_W, FRAME_H))

    # ── Coloured border around each panel based on risk ──
    bsize = 4
    cv2.rectangle(driver_frame, (bsize, bsize),
                  (FRAME_W - bsize, FRAME_H - bsize), risk_color, bsize)
    cv2.rectangle(road_frame, (bsize, bsize),
                  (FRAME_W - bsize, FRAME_H - bsize), risk_color, bsize)

    # ── Overlay stats ──
    driver_frame = annotate_driver_panel(driver_frame, ds, risk_color)
    road_frame   = annotate_road_panel(road_frame, rs, risk_color)

    # ── Stack horizontally ──
    top_row = np.hstack([driver_frame, road_frame])

    # ── Status bar ──
    status_bar = build_status_bar(risk_score, label, risk_color, ds, rs)

    return np.vstack([top_row, status_bar])


# ── TTS Alert System ──────────────────────────────────────────────────────────

class AlertEngine:
    """Non-blocking pyttsx3 TTS running in a daemon thread."""

    MESSAGES = {
        "DANGER" : "Danger! Immediate risk detected. Please pull over.",
        "WARNING": "Warning. Stay alert. Risk level elevated.",
    }

    def __init__(self):
        self._q              = queue.Queue(maxsize=1)
        self._last_alert_t   = 0.0
        self._last_label     = ""
        t = threading.Thread(target=self._worker, name="TTS", daemon=True)
        t.start()

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        while True:
            try:
                msg = self._q.get(timeout=1)
                engine.say(msg)
                engine.runAndWait()
            except queue.Empty:
                pass

    def maybe_alert(self, label: str):
        """Trigger an alert if label changed or cooldown expired."""
        now = time.time()
        if label == "SAFE":
            self._last_label = label
            return
        cooldown_ok  = (now - self._last_alert_t) >= ALERT_COOLDOWN
        label_change = (label != self._last_label)

        if cooldown_ok or label_change:
            msg = self.MESSAGES.get(label)
            if msg and self._q.empty():
                try:
                    self._q.put_nowait(msg)
                    self._last_alert_t = now
                    self._last_label   = label
                except queue.Full:
                    pass


# ── Siren Alarm Engine ───────────────────────────────────────────────────────

class AlarmEngine:
    """
    Plays a repeating two-tone siren (800 Hz ↔ 1200 Hz) in a daemon thread
    whenever the driver is detected as drowsy. Stops immediately on deactivation.
    Uses winsound (Windows built-in) — zero extra dependencies.
    """

    FREQ_HIGH  = 1200   # Hz
    FREQ_LOW   = 800    # Hz
    TONE_MS    = 280    # duration of each tone in milliseconds

    def __init__(self):
        self._active   = threading.Event()
        self._stop     = threading.Event()
        t = threading.Thread(target=self._worker, name="Alarm", daemon=True)
        t.start()

    def _worker(self):
        while not self._stop.is_set():
            if self._active.wait(timeout=0.1):
                # Play high → low sweep while still active
                try:
                    winsound.Beep(self.FREQ_HIGH, self.TONE_MS)
                    if self._active.is_set():
                        winsound.Beep(self.FREQ_LOW, self.TONE_MS)
                except Exception:
                    pass   # swallow any audio errors

    def activate(self):
        """Start siren — call every frame while drowsy."""
        self._active.set()

    def deactivate(self):
        """Stop siren — call when no longer drowsy."""
        self._active.clear()

    def shutdown(self):
        self._stop.set()


# ── CSV Incident Logger ───────────────────────────────────────────────────────

class IncidentLogger:
    def __init__(self):
        self._rows = []

    def record(self, risk_score: float, label: str, drowsy: bool, ttc: float):
        if risk_score >= CSV_LOG_THRESHOLD:
            self._rows.append({
                "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "risk_score": round(risk_score, 2),
                "status"    : label,
                "drowsy"    : drowsy,
                "ttc"       : round(ttc, 3) if ttc != float("inf") else -1,
            })

    def save(self, path: str = "incidents.csv"):
        if not self._rows:
            print("[INFO] No incidents to log.")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "risk_score",
                                                    "status", "drowsy", "ttc"])
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[✓] Incidents saved → {path}  ({len(self._rows)} rows)")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main(cam_index: int, video_source: str):
    print("\n╔══════════════════════════════════════╗")
    print("║       DualSentinel  v1.0             ║")
    print("║  Real-Time Accident Prevention HUD   ║")
    print("╚══════════════════════════════════════╝")
    print(f"\n[→] Cam 1 (driver) : webcam index {cam_index}")
    print(f"[→] Cam 2 (road)   : {video_source}")
    print(f"[→] Eye model      : {EYE_MODEL_PATH}")
    print("\n[INFO] Starting streams… (press 'q' to quit)\n")

    # ── Init streams ──
    driver = DriverStream(model_path=EYE_MODEL_PATH, cam_index=cam_index)
    road   = RoadStream(video_source=video_source)

    driver.start()
    road.start()

    alert_engine = AlertEngine()
    alarm_engine = AlarmEngine()
    logger       = IncidentLogger()

    time.sleep(1.5)   # let streams warm up

    cv2.namedWindow("DualSentinel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DualSentinel", FRAME_W * 2, FRAME_H + STATUS_BAR_H)

    fps_timer = time.time()
    frame_count = 0

    try:
        while True:
            ds = driver.get_state()
            rs = road.get_state()

            risk_score, label, risk_color = compute_risk(
                drowsy          = ds["drowsy"],
                proximity_score = rs["proximity_score"],
            )

            # ── Alerts + Siren ──
            alert_engine.maybe_alert(label)
            if ds["drowsy"]:
                alarm_engine.activate()
            else:
                alarm_engine.deactivate()

            # ── Log ──
            logger.record(risk_score, label, ds["drowsy"], rs["ttc"])

            # ── Build & display HUD ──
            hud = build_hud(ds, rs, risk_score, label, risk_color)
            cv2.imshow("DualSentinel", hud)

            # ── FPS counter in terminal ──
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 3.0:
                fps     = frame_count / elapsed
                ttc_str = f"{rs['ttc']:.1f}s" if rs["ttc"] != float("inf") else "--"
                print(f"[LIVE] FPS:{fps:.1f} | Risk:{risk_score:.1f} | "
                      f"Status:{label:<7} | Drowsy:{ds['drowsy']} | TTC:{ttc_str}",
                      flush=True)
                fps_timer   = time.time()
                frame_count = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print("\n[INFO] Shutting down…")
        alarm_engine.deactivate()
        alarm_engine.shutdown()
        driver.stop()
        road.stop()
        cv2.destroyAllWindows()
        logger.save("incidents.csv")
        print("[✓] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualSentinel — Real-Time Accident Prevention")
    parser.add_argument("--cam",   type=int, default=CAM_DEFAULT,
                        help=f"Webcam index for driver cam (default: {CAM_DEFAULT})")
    parser.add_argument("--video", type=str, default=DASHCAM_DEFAULT,
                        help=f"Path to dashcam video or webcam index (default: {DASHCAM_DEFAULT})")
    args = parser.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    main(cam_index=args.cam, video_source=source)
