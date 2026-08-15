"""
driver_monitor.py
==================
Simple real-time Driver Monitoring System — Live Laptop Camera.

Detects exactly 4 things, each with its own alert message that repeats
until the condition returns to normal, and logs every event to a CSV file:
    1. Looking LEFT or RIGHT        -> "DISTRACTION: Looking LEFT/RIGHT!"
    2. Mobile phone in view          -> "MOBILE PHONE DETECTED!"
    3. Eyes closed (drowsy)          -> "DROWSINESS ALERT! Eyes Closed"
    4. Mouth open (talking/yawning)  -> "MOUTH OPEN ALERT!"

ONLY ONE FILE — nothing else to download or link together.

------------------------------------------------------------------------
SETUP
------------------------------------------------------------------------
1. pip install -r requirements.txt
   (this includes ultralytics, which auto-downloads the phone-detection
   model the first time you run the script — no manual weights needed)
2. python driver_monitor.py
3. Press 'Q' to quit.

Every alert is logged to event_log.csv (created automatically in the
same folder) with a timestamp, event type, and how long it lasted.
------------------------------------------------------------------------
"""

import os
import csv
import cv2
import time
import math
import threading
from datetime import datetime
import mediapipe as mp

# ---------------- Beep alert (Windows) / terminal bell fallback ----------------
# Runs in a background thread so repeated beeps don't freeze the video loop.
try:
    import winsound
    def _beep_now(freq, dur):
        winsound.Beep(freq, dur)
except ImportError:
    def _beep_now(freq, dur):
        print("\a", end="")

def beep(freq=1000, dur=300):
    threading.Thread(target=_beep_now, args=(freq, dur), daemon=True).start()

BEEP_REPEAT_INTERVAL = 0.8  # seconds between repeated beeps while condition stays abnormal


# ---------------- CONFIG ----------------
EAR_THRESHOLD = 0.21          # eyes count as "closed" below this
EAR_CONSEC_FRAMES = 15        # frames closed before alert fires

MAR_THRESHOLD = 0.6           # mouth counts as "open" above this
MAR_CONSEC_FRAMES = 10

GAZE_CONSEC_FRAMES = 15       # frames looking away before alert fires

LOG_FILE = "event_log.csv"    # every alert start/end gets logged here

PHONE_DETECTION_ENABLED = True
PHONE_CONF_THRESHOLD = 0.35
PHONE_CONSEC_FRAMES = 5
CELL_PHONE_CLASS_ID = 67      # COCO class ID for "cell phone"

# MediaPipe Face Mesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
NOSE_TIP = 1
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454


# ---------------- MediaPipe setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ---------------- Phone detection setup (ultralytics YOLO) ----------------
phone_model = None
phone_detection_active = False

if PHONE_DETECTION_ENABLED:
    try:
        from ultralytics import YOLO
        phone_model = YOLO("yolo11n.pt")   # auto-downloads on first run
        phone_detection_active = True
        print("YOLO loaded — phone detection ENABLED.")
    except Exception as e:
        print(f"[INFO] Could not load YOLO ({e}). Phone detection disabled — "
              f"eyes, mouth, and gaze detection still run normally.")


def detect_phone(frame):
    """Returns True if a cell phone is detected in the frame above the confidence threshold."""
    if not phone_detection_active or phone_model is None:
        return False

    try:
        results = phone_model(frame, conf=PHONE_CONF_THRESHOLD, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for cls_id in result.boxes.cls.tolist():
                if int(cls_id) == CELL_PHONE_CLASS_ID:
                    return True
        return False
    except Exception as e:
        print("[PHONE DETECTION ERROR]", e)
        return False


# ---------------- CSV event logging ----------------
def log_event(event_type, status, details=""):
    """
    Appends one row to event_log.csv, creating the header on first write.
    status is either 'START' (alert just triggered) or 'END' (condition
    cleared / back to normal).
    """
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "event_type", "status", "details"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, status, details])


# ---------------- Helper functions ----------------
def euclidean(p1, p2):
    return math.dist(p1, p2)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    v1 = euclidean(pts[1], pts[5])
    v2 = euclidean(pts[2], pts[4])
    horiz = euclidean(pts[0], pts[3])
    return (v1 + v2) / (2.0 * horiz) if horiz else 0.0


def mouth_aspect_ratio(landmarks, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH]
    v1 = euclidean(pts[1], pts[7])
    v2 = euclidean(pts[2], pts[6])
    v3 = euclidean(pts[3], pts[5])
    horiz = euclidean(pts[0], pts[4])
    return (v1 + v2 + v3) / (2.0 * horiz) if horiz else 0.0


def gaze_direction(landmarks, w, h):
    nose_x = landmarks[NOSE_TIP].x * w
    left_x = landmarks[LEFT_FACE_EDGE].x * w
    right_x = landmarks[RIGHT_FACE_EDGE].x * w
    face_width = right_x - left_x
    if face_width == 0:
        return "CENTER"
    ratio = (nose_x - left_x) / face_width
    if ratio < 0.35:
        return "LEFT"
    elif ratio > 0.65:
        return "RIGHT"
    return "CENTER"


# ---------------- Main loop ----------------
def main():
    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # more reliable on Windows
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    eye_closed_count = 0
    mouth_open_count = 0
    gaze_away_count = 0
    phone_count = 0

    # Track whether each condition was active last frame (for the green "Normal" message)
    eye_was_alert = False
    mouth_was_alert = False
    gaze_was_alert = False
    phone_was_alert = False

    # Track last beep time per condition so beeps repeat on an interval, not every frame
    eye_last_beep = 0.0
    mouth_last_beep = 0.0
    gaze_last_beep = 0.0
    phone_last_beep = 0.0

    print("Driver Monitoring System started. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = time.time()
        alert_messages = []   # red — something wrong
        ok_messages = []      # green — back to normal

        # ---- Phone detection ----
        phone_seen = phone_detection_active and detect_phone(frame)
        if phone_seen:
            phone_count += 1
        else:
            phone_count = 0

        if phone_count >= PHONE_CONSEC_FRAMES:
            alert_messages.append("MOBILE PHONE DETECTED!")
            if now - phone_last_beep >= BEEP_REPEAT_INTERVAL:
                beep(1200, 300)
                phone_last_beep = now
            if not phone_was_alert:
                log_event("PHONE_USAGE", "START")
            phone_was_alert = True
        elif phone_was_alert:
            ok_messages.append("Phone Away - Normal")
            log_event("PHONE_USAGE", "END")
            phone_was_alert = False

        # ---- Face-based detection ----
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Eyes closed
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                eye_closed_count += 1
            else:
                eye_closed_count = 0

            if eye_closed_count >= EAR_CONSEC_FRAMES:
                alert_messages.append("DROWSINESS ALERT! Eyes Closed")
                if now - eye_last_beep >= BEEP_REPEAT_INTERVAL:
                    beep(1000, 400)
                    eye_last_beep = now
                if not eye_was_alert:
                    log_event("DROWSINESS", "START", f"EAR={avg_ear:.3f}")
                eye_was_alert = True
            elif eye_was_alert:
                ok_messages.append("Eyes Open - Normal")
                log_event("DROWSINESS", "END")
                eye_was_alert = False

            # Mouth open (talking / yawning)
            mar = mouth_aspect_ratio(landmarks, w, h)
            if mar > MAR_THRESHOLD:
                mouth_open_count += 1
            else:
                mouth_open_count = 0

            if mouth_open_count >= MAR_CONSEC_FRAMES:
                alert_messages.append("MOUTH OPEN ALERT!")
                if now - mouth_last_beep >= BEEP_REPEAT_INTERVAL:
                    beep(850, 300)
                    mouth_last_beep = now
                if not mouth_was_alert:
                    log_event("MOUTH_OPEN", "START", f"MAR={mar:.3f}")
                mouth_was_alert = True
            elif mouth_was_alert:
                ok_messages.append("Mouth Closed - Normal")
                log_event("MOUTH_OPEN", "END")
                mouth_was_alert = False

            # Gaze left/right
            direction = gaze_direction(landmarks, w, h)
            if direction != "CENTER":
                gaze_away_count += 1
            else:
                gaze_away_count = 0

            if gaze_away_count >= GAZE_CONSEC_FRAMES:
                alert_messages.append(f"DISTRACTION: Looking {direction}!")
                if now - gaze_last_beep >= BEEP_REPEAT_INTERVAL:
                    beep(900, 300)
                    gaze_last_beep = now
                if not gaze_was_alert:
                    log_event("DISTRACTION", "START", f"direction={direction}")
                gaze_was_alert = True
            elif gaze_was_alert:
                ok_messages.append("Looking Center - Normal")
                log_event("DISTRACTION", "END")
                gaze_was_alert = False

            cv2.putText(frame, f"EAR:{avg_ear:.2f} MAR:{mar:.2f} Gaze:{direction}",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        else:
            alert_messages.append("FACE NOT DETECTED")

        # ---- Draw messages on screen: red = alert, green = back to normal ----
        for i, msg in enumerate(alert_messages):
            cv2.putText(frame, msg, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        for i, msg in enumerate(ok_messages):
            cv2.putText(frame, msg, (10, 30 + (len(alert_messages) + i) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

        cv2.imshow("Driver Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()