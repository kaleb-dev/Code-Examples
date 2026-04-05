"""
PTZOptics Camera Person Detection and Tracking

Detects and tracks people using MobileNet SSD (deep learning) for reliable
person detection and CSRT tracker for smooth frame-to-frame tracking.
Click on a detected person to lock the PTZ camera onto them.

Detection: MobileNet SSD via cv2.dnn — trained on PASCAL VOC 'person' class,
detects standing, seated, and partially visible people reliably. Will NOT
false-detect on equipment, curtains, chairs, or other non-human objects.

Tracking: CSRT (Channel and Spatial Reliability Tracker) follows the selected
person frame-by-frame without needing re-detection. Much smoother and more
reliable than detection-based matching.

Requirements:
- PTZOptics camera with RTSP streaming enabled
- opencv-contrib-python (for CSRT tracker)
- MobileNet SSD model files (run download_models.py first)
"""

import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import time
import threading
import argparse
import requests
from requests.auth import HTTPDigestAuth

# =============================================================================
# CONFIGURATION
# =============================================================================

# Camera
DEFAULT_RTSP_STREAM = "stream1"
TARGET_FPS = 30

# MobileNet SSD person detection
DNN_CONFIDENCE = 0.5             # Minimum confidence for person detection (0.0-1.0)
DNN_INPUT_SIZE = (300, 300)      # MobileNet SSD input size
DNN_DETECT_INTERVAL = 10        # Re-run full detection every N frames
PERSON_CLASS_ID = 15              # 'person' class in PASCAL VOC

# CSRT tracker re-initialization
TRACKER_MAX_FAILURES = 15         # Frames of tracker failure before giving up

# PTZ tracking
PTZ_DEAD_ZONE = 80
PTZ_SPEED_DIVISOR = 50
PTZ_MAX_SPEED = 12
PTZ_TILT_MAX_SPEED = 10

# Display
DISPLAY_WIDTH = 0                # 0 = native resolution
DISPLAY_HEIGHT = 0
PERSON_COLOR = (0, 255, 0)       # Green for detected people
TRACKING_COLOR = (255, 165, 0)   # Orange for tracked target
STATUS_COLOR = (0, 255, 255)     # Yellow for status text
GREEN_LIGHT = (0, 255, 0)
RED_LIGHT = (0, 0, 255)

# =============================================================================

class PTZPersonTracker:
    """
    PTZ camera person detection and tracking system.

    Uses MobileNet SSD for person detection and CSRT for frame-to-frame tracking.
    Only detects and tracks humans — nothing else.
    """

    def __init__(self, camera_ip, stream="stream1", enable_tracking=False,
                 username="admin", password="admin"):
        self.camera_ip = camera_ip
        self.rtsp_url = f"rtsp://{camera_ip}/{stream}"
        self.cap = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._grabber_running = False

        # PTZ control
        self.tracking_enabled = enable_tracking
        self.cgi_url = f"http://{camera_ip}/cgi-bin/ptzctrl.cgi"
        self.http_session = requests.Session()
        if username and password:
            self.http_session.auth = HTTPDigestAuth(username, password)
        self.tracking_target = None

        # Load MobileNet SSD model
        model_dir = os.path.dirname(os.path.abspath(__file__))
        prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        caffemodel = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")

        if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
            print("Model files not found. Run: python download_models.py")
            raise FileNotFoundError("MobileNet SSD model files missing")

        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("MobileNet SSD person detector loaded")

        # CSRT tracker state
        self.tracker = None
        self.tracker_bbox = None
        self.tracker_failures = 0
        self.click_point = None
        self.is_tracking = False

        # Detection results cache
        self.last_detections = []
        self.detect_counter = 0

    # --- Person Detection (MobileNet SSD) ---

    def detect_people(self, frame):
        """
        Detect people in frame using MobileNet SSD.

        Returns list of (x, y, w, h, confidence) for each detected person.
        Only returns 'person' class detections above the confidence threshold.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, DNN_INPUT_SIZE, 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        people = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if class_id == PERSON_CLASS_ID and confidence > DNN_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                # Clamp to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    people.append((x1, y1, x2 - x1, y2 - y1, float(confidence)))

        return people

    def detect_people_cached(self, frame):
        """Run detection every N frames, return cached results otherwise."""
        self.detect_counter += 1
        if self.detect_counter >= DNN_DETECT_INTERVAL:
            self.detect_counter = 0
            self.last_detections = self.detect_people(frame)
        return self.last_detections

    # --- CSRT Tracking ---

    def start_tracking(self, frame, bbox):
        """Initialize CSRT tracker on the selected person."""
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.tracker_bbox = bbox
        self.tracker_failures = 0
        self.is_tracking = True
        x, y, w, h = [int(v) for v in bbox]
        print(f"CSRT tracker locked onto person at ({x}, {y}, {w}x{h})")

    def update_tracking(self, frame):
        """
        Update CSRT tracker. Returns (success, bbox) where bbox is (x, y, w, h).

        CSRT tracks the visual pattern of the selected person frame-to-frame.
        No detection needed — it follows the pixels.
        """
        if self.tracker is None:
            return False, None

        success, bbox = self.tracker.update(frame)

        if success:
            self.tracker_bbox = tuple(int(v) for v in bbox)
            self.tracker_failures = 0
            return True, self.tracker_bbox
        else:
            self.tracker_failures += 1
            if self.tracker_failures >= TRACKER_MAX_FAILURES:
                print("CSRT tracker lost target")
                self.stop_tracking()
                return False, None
            # Return last known position during brief failures
            return True, self.tracker_bbox

    def stop_tracking(self):
        """Stop tracking and reset state."""
        self.tracker = None
        self.tracker_bbox = None
        self.tracker_failures = 0
        self.is_tracking = False
        self.tracking_target = None
        print("Tracking stopped")

    def try_reacquire(self, frame, detections):
        """
        If CSRT is losing the target, try to re-initialize from nearby detection.

        Only re-acquires from a detection that overlaps with the current tracker
        position — won't jump to a different person.
        """
        if self.tracker_bbox is None or self.tracker_failures < 3:
            return

        tx, ty, tw, th = self.tracker_bbox
        for (x, y, w, h, conf) in detections:
            # Check IoU overlap with tracker position
            ix1 = max(tx, x)
            iy1 = max(ty, y)
            ix2 = min(tx + tw, x + w)
            iy2 = min(ty + th, y + h)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = tw * th + w * h - intersection
                iou = intersection / union if union > 0 else 0

                if iou > 0.2:
                    # Re-initialize tracker with this detection
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                    self.tracker.init(frame, (x, y, w, h))
                    self.tracker_bbox = (x, y, w, h)
                    self.tracker_failures = 0
                    return

    # --- Mouse Interaction ---

    def on_mouse_click(self, event, x, y, flags, param):
        """Left-click to select a person, right-click to stop tracking."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.stop_tracking()
            self.stop_camera()

    def handle_click(self, frame, detections):
        """Match click to a detected person and start CSRT tracking."""
        if self.click_point is None:
            return

        cx, cy = self.click_point
        self.click_point = None

        for (x, y, w, h, conf) in detections:
            if x <= cx <= x + w and y <= cy <= y + h:
                self.start_tracking(frame, (x, y, w, h))
                return

        print("No person detected at click location")

    # --- PTZ Control ---

    def move_camera(self, direction, pan_speed=5, tilt_speed=5):
        url = f"{self.cgi_url}?ptzcmd&{direction}&{pan_speed}&{tilt_speed}"
        try:
            self.http_session.get(url, timeout=1)
        except requests.RequestException:
            pass

    def stop_camera(self):
        url = f"{self.cgi_url}?ptzcmd&ptzstop&0&0"
        try:
            self.http_session.get(url, timeout=1)
        except requests.RequestException:
            pass

    def track_target(self, target_x, target_y, frame_width, frame_height):
        self.tracking_target = (target_x, target_y)

        center_x = frame_width // 2
        center_y = frame_height // 2
        dx = target_x - center_x
        dy = target_y - center_y

        if abs(dx) < PTZ_DEAD_ZONE and abs(dy) < PTZ_DEAD_ZONE:
            self.stop_camera()
            return

        max_offset = max(abs(dx), abs(dy))
        pan_speed = min(PTZ_MAX_SPEED, max(1, int(max_offset / PTZ_SPEED_DIVISOR)))
        tilt_speed = min(PTZ_TILT_MAX_SPEED, max(1, int(max_offset / PTZ_SPEED_DIVISOR)))

        h_dir = "right" if dx >= PTZ_DEAD_ZONE else "left" if dx <= -PTZ_DEAD_ZONE else ""
        v_dir = "down" if dy >= PTZ_DEAD_ZONE else "up" if dy <= -PTZ_DEAD_ZONE else ""

        if h_dir and v_dir:
            self.move_camera(h_dir + v_dir, pan_speed, tilt_speed)
        elif h_dir:
            self.move_camera(h_dir, pan_speed, tilt_speed)
        elif v_dir:
            self.move_camera(v_dir, pan_speed, tilt_speed)

    # --- Frame Grabber ---

    def start_frame_grabber(self):
        print(f"Connecting to RTSP stream: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if not self.cap.isOpened():
            tcp_url = f"{self.rtsp_url}?tcp"
            print(f"Retrying with TCP: {tcp_url}")
            self.cap = cv2.VideoCapture(tcp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print("Failed to open RTSP stream")
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"RTSP stream connected: {w}x{h} @ {fps:.1f}fps")

        self._grabber_running = True
        self._grab_thread = threading.Thread(target=self._frame_grab_loop, daemon=True)
        self._grab_thread.start()
        return True

    def _frame_grab_loop(self):
        while self._grabber_running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)

    def get_camera_frame(self):
        if self._latest_frame is None:
            return None
        with self._frame_lock:
            return self._latest_frame.copy()

    def cleanup(self):
        self._grabber_running = False
        if hasattr(self, '_grab_thread') and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2)
        if self.tracking_enabled:
            self.stop_camera()
        if self.cap is not None:
            self.cap.release()

    # --- Main Loop ---

    def run(self):
        print(f"Starting person detection for camera at {self.camera_ip}")
        if self.tracking_enabled:
            print("Click on a detected person to track them")
            print("Left-click: select | Right-click: stop tracking")
        print("Press 'q' to quit")

        if not self.start_frame_grabber():
            print("Could not connect to camera.")
            return

        # Wait for first frame
        for _ in range(50):
            if self.get_camera_frame() is not None:
                break
            time.sleep(0.1)

        window_name = "PTZOptics Person Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        first_frame = self.get_camera_frame()
        if first_frame is not None:
            if DISPLAY_WIDTH > 0 and DISPLAY_HEIGHT > 0:
                cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            else:
                cv2.resizeWindow(window_name, first_frame.shape[1], first_frame.shape[0])
        if self.tracking_enabled:
            cv2.setMouseCallback(window_name, self.on_mouse_click)

        frame_interval = 1.0 / TARGET_FPS

        while True:
            loop_start = time.perf_counter()

            frame = self.get_camera_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # Detect people (cached, runs every N frames)
            detections = self.detect_people_cached(frame)

            # Handle tracking
            if self.tracking_enabled:
                self.handle_click(frame, detections)

                if self.is_tracking:
                    success, bbox = self.update_tracking(frame)

                    # Try to re-acquire from detections if tracker is struggling
                    if self.tracker_failures > 0 and detections:
                        self.try_reacquire(frame, detections)

                    if success and bbox and self.tracker_failures == 0:
                        x, y, w, h = bbox
                        target_x = x + w // 2
                        target_y = y + h // 2
                        self.track_target(target_x, target_y,
                                          frame.shape[1], frame.shape[0])
                    elif not success:
                        self.tracking_target = None
                        self.stop_camera()
                    else:
                        # Tracker struggling — stop camera, don't pan randomly
                        self.stop_camera()
                else:
                    self.tracking_target = None

            # --- Display ---
            display_frame = frame.copy()

            # Draw all detected people
            for (x, y, w, h, conf) in detections:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), PERSON_COLOR, 2)
                label = f"Person {conf:.0%}"
                cv2.putText(display_frame, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_COLOR, 2)

            # Draw tracking overlay
            if self.is_tracking and self.tracker_bbox:
                tx, ty, tw, th = self.tracker_bbox
                cv2.rectangle(display_frame, (tx, ty), (tx + tw, ty + th),
                              TRACKING_COLOR, 3)
                center = (tx + tw // 2, ty + th // 2)
                cv2.drawMarker(display_frame, center, TRACKING_COLOR,
                               cv2.MARKER_CROSS, 30, 2)
                cv2.putText(display_frame, "TRACKING", (tx, ty - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRACKING_COLOR, 2)

            # Status indicator
            has_people = len(detections) > 0
            indicator_color = GREEN_LIGHT if has_people else RED_LIGHT
            cv2.circle(display_frame, (30, 40), 25, indicator_color, -1)
            cv2.circle(display_frame, (30, 40), 25, (255, 255, 255), 2)

            # Status text
            if self.is_tracking:
                status = "TRACKING PERSON"
            elif self.tracking_enabled:
                status = "CLICK A PERSON TO TRACK"
            elif has_people:
                n = len(detections)
                status = f"{n} PERSON{'S' if n > 1 else ''} DETECTED"
            else:
                status = "NO PERSONS DETECTED"
            cv2.putText(display_frame, status, (70, 48),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATUS_COLOR, 2)

            # Help text
            fh = display_frame.shape[0]
            cv2.putText(display_frame,
                      "Press 'q' to quit | Left-click: track | Right-click: stop",
                      (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # FPS pacing
            elapsed = time.perf_counter() - loop_start
            wait_time = frame_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        self.cleanup()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='PTZOptics Person Detection and Tracking',
        epilog='Example: python main.py 192.168.1.100 --track'
    )
    parser.add_argument('camera_ip', help='IP address of PTZOptics camera')
    parser.add_argument('--stream', type=int, default=1, choices=[1, 2],
                       help='Camera stream: 1 for native resolution, 2 for lower latency (default: 1)')
    parser.add_argument('--track', action='store_true',
                       help='Enable PTZ tracking - click a person to follow them')
    parser.add_argument('--username', default='admin',
                       help='Camera username (default: admin)')
    parser.add_argument('--password', default='admin',
                       help='Camera password (default: admin)')

    args = parser.parse_args()

    tracker = PTZPersonTracker(
        camera_ip=args.camera_ip,
        stream=f"stream{args.stream}",
        enable_tracking=args.track,
        username=args.username,
        password=args.password
    )

    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nStopped by user")

if __name__ == "__main__":
    main()
