"""
PTZOptics Camera Motion Detection and Person Tracking Example

This example demonstrates how to connect to a PTZOptics camera and perform
real-time motion detection and person tracking using OpenCV. The system detects
motion, faces, and full human bodies, and can track a selected person using
PTZ camera controls via HTTP-CGI.

Requirements:
- PTZOptics camera with RTSP streaming enabled
- Network connectivity to the camera
- OpenCV Python package
- requests library (for PTZ tracking)
"""

import os
# Suppress noisy ffmpeg/libav h264 decode warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
# Force RTSP over TCP to prevent UDP packet loss (eliminates h264 decode errors)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import time
from datetime import datetime
import argparse
import requests
from requests.auth import HTTPDigestAuth

# =============================================================================
# CONFIGURATION VARIABLES - Modify these for your specific setup
# =============================================================================

# Camera connection settings
DEFAULT_RTSP_STREAM = "stream2"  # Use stream2 for lower latency, stream1 for higher quality

# Motion detection parameters
DEFAULT_SENSITIVITY = 50         # Motion sensitivity (lower = more sensitive)
DEFAULT_MIN_AREA = 1500          # Minimum motion area in pixels to trigger detection
MOTION_HOLD_FRAMES = 15          # Frames to hold motion state (reduces flickering)

# Face detection parameters
FACE_DETECTION_INTERVAL = 5      # Run face detection every N frames (for performance)
FACE_HOLD_FRAMES = 15            # Frames to hold face detection state
BODY_DETECTION_INTERVAL = 3      # Run body detection every N frames

# Performance settings
DETECTION_FRAME_WIDTH = 640      # Downscale frames to this width for detection (0 = no downscale)

# Background subtractor settings
BACKGROUND_DETECT_SHADOWS = True # Enable shadow detection in background subtraction
MORPH_KERNEL_SIZE = (3, 3)       # Kernel size for morphological operations

# Person detection settings
PERSON_SCALE_FACTOR = 1.1        # Scale factor for person detection
PERSON_MIN_NEIGHBORS = 3         # Minimum neighbors for person detection
PERSON_MIN_SIZE = (30, 30)       # Minimum size for person detection

# Full body detection settings
BODY_SCALE_FACTOR = 1.05         # Scale factor for full body detection
BODY_MIN_NEIGHBORS = 3           # Minimum neighbors for full body detection
BODY_MIN_SIZE = (60, 120)        # Minimum size for full body detection

# PTZ tracking settings
PTZ_DEAD_ZONE = 80               # Pixels from center before camera moves (wider = smoother)
PTZ_SPEED_DIVISOR = 50           # Higher = slower speed scaling for smoother movement
PTZ_MAX_SPEED = 12               # Maximum pan speed (PTZOptics range: 1-24, capped low for smoothness)
PTZ_TILT_MAX_SPEED = 10          # Maximum tilt speed (PTZOptics range: 1-20, capped low for smoothness)

# Display settings
DISPLAY_WIDTH = 1280             # Window display width (0 = native frame size)
DISPLAY_HEIGHT = 720             # Window display height (0 = native frame size)
MOTION_COLOR = (0, 255, 0)       # Green color for motion bounding boxes (BGR)
PERSON_COLOR = (0, 0, 255)       # Red color for face bounding boxes (BGR)
BODY_COLOR = (255, 0, 255)       # Magenta color for body bounding boxes (BGR)
TRACKING_COLOR = (255, 165, 0)   # Orange color for tracking crosshair (BGR)
STATUS_COLOR = (0, 255, 255)     # Yellow color for status text (BGR)

# Status indicator settings
STATUS_INDICATOR_SIZE = 50       # Size of the status indicator circle
STATUS_INDICATOR_POSITION = (30, 40)  # Position (x, y) from top-left corner
GREEN_LIGHT = (0, 255, 0)        # Green for active motion/person detection
RED_LIGHT = (0, 0, 255)          # Red for no motion detected


# =============================================================================

class PTZMotionDetector:
    """
    PTZOptics Camera Motion Detection and Person Tracking System

    This class provides real-time motion detection and person tracking for
    PTZOptics cameras using RTSP streaming. It combines background subtraction
    for motion detection with Haar cascade classifiers for face and body detection.

    When tracking is enabled, click on a detected person to lock the camera
    onto them. The camera will follow that specific person until you right-click
    to stop tracking.
    """

    def __init__(self, camera_ip, sensitivity=DEFAULT_SENSITIVITY, min_area=DEFAULT_MIN_AREA,
                 stream="stream2", enable_tracking=False, username="admin", password="admin"):
        self.camera_ip = camera_ip
        self.rtsp_url = f"rtsp://{camera_ip}/{stream}"
        self.cap = None

        self.sensitivity = sensitivity
        self.min_area = min_area

        # PTZ tracking configuration
        self.tracking_enabled = enable_tracking
        self.cgi_url = f"http://{camera_ip}/cgi-bin/ptzctrl.cgi"
        self.http_session = requests.Session()
        if username and password:
            self.http_session.auth = HTTPDigestAuth(username, password)
        self.tracking_target = None

        # Initialize background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=BACKGROUND_DETECT_SHADOWS
        )

        # Load Haar cascade classifiers for face and full body detection
        self.person_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )

        # Motion smoothing
        self.motion_frames_count = 0
        self.motion_hold_frames = MOTION_HOLD_FRAMES

        # Face detection frame skipping and persistence
        self.face_detection_counter = 0
        self.face_frames_count = 0
        self.face_hold_frames = FACE_HOLD_FRAMES
        self.last_face_result = (False, [])

        # Body detection frame skipping and persistence
        self.body_detection_counter = 0
        self.body_frames_count = 0
        self.last_body_result = (False, [])

        # Detection scale factor (computed on first frame)
        self.detection_scale = 1.0

        # Click-to-track: user selects a specific person by clicking on them
        self.selected_target = None
        self.selected_target_center = None
        self.click_point = None
        # Count frames since target was last matched (for graceful loss)
        self.target_lost_frames = 0
        self.TARGET_LOST_THRESHOLD = 30  # Frames before declaring target truly lost

    def on_mouse_click(self, event, x, y, flags, param):
        """Mouse callback for click-to-track. Left-click to select, right-click to stop."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_target = None
            self.selected_target_center = None
            self.click_point = None
            self.target_lost_frames = 0
            print("Tracking stopped")

    def select_target_from_click(self, person_detections):
        """Match a mouse click to a detected person (body or face only)."""
        if self.click_point is None:
            return

        cx, cy = self.click_point
        self.click_point = None

        for (x, y, w, h) in person_detections:
            if x <= cx <= x + w and y <= cy <= y + h:
                self.selected_target = (x, y, w, h)
                self.selected_target_center = (x + w // 2, y + h // 2)
                self.target_lost_frames = 0
                print(f"Locked onto person at ({x}, {y}, {w}x{h})")
                return

        print("No person detected at click location - click on a highlighted body or face")

    def update_selected_target(self, person_detections):
        """
        Update the locked target using IoU + proximity scoring.

        Sticks to the same person by heavily weighting overlap and penalizing
        distance. Won't jump to a different person across the frame.
        """
        if self.selected_target is None:
            return None

        sx, sy, sw, sh = self.selected_target
        s_cx = sx + sw // 2
        s_cy = sy + sh // 2
        best_match = None
        best_score = -1

        for (x, y, w, h) in person_detections:
            # Calculate IoU overlap
            ix1 = max(sx, x)
            iy1 = max(sy, y)
            ix2 = min(sx + sw, x + w)
            iy2 = min(sy + sh, y + h)

            iou = 0
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = sw * sh + w * h - intersection
                iou = intersection / union if union > 0 else 0

            # Calculate center distance
            d_cx = x + w // 2
            d_cy = y + h // 2
            dist = ((s_cx - d_cx) ** 2 + (s_cy - d_cy) ** 2) ** 0.5

            # Score: heavily weight IoU, penalize distance
            dist_penalty = min(dist / 500.0, 1.0)
            score = iou * 2.0 + (1.0 - dist_penalty)

            if score > best_score:
                best_score = score
                best_match = (x, y, w, h)

        # Require minimum score to prevent jumping to a distant person
        if best_match and best_score > 0.5:
            x, y, w, h = best_match
            self.selected_target = best_match
            self.selected_target_center = (x + w // 2, y + h // 2)
            self.target_lost_frames = 0
            return self.selected_target_center
        else:
            # Target not matched this frame - hold position briefly
            self.target_lost_frames += 1
            if self.target_lost_frames < self.TARGET_LOST_THRESHOLD:
                # Keep tracking last known position
                return self.selected_target_center
            else:
                print("Target lost")
                self.selected_target = None
                self.selected_target_center = None
                self.target_lost_frames = 0
                return None

    def move_camera(self, direction, pan_speed=5, tilt_speed=5):
        """Send a PTZ move command via HTTP-CGI."""
        url = f"{self.cgi_url}?ptzcmd&{direction}&{pan_speed}&{tilt_speed}"
        try:
            self.http_session.get(url, timeout=1)
        except requests.RequestException:
            pass

    def stop_camera(self):
        """Send a PTZ stop command via HTTP-CGI."""
        url = f"{self.cgi_url}?ptzcmd&ptzstop&0&0"
        try:
            self.http_session.get(url, timeout=1)
        except requests.RequestException:
            pass

    def track_target(self, target_x, target_y, frame_width, frame_height):
        """Move camera to center on a target position with dead zone and speed scaling."""
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

        h_dir = ""
        v_dir = ""
        if abs(dx) >= PTZ_DEAD_ZONE:
            h_dir = "right" if dx > 0 else "left"
        if abs(dy) >= PTZ_DEAD_ZONE:
            v_dir = "down" if dy > 0 else "up"

        if h_dir and v_dir:
            self.move_camera(h_dir + v_dir, pan_speed, tilt_speed)
        elif h_dir:
            self.move_camera(h_dir, pan_speed, tilt_speed)
        elif v_dir:
            self.move_camera(v_dir, pan_speed, tilt_speed)

    def get_camera_frame(self):
        """
        Retrieve the most recent frame from the PTZOptics camera via RTSP.

        Uses TCP transport to eliminate h264 decode errors from UDP packet loss.
        Drains the buffer to always process the latest frame.
        """
        if self.cap is None:
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
                return None
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("RTSP stream connected")

        # Drain buffer to get the latest frame (prevents falling behind real-time)
        for _ in range(5):
            grabbed = self.cap.grab()
            if not grabbed:
                break

        ret, frame = self.cap.retrieve()
        if ret:
            return frame

        ret, frame = self.cap.read()
        if ret:
            return frame

        print("Failed to read frame, attempting reconnection...")
        self.cap.release()
        self.cap = None
        return None

    def detect_motion(self, frame):
        """Detect motion using background subtraction."""
        fg_mask = self.background_subtractor.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        motion_areas = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))

        return motion_detected, motion_areas, fg_mask

    def smooth_motion_detection(self, raw_motion_detected):
        """Apply smoothing to motion detection to reduce flickering."""
        if raw_motion_detected:
            self.motion_frames_count = self.motion_hold_frames
            return True
        else:
            if self.motion_frames_count > 0:
                self.motion_frames_count -= 1
                return True
            return False

    def get_detection_frame(self, frame):
        """Prepare a downscaled grayscale frame for detection."""
        if DETECTION_FRAME_WIDTH > 0 and frame.shape[1] > DETECTION_FRAME_WIDTH:
            self.detection_scale = DETECTION_FRAME_WIDTH / frame.shape[1]
            small = cv2.resize(frame, None, fx=self.detection_scale, fy=self.detection_scale)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            self.detection_scale = 1.0
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def scale_detections(self, detections):
        """Scale detection rectangles back to full frame coordinates."""
        if self.detection_scale == 1.0 or len(detections) == 0:
            return detections
        s = 1.0 / self.detection_scale
        return [(int(x * s), int(y * s), int(w * s), int(h * s)) for (x, y, w, h) in detections]

    def detect_person_with_persistence(self, gray):
        """Detect faces with frame skipping and result persistence."""
        self.face_detection_counter += 1

        if self.face_detection_counter >= FACE_DETECTION_INTERVAL:
            self.face_detection_counter = 0
            people = self.person_cascade.detectMultiScale(
                gray, scaleFactor=PERSON_SCALE_FACTOR,
                minNeighbors=PERSON_MIN_NEIGHBORS, minSize=PERSON_MIN_SIZE
            )
            faces = self.scale_detections(people)
            face_detected = len(faces) > 0

            if face_detected:
                self.face_frames_count = self.face_hold_frames
                self.last_face_result = (True, faces)
                return True, faces
            else:
                if self.face_frames_count <= 0:
                    self.last_face_result = (False, [])

        if self.face_frames_count > 0:
            self.face_frames_count -= 1
            return self.last_face_result
        return False, []

    def detect_bodies_with_persistence(self, gray):
        """Detect full bodies with frame skipping and result persistence."""
        self.body_detection_counter += 1

        if self.body_detection_counter >= BODY_DETECTION_INTERVAL:
            self.body_detection_counter = 0
            raw_bodies = self.body_cascade.detectMultiScale(
                gray, scaleFactor=BODY_SCALE_FACTOR,
                minNeighbors=BODY_MIN_NEIGHBORS, minSize=BODY_MIN_SIZE
            )
            bodies = self.scale_detections(raw_bodies)
            body_detected = len(bodies) > 0

            if body_detected:
                self.body_frames_count = FACE_HOLD_FRAMES
                self.last_body_result = (True, bodies)
                return True, bodies
            else:
                if self.body_frames_count <= 0:
                    self.last_body_result = (False, [])

        if self.body_frames_count > 0:
            self.body_frames_count -= 1
            return self.last_body_result
        return False, []

    def cleanup(self):
        """Clean up camera resources and stop PTZ movement."""
        if self.tracking_enabled:
            self.stop_camera()
        if self.cap is not None:
            self.cap.release()

    def run_detection(self, display=False, save_detections=False):
        """Run the main motion detection and tracking loop."""
        print(f"Starting PTZOptics motion detection for camera at {self.camera_ip}")
        if self.tracking_enabled:
            print("PTZ tracking enabled - click on a person (body/face) to lock and track them")
            print("Left-click to select, right-click to stop tracking")
        print("Press 'q' to quit the display window")

        window_name = "PTZOptics Motion Detection"
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if DISPLAY_WIDTH > 0 and DISPLAY_HEIGHT > 0:
                cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            if self.tracking_enabled:
                cv2.setMouseCallback(window_name, self.on_mouse_click)

        frame_count = 0

        while True:
            frame = self.get_camera_frame()
            if frame is None:
                print("Failed to get frame, retrying in 2 seconds...")
                time.sleep(2)
                continue

            frame_count += 1

            # Detect motion
            raw_motion_detected, motion_areas, fg_mask = self.detect_motion(frame)
            motion_detected = self.smooth_motion_detection(raw_motion_detected)

            # Prepare downscaled grayscale once, shared by face and body detection
            gray = self.get_detection_frame(frame)

            # Detect faces and bodies with frame skipping
            person_detected, people = self.detect_person_with_persistence(gray)
            bodies_detected, bodies = self.detect_bodies_with_persistence(gray)

            # PTZ tracking: only tracks people, only when user clicks to select
            if self.tracking_enabled:
                person_detections = list(bodies) + list(people)
                self.select_target_from_click(person_detections)

                if self.selected_target is not None:
                    target_pos = self.update_selected_target(person_detections)
                    if target_pos:
                        self.track_target(target_pos[0], target_pos[1],
                                          frame.shape[1], frame.shape[0])
                    else:
                        self.tracking_target = None
                        self.stop_camera()
                else:
                    self.tracking_target = None

            # Display
            if display:
                display_frame = frame.copy()

                # Draw motion areas in green
                for (x, y, w, h) in motion_areas:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), MOTION_COLOR, 2)

                # Draw face detections in red
                for (x, y, w, h) in people:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), PERSON_COLOR, 2)
                    cv2.putText(display_frame, "Face", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_COLOR, 2)

                # Draw full body detections in magenta
                for (x, y, w, h) in bodies:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), BODY_COLOR, 2)
                    cv2.putText(display_frame, "Body", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, BODY_COLOR, 2)

                # Draw tracking crosshair and selected target highlight
                if self.tracking_enabled and self.tracking_target:
                    tx, ty = self.tracking_target
                    cv2.drawMarker(display_frame, (tx, ty), TRACKING_COLOR,
                                   cv2.MARKER_CROSS, 30, 2)
                    cv2.putText(display_frame, "TRACKING", (tx + 15, ty - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TRACKING_COLOR, 2)

                if self.tracking_enabled and self.selected_target:
                    sx, sy, sw, sh = self.selected_target
                    cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh),
                                  TRACKING_COLOR, 3)

                # Status indicator
                indicator_color = GREEN_LIGHT if motion_detected else RED_LIGHT
                cv2.circle(display_frame, STATUS_INDICATOR_POSITION,
                          STATUS_INDICATOR_SIZE // 2, indicator_color, -1)
                cv2.circle(display_frame, STATUS_INDICATOR_POSITION,
                          STATUS_INDICATOR_SIZE // 2, (255, 255, 255), 2)

                # Status text
                if self.tracking_enabled and self.selected_target:
                    status = "TRACKING PERSON"
                elif self.tracking_enabled:
                    status = "CLICK A PERSON TO TRACK"
                elif motion_detected and person_detected:
                    status = "MOTION + PERSON"
                elif motion_detected:
                    status = "MOTION DETECTED"
                else:
                    status = "NO MOTION"
                cv2.putText(display_frame, status,
                          (STATUS_INDICATOR_POSITION[0] + 40, STATUS_INDICATOR_POSITION[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATUS_COLOR, 2)

                # Help text at bottom
                frame_height = display_frame.shape[0]
                cv2.putText(display_frame, "Press 'q' to quit | Left-click: track | Right-click: stop",
                          (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cleanup()
        if display:
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description='PTZOptics Camera Motion Detection and Person Tracking',
        epilog='Example: python main.py 192.168.1.100 --track'
    )
    parser.add_argument('camera_ip', help='IP address of PTZOptics camera')
    parser.add_argument('--sensitivity', type=int, default=DEFAULT_SENSITIVITY,
                       help=f'Motion sensitivity (lower = more sensitive, default: {DEFAULT_SENSITIVITY})')
    parser.add_argument('--min-area', type=int, default=DEFAULT_MIN_AREA,
                       help=f'Minimum motion area in pixels (default: {DEFAULT_MIN_AREA})')
    parser.add_argument('--stream', type=int, default=2, choices=[1, 2],
                       help='Camera stream: 1 for higher quality, 2 for lower latency (default: 2)')
    parser.add_argument('--track', action='store_true',
                       help='Enable PTZ tracking - click a person to follow them')
    parser.add_argument('--username', default='admin',
                       help='Camera username for HTTP-CGI authentication (default: admin)')
    parser.add_argument('--password', default='admin',
                       help='Camera password for HTTP-CGI authentication (default: admin)')

    args = parser.parse_args()

    detector = PTZMotionDetector(
        camera_ip=args.camera_ip,
        sensitivity=args.sensitivity,
        min_area=args.min_area,
        stream=f"stream{args.stream}",
        enable_tracking=args.track,
        username=args.username,
        password=args.password
    )

    try:
        detector.run_detection(display=True, save_detections=False)
    except KeyboardInterrupt:
        print("\nPTZOptics motion detection stopped by user")

if __name__ == "__main__":
    main()
