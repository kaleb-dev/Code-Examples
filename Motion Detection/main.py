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
import threading
from datetime import datetime
import argparse
import requests
from requests.auth import HTTPDigestAuth

# =============================================================================
# CONFIGURATION VARIABLES - Modify these for your specific setup
# =============================================================================

# Camera connection settings
DEFAULT_RTSP_STREAM = "stream1"  # Use stream1 for native resolution, stream2 for lower latency
TARGET_FPS = 30                  # Target display frame rate

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

# Face detection settings (Haar cascade)
FACE_SCALE_FACTOR = 1.1          # Scale factor for face detection cascade
FACE_MIN_NEIGHBORS = 4           # Minimum neighbors for face detection (higher = fewer false positives)
FACE_MIN_SIZE = (30, 30)         # Minimum face size in pixels

# HOG body detection settings (cv2.HOGDescriptor - trained on human pedestrians only)
HOG_WIN_STRIDE = (8, 8)          # Sliding window stride (smaller = more thorough, slower)
HOG_PADDING = (8, 8)             # Padding around detection window
HOG_SCALE = 1.05                 # Image pyramid scale factor
HOG_HIT_THRESHOLD = 0.3          # SVM confidence threshold (higher = fewer false positives)

# Upper body detection settings (Haar cascade - catches seated/partial people)
UPPERBODY_SCALE_FACTOR = 1.1     # Scale factor for upper body detection
UPPERBODY_MIN_NEIGHBORS = 4      # Minimum neighbors (higher = fewer false positives)
UPPERBODY_MIN_SIZE = (40, 40)    # Minimum upper body size

# PTZ tracking settings
PTZ_DEAD_ZONE = 80               # Pixels from center before camera moves (wider = smoother)
PTZ_SPEED_DIVISOR = 50           # Higher = slower speed scaling for smoother movement
PTZ_MAX_SPEED = 12               # Maximum pan speed (PTZOptics range: 1-24, capped low for smoothness)
PTZ_TILT_MAX_SPEED = 10          # Maximum tilt speed (PTZOptics range: 1-20, capped low for smoothness)

# Display settings (0 = use native stream resolution)
DISPLAY_WIDTH = 0                # Window display width (0 = native frame size)
DISPLAY_HEIGHT = 0               # Window display height (0 = native frame size)
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
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._grabber_running = False

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

        # Face detection: Haar cascade for frontal faces
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Upper body detection: Haar cascade for seated/partial people
        self.upperbody_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        )

        # Full body detection: HOG descriptor with pre-trained people detector
        # This is OpenCV's most reliable human detector - trained specifically
        # on upright human bodies, won't false-detect on curtains/equipment/props
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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
        Update the locked target strictly using IoU overlap.

        Only matches detections that actually overlap with the current target
        bounding box. Will NOT jump to a person in a different part of the frame.
        If the person moves, the camera follows via PTZ commands which shifts
        the detection box naturally — IoU stays high for the same person.
        """
        if self.selected_target is None:
            return None

        sx, sy, sw, sh = self.selected_target
        best_match = None
        best_iou = 0

        for (x, y, w, h) in person_detections:
            # Calculate IoU overlap — the ONLY matching criterion
            ix1 = max(sx, x)
            iy1 = max(sy, y)
            ix2 = min(sx + sw, x + w)
            iy2 = min(sy + sh, y + h)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = sw * sh + w * h - intersection
                iou = intersection / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_match = (x, y, w, h)

        # Require real overlap — no distance-only matches
        if best_match and best_iou > 0.15:
            x, y, w, h = best_match
            self.selected_target = best_match
            self.selected_target_center = (x + w // 2, y + h // 2)
            self.target_lost_frames = 0
            return self.selected_target_center
        else:
            # No overlapping detection found
            self.target_lost_frames += 1
            if self.target_lost_frames < self.TARGET_LOST_THRESHOLD:
                # Hold last known position briefly (person may be between detection frames)
                return self.selected_target_center
            else:
                print("Target lost - no matching person detection")
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

    def start_frame_grabber(self):
        """
        Start a background thread that continuously grabs frames from RTSP.

        This decouples frame capture from processing so the display stays smooth
        at 30fps even when detection takes longer. The thread always holds the
        latest frame, discarding older ones automatically.
        """
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

        # Read stream properties
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"RTSP stream connected: {w}x{h} @ {fps:.1f}fps")

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._grabber_running = True
        self._grab_thread = threading.Thread(target=self._frame_grab_loop, daemon=True)
        self._grab_thread.start()
        return True

    def _frame_grab_loop(self):
        """Background thread that continuously reads the latest frame."""
        while self._grabber_running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                # Brief pause on read failure before retry
                time.sleep(0.01)

    def get_camera_frame(self):
        """Get the most recent frame from the background grabber thread."""
        if self._latest_frame is None:
            return None
        with self._frame_lock:
            return self._latest_frame.copy()

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

    def detect_faces_with_persistence(self, gray):
        """Detect faces with frame skipping and result persistence."""
        self.face_detection_counter += 1

        if self.face_detection_counter >= FACE_DETECTION_INTERVAL:
            self.face_detection_counter = 0
            raw_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=FACE_SCALE_FACTOR,
                minNeighbors=FACE_MIN_NEIGHBORS, minSize=FACE_MIN_SIZE
            )
            faces = self.scale_detections(raw_faces)
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

    def detect_bodies_with_persistence(self, frame, gray):
        """
        Detect human bodies using HOG people detector + upper body cascade.

        HOG (Histogram of Oriented Gradients) with the default people detector
        is OpenCV's most reliable human detector. It's trained specifically on
        upright human pedestrians and will NOT false-detect on curtains, stage
        equipment, or other non-human objects.

        Upper body cascade catches seated or partially visible people that
        HOG might miss.
        """
        self.body_detection_counter += 1

        if self.body_detection_counter >= BODY_DETECTION_INTERVAL:
            self.body_detection_counter = 0

            # Downscale frame for HOG detection (needs color or gray, works on both)
            if DETECTION_FRAME_WIDTH > 0 and frame.shape[1] > DETECTION_FRAME_WIDTH:
                scale = DETECTION_FRAME_WIDTH / frame.shape[1]
                small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                scale = 1.0
                small_frame = frame

            # HOG people detector — the gold standard for human body detection
            hog_rects, weights = self.hog.detectMultiScale(
                small_frame,
                winStride=HOG_WIN_STRIDE,
                padding=HOG_PADDING,
                scale=HOG_SCALE,
                hitThreshold=HOG_HIT_THRESHOLD
            )

            # Upper body cascade for seated/partial people
            upperbody_rects = self.upperbody_cascade.detectMultiScale(
                gray, scaleFactor=UPPERBODY_SCALE_FACTOR,
                minNeighbors=UPPERBODY_MIN_NEIGHBORS,
                minSize=UPPERBODY_MIN_SIZE
            )

            # Scale HOG detections back to full resolution
            bodies = []
            if len(hog_rects) > 0:
                for (x, y, w, h) in hog_rects:
                    bodies.append((int(x / scale), int(y / scale),
                                   int(w / scale), int(h / scale)))

            # Scale upper body detections back to full resolution
            upper_bodies = self.scale_detections(upperbody_rects)

            # Merge: add upper bodies that don't overlap with HOG detections
            for ub in upper_bodies:
                overlaps = False
                for b in bodies:
                    ix1 = max(ub[0], b[0])
                    iy1 = max(ub[1], b[1])
                    ix2 = min(ub[0] + ub[2], b[0] + b[2])
                    iy2 = min(ub[1] + ub[3], b[1] + b[3])
                    if ix2 > ix1 and iy2 > iy1:
                        overlaps = True
                        break
                if not overlaps:
                    bodies.append(ub)

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
        """Clean up camera resources, stop grabber thread, and stop PTZ movement."""
        self._grabber_running = False
        if hasattr(self, '_grab_thread') and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2)
        if self.tracking_enabled:
            self.stop_camera()
        if self.cap is not None:
            self.cap.release()

    def run_detection(self, display=False, save_detections=False):
        """Run the main motion detection and tracking loop at target FPS."""
        print(f"Starting PTZOptics motion detection for camera at {self.camera_ip}")
        if self.tracking_enabled:
            print("PTZ tracking enabled - click on a person (body/face) to lock and track them")
            print("Left-click to select, right-click to stop tracking")
        print("Press 'q' to quit the display window")

        # Start background frame grabber
        if not self.start_frame_grabber():
            print("Could not connect to camera. Exiting.")
            return

        # Wait for first frame
        for _ in range(50):
            if self.get_camera_frame() is not None:
                break
            time.sleep(0.1)

        window_name = "PTZOptics Motion Detection"
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Use native resolution or configured display size
            first_frame = self.get_camera_frame()
            if first_frame is not None:
                if DISPLAY_WIDTH > 0 and DISPLAY_HEIGHT > 0:
                    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                else:
                    cv2.resizeWindow(window_name, first_frame.shape[1], first_frame.shape[0])
            if self.tracking_enabled:
                cv2.setMouseCallback(window_name, self.on_mouse_click)

        frame_count = 0
        frame_interval = 1.0 / TARGET_FPS

        while True:
            loop_start = time.perf_counter()

            frame = self.get_camera_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            frame_count += 1

            # Detect motion
            raw_motion_detected, motion_areas, fg_mask = self.detect_motion(frame)
            motion_detected = self.smooth_motion_detection(raw_motion_detected)

            # Prepare downscaled grayscale once, shared by face and body detection
            gray = self.get_detection_frame(frame)

            # Detect human faces and bodies only (no generic motion tracking)
            person_detected, people = self.detect_faces_with_persistence(gray)
            bodies_detected, bodies = self.detect_bodies_with_persistence(frame, gray)

            # PTZ tracking: only tracks human detections, only when user clicks
            if self.tracking_enabled:
                # Only human detections — bodies first (larger/more stable), then faces
                person_detections = list(bodies) + list(people)
                self.select_target_from_click(person_detections)

                if self.selected_target is not None:
                    target_pos = self.update_selected_target(person_detections)
                    if target_pos and self.target_lost_frames == 0:
                        # Only send PTZ commands when we have a fresh match
                        self.track_target(target_pos[0], target_pos[1],
                                          frame.shape[1], frame.shape[0])
                    elif target_pos is None:
                        # Target truly lost — stop immediately
                        self.tracking_target = None
                        self.stop_camera()
                    else:
                        # Target temporarily lost — stop camera, don't pan randomly
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

                # Draw human body detections in magenta (HOG + upper body)
                for (x, y, w, h) in bodies:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), BODY_COLOR, 2)
                    cv2.putText(display_frame, "Person", (x, y - 10),
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

            # Pace to target FPS
            elapsed = time.perf_counter() - loop_start
            wait_time = frame_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

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
    parser.add_argument('--stream', type=int, default=1, choices=[1, 2],
                       help='Camera stream: 1 for native resolution, 2 for lower latency (default: 1)')
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
