"""
PTZOptics Camera Motion Detection Example

This example demonstrates how to connect to a PTZOptics camera and perform
real-time motion detection using OpenCV. The system can detect both general
motion and specifically identify when people are moving in the camera's view.

Requirements:
- PTZOptics camera with RTSP streaming enabled
- Network connectivity to the camera
- OpenCV Python package
"""

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
DEFAULT_SENSITIVITY = 25         # Motion sensitivity (lower = more sensitive)
DEFAULT_MIN_AREA = 300           # Minimum motion area in pixels to trigger detection
MOTION_HOLD_FRAMES = 10          # Frames to hold motion state (reduces flickering)

# Face detection parameters
FACE_DETECTION_INTERVAL = 5      # Run face detection every N frames (for performance)
FACE_HOLD_FRAMES = 15            # Frames to hold face detection state

# Background subtractor settings
BACKGROUND_DETECT_SHADOWS = True # Enable shadow detection in background subtraction
MORPH_KERNEL_SIZE = (3, 3)       # Kernel size for morphological operations

# Person detection settings
PERSON_SCALE_FACTOR = 1.1        # Scale factor for person detection
PERSON_MIN_NEIGHBORS = 3         # Minimum neighbors for person detection
PERSON_MIN_SIZE = (30, 30)       # Minimum size for person detection

# Full body detection settings
BODY_SCALE_FACTOR = 1.05         # Scale factor for full body detection
BODY_MIN_NEIGHBORS = 2           # Minimum neighbors for full body detection
BODY_MIN_SIZE = (50, 100)        # Minimum size for full body detection (wider aspect)

# PTZ tracking settings
PTZ_DEAD_ZONE = 50               # Pixels from center before camera moves
PTZ_SPEED_DIVISOR = 30           # Higher = slower speed scaling (distance / divisor = speed)
PTZ_MAX_SPEED = 24               # Maximum pan speed (PTZOptics range: 1-24)
PTZ_TILT_MAX_SPEED = 20          # Maximum tilt speed (PTZOptics range: 1-20)

# Display settings
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
    PTZOptics Camera Motion Detection System
    
    This class provides real-time motion detection capabilities for PTZOptics cameras
    using RTSP streaming. It combines background subtraction for motion detection 
    with Haar cascade classifiers for face detection.
    """
    
    def __init__(self, camera_ip, sensitivity=DEFAULT_SENSITIVITY, min_area=DEFAULT_MIN_AREA,
                 stream="stream2", enable_tracking=False, username="admin", password="admin"):
        """
        Initialize the motion detector for a PTZOptics camera.

        Args:
            camera_ip (str): IP address of the PTZOptics camera
            sensitivity (int): Motion sensitivity threshold (lower = more sensitive)
            min_area (int): Minimum area in pixels to consider as motion
            stream (str): RTSP stream to use (stream1 for higher quality, stream2 for lower latency)
            enable_tracking (bool): Enable PTZ tracking to follow detected targets
            username (str): Camera username for HTTP-CGI authentication
            password (str): Camera password for HTTP-CGI authentication
        """
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
        self.tracking_target = None  # (x, y) of current tracking target for display

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

        # Motion smoothing to reduce flickering between motion/no-motion states
        self.motion_frames_count = 0
        self.motion_hold_frames = MOTION_HOLD_FRAMES

        # Face detection frame skipping and persistence
        self.face_detection_counter = 0
        self.face_frames_count = 0
        self.face_hold_frames = FACE_HOLD_FRAMES
        self.last_face_result = (False, [])

        # Click-to-track: user selects a specific target by clicking on it
        self.selected_target = None        # (x, y, w, h) of the user-selected target
        self.selected_target_center = None # (x, y) center of selected target
        self.click_point = None            # Raw click coordinates for initial selection
        
    def on_mouse_click(self, event, x, y, flags, param):
        """
        Mouse callback for click-to-track.

        Left-click on a detected face or motion area to lock tracking onto it.
        Right-click to clear the selection and return to automatic tracking.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click clears selection
            self.selected_target = None
            self.selected_target_center = None
            self.click_point = None
            print("Target selection cleared - returning to automatic tracking")

    def select_target_from_click(self, detections):
        """
        Match a mouse click to a detected object (body, face, or motion area).

        Selects the first detection whose bounding box contains the click point.
        Pass detections in priority order (bodies first, then faces, then motion).

        Args:
            detections: List of (x, y, w, h) detection rectangles in priority order
        """
        if self.click_point is None:
            return

        cx, cy = self.click_point
        self.click_point = None

        for (x, y, w, h) in detections:
            if x <= cx <= x + w and y <= cy <= y + h:
                self.selected_target = (x, y, w, h)
                self.selected_target_center = (x + w // 2, y + h // 2)
                print(f"Selected target at ({x}, {y}, {w}x{h})")
                return

        print("No detected target at click location")

    def update_selected_target(self, detections):
        """
        Update the selected target position by finding the closest match
        in the current frame's detections.

        Uses overlap (IoU) to track the same object across frames.

        Args:
            detections: List of (x, y, w, h) detection rectangles

        Returns:
            tuple or None: (target_x, target_y) center of the matched target
        """
        if self.selected_target is None:
            return None

        sx, sy, sw, sh = self.selected_target
        best_match = None
        best_overlap = 0

        for (x, y, w, h) in detections:
            # Calculate intersection
            ix1 = max(sx, x)
            iy1 = max(sy, y)
            ix2 = min(sx + sw, x + w)
            iy2 = min(sy + sh, y + h)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                union = sw * sh + w * h - intersection
                overlap = intersection / union if union > 0 else 0

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (x, y, w, h)

        if best_match and best_overlap > 0.1:
            x, y, w, h = best_match
            self.selected_target = best_match
            self.selected_target_center = (x + w // 2, y + h // 2)
            return self.selected_target_center
        else:
            # Target lost
            print("Selected target lost")
            self.selected_target = None
            self.selected_target_center = None
            return None

    def move_camera(self, direction, pan_speed=5, tilt_speed=5):
        """
        Send a PTZ move command via HTTP-CGI.

        Args:
            direction (str): Movement direction (left, right, up, down, leftup, rightup, leftdown, rightdown)
            pan_speed (int): Pan speed 1-24
            tilt_speed (int): Tilt speed 1-20
        """
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
        """
        Move camera to center on a target position.

        Calculates the offset from frame center and sends appropriate PTZ commands
        to move the camera toward the target. Uses a dead zone to avoid jitter
        and scales speed based on distance from center.

        Args:
            target_x (int): Target X position in pixels
            target_y (int): Target Y position in pixels
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame
        """
        self.tracking_target = (target_x, target_y)

        center_x = frame_width // 2
        center_y = frame_height // 2

        dx = target_x - center_x
        dy = target_y - center_y

        # Dead zone - don't move if target is close enough to center
        if abs(dx) < PTZ_DEAD_ZONE and abs(dy) < PTZ_DEAD_ZONE:
            self.stop_camera()
            return

        # Scale speed based on how far off-center
        max_offset = max(abs(dx), abs(dy))
        pan_speed = min(PTZ_MAX_SPEED, max(1, int(max_offset / PTZ_SPEED_DIVISOR)))
        tilt_speed = min(PTZ_TILT_MAX_SPEED, max(1, int(max_offset / PTZ_SPEED_DIVISOR)))

        # Determine combined direction for diagonal movement
        h_dir = ""
        v_dir = ""
        if abs(dx) >= PTZ_DEAD_ZONE:
            h_dir = "right" if dx > 0 else "left"
        if abs(dy) >= PTZ_DEAD_ZONE:
            v_dir = "down" if dy > 0 else "up"

        if h_dir and v_dir:
            # Diagonal: PTZOptics uses leftup, rightup, leftdown, rightdown
            self.move_camera(h_dir + v_dir, pan_speed, tilt_speed)
        elif h_dir:
            self.move_camera(h_dir, pan_speed, tilt_speed)
        elif v_dir:
            self.move_camera(v_dir, pan_speed, tilt_speed)

    def get_camera_frame(self):
        """
        Retrieve a single frame from the PTZOptics camera via RTSP.
        
        Establishes RTSP connection on first call and maintains the stream
        for subsequent frame captures.
        
        Returns:
            numpy.ndarray: Camera frame as BGR image, or None if retrieval failed
        """
        if self.cap is None:
            print(f"Connecting to RTSP stream: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                print("Failed to open RTSP stream")
                return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("Failed to read frame from RTSP stream, attempting reconnection...")
            # Try to reconnect
            self.cap.release()
            self.cap = None
            return None
    
    def detect_motion(self, frame):
        """
        Detect motion in the current frame using background subtraction.
        
        This method uses MOG2 background subtractor to identify moving objects
        by comparing the current frame against a learned background model.
        
        Args:
            frame (numpy.ndarray): Input frame from camera
            
        Returns:
            tuple: (motion_detected, motion_areas, foreground_mask)
                - motion_detected (bool): True if motion above threshold detected
                - motion_areas (list): List of (x,y,w,h) bounding rectangles
                - foreground_mask (numpy.ndarray): Binary mask of detected motion
        """
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
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
        """Apply smoothing to motion detection to reduce flickering"""
        if raw_motion_detected:
            self.motion_frames_count = self.motion_hold_frames
            return True
        else:
            if self.motion_frames_count > 0:
                self.motion_frames_count -= 1
                return True
            else:
                return False
    
    def detect_person(self, frame):
        """
        Detect faces in the frame using Haar cascade classifier.
        
        Uses OpenCV's pre-trained face detector to identify human faces.
        More reliable than full-body detection for people at desks or partially visible.
        
        Args:
            frame (numpy.ndarray): Input frame from camera
            
        Returns:
            tuple: (person_detected, people_rectangles)
                - person_detected (bool): True if one or more faces detected
                - people_rectangles (list): List of (x,y,w,h) detection rectangles
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade classifier
        people = self.person_cascade.detectMultiScale(
            gray, 
            scaleFactor=PERSON_SCALE_FACTOR, 
            minNeighbors=PERSON_MIN_NEIGHBORS,
            minSize=PERSON_MIN_SIZE
        )
        
        return len(people) > 0, people

    def detect_bodies(self, frame):
        """
        Detect full human bodies in the frame using Haar cascade classifier.

        Args:
            frame (numpy.ndarray): Input frame from camera

        Returns:
            tuple: (bodies_detected, body_rectangles)
                - bodies_detected (bool): True if one or more bodies detected
                - body_rectangles (list): List of (x,y,w,h) detection rectangles
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bodies = self.body_cascade.detectMultiScale(
            gray,
            scaleFactor=BODY_SCALE_FACTOR,
            minNeighbors=BODY_MIN_NEIGHBORS,
            minSize=BODY_MIN_SIZE
        )

        return len(bodies) > 0, bodies

    def detect_person_with_persistence(self, frame):
        """
        Detect faces with frame skipping and result persistence for performance.
        
        Runs face detection every N frames and persists the result for smooth display.
        This reduces CPU usage while maintaining responsive face detection.
        
        Args:
            frame (numpy.ndarray): Input frame from camera
            
        Returns:
            tuple: (person_detected, people_rectangles)
                - person_detected (bool): True if faces detected (with persistence)
                - people_rectangles (list): List of (x,y,w,h) detection rectangles
        """
        self.face_detection_counter += 1
        
        # Run actual face detection every N frames
        if self.face_detection_counter >= FACE_DETECTION_INTERVAL:
            self.face_detection_counter = 0
            face_detected, faces = self.detect_person(frame)
            
            if face_detected:
                # Face found - reset persistence counter and store result
                self.face_frames_count = self.face_hold_frames
                self.last_face_result = (True, faces)
                return True, faces
            else:
                # No face found - but don't immediately clear if we were persisting
                if self.face_frames_count <= 0:
                    self.last_face_result = (False, [])
        
        # Check if we should persist previous face detection
        if self.face_frames_count > 0:
            self.face_frames_count -= 1
            # Return the persisted result with the original face rectangles
            return self.last_face_result
        else:
            return False, []
    
    def cleanup(self):
        """
        Clean up camera resources.

        Releases the RTSP video capture object and stops PTZ movement.
        Should be called when detection is finished.
        """
        if self.tracking_enabled:
            self.stop_camera()
        if self.cap is not None:
            self.cap.release()
    
    def run_detection(self, display=False, save_detections=False):
        """
        Run the main motion detection loop.
        
        Continuously captures frames from the camera and processes them for motion
        and person detection. Optionally displays results and saves detection images.
        
        Args:
            display (bool): Show live video feed with detection overlays
            save_detections (bool): Save images when person motion is detected
            
        Note:
            Press 'q' to quit when display is enabled, or use Ctrl+C to interrupt.
        """
        print(f"Starting PTZOptics motion detection for camera at {self.camera_ip}")
        if self.tracking_enabled:
            print("PTZ tracking enabled - click a detected body, face, or motion area to track it")
            print("Left-click to select a target, right-click to stop tracking")
        print("Press 'q' to quit the display window")

        if display and self.tracking_enabled:
            cv2.namedWindow("PTZOptics Motion Detection")
            cv2.setMouseCallback("PTZOptics Motion Detection", self.on_mouse_click)

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
            
            # Apply smoothing to motion detection
            motion_detected = self.smooth_motion_detection(raw_motion_detected)
            
            # Check for faces with frame skipping and persistence (runs independently of motion)
            person_detected, people = self.detect_person_with_persistence(frame)

            # Detect full human bodies
            bodies_detected, bodies = self.detect_bodies(frame)

            # PTZ tracking: only tracks when user clicks to select a target
            if self.tracking_enabled:
                # All clickable detections: bodies first, then faces, then motion
                all_clickable = list(bodies) + list(people) + list(motion_areas)
                self.select_target_from_click(all_clickable)

                if self.selected_target is not None:
                    target_pos = self.update_selected_target(all_clickable)
                    if target_pos:
                        self.track_target(target_pos[0], target_pos[1],
                                          frame.shape[1], frame.shape[0])
                    else:
                        self.tracking_target = None
                        self.stop_camera()
                else:
                    self.tracking_target = None
                    self.stop_camera()

            # Display if requested
            if display:
                display_frame = frame.copy()
                
                # Draw motion detection areas in green
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

                # Draw tracking visuals
                if self.tracking_enabled and self.tracking_target:
                    tx, ty = self.tracking_target
                    cv2.drawMarker(display_frame, (tx, ty), TRACKING_COLOR,
                                   cv2.MARKER_CROSS, 30, 2)
                    label = "TRACKING"
                    cv2.putText(display_frame, label, (tx + 15, ty - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TRACKING_COLOR, 2)

                # Highlight the user-selected target with a thicker box
                if self.tracking_enabled and self.selected_target:
                    sx, sy, sw, sh = self.selected_target
                    cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh),
                                  TRACKING_COLOR, 3)
                
                # Draw status indicator circle (red/green light)
                indicator_color = GREEN_LIGHT if motion_detected else RED_LIGHT
                cv2.circle(display_frame, STATUS_INDICATOR_POSITION, 
                          STATUS_INDICATOR_SIZE//2, indicator_color, -1)
                
                # Add a white border around the status indicator
                cv2.circle(display_frame, STATUS_INDICATOR_POSITION, 
                          STATUS_INDICATOR_SIZE//2, (255, 255, 255), 2)
                
                # Display detection status text
                if self.tracking_enabled and self.selected_target:
                    status = "TRACKING SELECTED TARGET"
                elif self.tracking_enabled:
                    status = "CLICK A TARGET TO TRACK"
                elif motion_detected and person_detected:
                    status = "MOTION + FACE"
                elif motion_detected:
                    status = "MOTION DETECTED"
                else:
                    status = "NO MOTION"
                cv2.putText(display_frame, status, (STATUS_INDICATOR_POSITION[0] + 40, STATUS_INDICATOR_POSITION[1] + 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATUS_COLOR, 2)
                
                # Add quit instruction at bottom of frame
                frame_height = display_frame.shape[0]
                cv2.putText(display_frame, "Press 'q' to quit", (10, frame_height - 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("PTZOptics Motion Detection", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Brief pause to prevent excessive CPU usage
            time.sleep(0.01)
        
        self.cleanup()
        if display:
            cv2.destroyAllWindows()

def main():
    """
    Main entry point for PTZOptics motion detection example.
    
    Parses command line arguments and starts the motion detection system.
    Run with --help for usage information.
    """
    parser = argparse.ArgumentParser(
        description='PTZOptics Camera Motion Detection Example',
        epilog='Example: python main.py 192.168.1.100'
    )
    parser.add_argument('camera_ip', help='IP address of PTZOptics camera')
    parser.add_argument('--sensitivity', type=int, default=DEFAULT_SENSITIVITY, 
                       help=f'Motion sensitivity (lower = more sensitive, default: {DEFAULT_SENSITIVITY})')
    parser.add_argument('--min-area', type=int, default=DEFAULT_MIN_AREA, 
                       help=f'Minimum motion area in pixels (default: {DEFAULT_MIN_AREA})')
    parser.add_argument('--stream', type=int, default=2, choices=[1, 2],
                       help='Camera stream to use: 1 for higher quality, 2 for lower latency (default: 2)')
    parser.add_argument('--track', action='store_true',
                       help='Enable PTZ tracking to follow detected faces/motion')
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
