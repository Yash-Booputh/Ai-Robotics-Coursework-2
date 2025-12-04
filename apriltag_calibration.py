"""
AprilTag-Based Shelf Calibration System
========================================

This tool uses AprilTags to automatically calibrate the robot's shelf scanning positions.

Setup:
1. Print AprilTag IDs 0 and 1 (36h11 family) at 50mm x 50mm
2. Attach Tag 0 to top-left of Row 1
3. Attach Tag 1 to bottom-left of Row 2
4. Run this script to calibrate scan positions

The system will:
- Detect shelf tags and compute their 3D pose
- Calculate servo angles for scanning each cube slot
- Save calibration data to apriltag_shelf_config.json
"""

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
try:
    from pupil_apriltags import Detector
except ImportError:
    print("ERROR: pupil-apriltags not installed!")
    print("Install with: pip install pupil-apriltags")
    exit(1)

try:
    from Arm_Lib import Arm_Device
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False
    print("Warning: Arm_Lib not available. Robot arm control disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Robot arm configuration
NUM_SERVOS = 6
ANGLE_STEP = 2        # Angle step per key press
MOVE_SPEEDS = {
    'slow': 500,
    'normal': 300,
    'fast': 150,
    'instant': 50
}

# AprilTag Configuration
TAG_FAMILY = "tag36h11"  # Standard family
TAG_SIZE = 0.050  # 50mm in meters

# Row reference tags
ROW1_TAG_ID = 0
ROW2_TAG_ID = 1

# Shelf layout (6 cube slots in 2 rows, 3 columns)
# Note: Ingredients are NOT pre-assigned - they will be detected dynamically with YOLO
SLOTS = {
    "R1C1": {"row": 1, "col": 1},
    "R1C2": {"row": 1, "col": 2},
    "R1C3": {"row": 1, "col": 3},
    "R2C1": {"row": 2, "col": 1},
    "R2C2": {"row": 2, "col": 2},
    "R2C3": {"row": 2, "col": 3},
}

# Physical measurements (in meters) - YOU WILL NEED TO MEASURE THESE
CUBE_WIDTH = 0.03 # 80mm cube width (adjust to your actual cube size)
CUBE_SPACING = 0.06  # 10mm spacing between cubes

# Camera intrinsic parameters (will use default, but calibration is better)
# For better accuracy, run camera calibration separately
DEFAULT_CAMERA_PARAMS = {
    "fx": 600.0,  # Focal length X (pixels)
    "fy": 600.0,  # Focal length Y (pixels)
    "cx": 320.0,  # Principal point X (image center)
    "cy": 240.0,  # Principal point Y (image center)
}

# Output file
OUTPUT_FILE = "apriltag_shelf_config.json"


# ============================================================================
# APRILTAG DETECTOR
# ============================================================================

class ShelfCalibrator:
    """AprilTag-based shelf calibration system"""

    def __init__(self, camera_id=0):
        """Initialize calibrator with camera"""
        self.camera_id = camera_id
        self.camera = None

        # Initialize robot arm if available
        self.arm = None
        if ARM_AVAILABLE:
            print("Initializing DOFBOT...")
            self.arm = Arm_Device()
            time.sleep(0.1)

            # Initialize current servo angles
            self.current_angles = {}
            for i in range(NUM_SERVOS):
                servo_id = i + 1
                try:
                    angle = self.arm.Arm_serial_servo_read(servo_id)
                    self.current_angles[servo_id] = angle if angle is not None else 90
                except:
                    self.current_angles[servo_id] = 90
                time.sleep(0.01)

            # Movement settings
            self.speed_mode = 'normal'
            self.current_speed = MOVE_SPEEDS['normal']
            self.angle_step = ANGLE_STEP
            print("✓ Robot arm initialized")
        else:
            print("Robot arm not available - running in camera-only mode")

        # Initialize AprilTag detector
        # Adjusted settings to reduce "more than one new minima" errors
        self.detector = Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0,      # Reduced from 2.0 for better accuracy
            quad_sigma=0.8,         # Increased from 0.0 for noise reduction
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # Camera parameters
        self.camera_params = DEFAULT_CAMERA_PARAMS.copy()

        # Detected shelf configuration
        self.shelf_config = None

    def open_camera(self):
        """Open camera feed"""
        # Fix Qt platform warning - force X11 instead of Wayland
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")

        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("✓ Camera opened successfully")

    def close_camera(self):
        """Close camera feed"""
        if self.camera:
            self.camera.release()

    def move_servo(self, servo_id, angle):
        """Move a specific servo to an angle"""
        if not self.arm:
            return False
        # Clamp angle to safe range (0-180)
        angle = max(0, min(180, angle))
        try:
            self.arm.Arm_serial_servo_write(servo_id, angle, self.current_speed)
            self.current_angles[servo_id] = angle
            return True
        except Exception as e:
            print(f"Error moving servo {servo_id}: {e}")
            return False

    def adjust_servo(self, servo_id, delta):
        """Adjust servo by delta amount"""
        if not self.arm:
            return
        current = self.current_angles.get(servo_id, 90)
        new_angle = current + delta
        self.move_servo(servo_id, new_angle)

    def change_speed(self, direction):
        """Change movement speed"""
        if not self.arm:
            return
        speeds = list(MOVE_SPEEDS.keys())
        current_idx = speeds.index(self.speed_mode)

        if direction == 'up' and current_idx < len(speeds) - 1:
            current_idx += 1
        elif direction == 'down' and current_idx > 0:
            current_idx -= 1

        self.speed_mode = speeds[current_idx]
        self.current_speed = MOVE_SPEEDS[self.speed_mode]
        print(f"Speed mode: {self.speed_mode.upper()} ({self.current_speed}ms)")

    def adjust_step_size(self, delta):
        """Adjust angle step size"""
        if not self.arm:
            return
        self.angle_step = max(1, min(10, self.angle_step + delta))
        print(f"Angle step size: {self.angle_step}°")

    def display_arm_overlay(self, frame):
        """Overlay servo angles on camera frame if arm is available"""
        if not self.arm:
            return frame

        h, w = frame.shape[:2]
        y_offset = 20

        # Display servo angles on left side
        for servo_id in range(1, NUM_SERVOS + 1):
            angle = self.current_angles.get(servo_id, "N/A")
            text = f"S{servo_id}:{angle:3}°"
            cv2.putText(frame, text, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 18

        # Display settings on right side
        right_x = w - 120
        y_offset = 20
        cv2.putText(frame, f"Spd:{self.speed_mode[:3].upper()}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        y_offset += 18
        cv2.putText(frame, f"Step:{self.angle_step}°", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        return frame

    def display_controls(self, frame):
        """Display keyboard controls on frame"""
        h, w = frame.shape[:2]

        # Instructions at bottom
        instructions = []
        if self.arm:
            instructions.extend([
                "WASD/IKJLUO/P;:Move Servos",
                "+/-:Step | []:Speed"
            ])

        y_offset = h - 40
        for instruction in instructions:
            cv2.putText(frame, instruction, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 16

        return frame

    def detect_tags(self, gray_image):
        """Detect AprilTags in grayscale image"""
        # Camera parameters for pose estimation
        camera_params = [
            self.camera_params["fx"],
            self.camera_params["fy"],
            self.camera_params["cx"],
            self.camera_params["cy"]
        ]

        # Detect tags
        detections = self.detector.detect(
            gray_image,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE
        )

        return detections

    def draw_detections(self, image, detections):
        """Draw detected tags on image"""
        for detection in detections:
            # Draw tag outline
            corners = detection.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            # Draw tag ID
            center = detection.center.astype(int)
            cv2.putText(
                image,
                f"ID {detection.tag_id}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw coordinate axes if pose is available
            if detection.pose_R is not None:
                self.draw_axes(image, detection)

        return image

    def draw_axes(self, image, detection):
        """Draw 3D axes on tag"""
        # Get rotation and translation
        R = detection.pose_R
        t = detection.pose_t.flatten()

        # Define axis points (50mm length)
        axis_length = TAG_SIZE
        axes = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, -axis_length]  # Z-axis (blue, pointing toward camera)
        ])

        # Project 3D points to 2D
        camera_matrix = np.array([
            [self.camera_params["fx"], 0, self.camera_params["cx"]],
            [0, self.camera_params["fy"], self.camera_params["cy"]],
            [0, 0, 1]
        ])

        imgpts, _ = cv2.projectPoints(
            axes,
            R,
            t,
            camera_matrix,
            np.zeros(5)  # No distortion
        )

        imgpts = imgpts.astype(int).reshape(-1, 2)

        # Draw axes
        origin = tuple(imgpts[0])
        cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X red
        cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y green
        cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z blue

    def calculate_slot_positions(self, row1_tag, row2_tag):
        """
        Calculate 3D positions of all cube slots relative to detected tags

        Args:
            row1_tag: Detection for Row 1 reference tag (ID 0)
            row2_tag: Detection for Row 2 reference tag (ID 1)

        Returns:
            dict: Slot positions and orientations
        """
        slot_positions = {}

        # Calculate positions for Row 1 slots (relative to Tag 0)
        for slot_name, slot_data in SLOTS.items():
            if slot_data["row"] == 1:
                # Offset from Row 1 tag
                col_offset = (slot_data["col"] - 1) * (CUBE_WIDTH + CUBE_SPACING)

                # Position relative to tag (tag is at left edge of row)
                offset_x = col_offset + CUBE_WIDTH / 2  # Center of cube
                offset_y = 0.0
                offset_z = 0.0

                slot_positions[slot_name] = {
                    "reference_tag": ROW1_TAG_ID,
                    "offset": [offset_x, offset_y, offset_z],
                    "row": slot_data["row"],
                    "col": slot_data["col"]
                }

        # Calculate positions for Row 2 slots (relative to Tag 1)
        for slot_name, slot_data in SLOTS.items():
            if slot_data["row"] == 2:
                # Offset from Row 2 tag
                col_offset = (slot_data["col"] - 1) * (CUBE_WIDTH + CUBE_SPACING)

                # Position relative to tag
                offset_x = col_offset + CUBE_WIDTH / 2  # Center of cube
                offset_y = 0.0
                offset_z = 0.0

                slot_positions[slot_name] = {
                    "reference_tag": ROW2_TAG_ID,
                    "offset": [offset_x, offset_y, offset_z],
                    "row": slot_data["row"],
                    "col": slot_data["col"]
                }

        return slot_positions

    def calibrate_shelf(self):
        """
        Interactive calibration - detect tags and calculate shelf configuration
        """
        print("\n" + "="*60)
        print("APRILTAG SHELF CALIBRATION")
        print("="*60)
        print("\nInstructions:")
        print("1. Position robot so camera can see BOTH row tags (ID 0 and ID 1)")
        print("2. Press SPACE to capture and calibrate")
        print("3. Press 'q' to quit without saving")
        print("\nMake sure tags are clearly visible and well-lit!")

        if self.arm:
            print("\n" + "="*60)
            print("KEYBOARD CONTROLS (Video Game Style)")
            print("="*60)
            print("  W/S         - Servo 1 (Base rotation)")
            print("  A/D         - Servo 2 (Shoulder)")
            print("  I/K         - Servo 3 (Elbow)")
            print("  J/L         - Servo 4 (Wrist pitch)")
            print("  U/O         - Servo 5 (Wrist roll)")
            print("  P/;         - Servo 6 (Gripper)")
            print("  +/-         - Adjust step size")
            print("  [ / ]       - Slower / Faster speed")

        print("="*60 + "\n")

        self.open_camera()

        row1_detection = None
        row2_detection = None

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect tags
                try:
                    detections = self.detect_tags(gray)
                except Exception as e:
                    # Silently handle detection errors (e.g., "more than one new minima")
                    detections = []

                # Draw detections
                display_frame = frame.copy()
                self.draw_detections(display_frame, detections)

                # Add arm overlay if available
                display_frame = self.display_arm_overlay(display_frame)

                # Add controls overlay
                display_frame = self.display_controls(display_frame)

                # Check which tags are detected
                tags_detected = {d.tag_id for d in detections}
                row1_visible = ROW1_TAG_ID in tags_detected
                row2_visible = ROW2_TAG_ID in tags_detected

                # Calculate status text position based on arm availability
                status_y_start = 140 if self.arm else 30

                # Display status
                status_text = f"Row 1 Tag (ID {ROW1_TAG_ID}): {'✓ DETECTED' if row1_visible else '✗ NOT FOUND'}"
                cv2.putText(display_frame, status_text, (10, status_y_start),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0) if row1_visible else (0, 0, 255), 2)

                status_text = f"Row 2 Tag (ID {ROW2_TAG_ID}): {'✓ DETECTED' if row2_visible else '✗ NOT FOUND'}"
                cv2.putText(display_frame, status_text, (10, status_y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0) if row2_visible else (0, 0, 255), 2)

                # Instructions
                if row1_visible and row2_visible:
                    cv2.putText(display_frame, "Press SPACE to calibrate", (10, 450),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Position camera to see both tags", (10, 450),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(display_frame, "Press 'q' to quit", (10, 420),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show frame
                cv2.imshow("AprilTag Shelf Calibration", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nCalibration cancelled")
                    break

                # === ROBOT ARM CONTROLS ===
                # Servo 1 - Base rotation (W/S)
                elif key == ord('w') or key == ord('W'):
                    self.adjust_servo(1, self.angle_step)
                elif key == ord('s') or key == ord('S'):
                    self.adjust_servo(1, -self.angle_step)

                # Servo 2 - Shoulder (A/D)
                elif key == ord('a') or key == ord('A'):
                    self.adjust_servo(2, self.angle_step)
                elif key == ord('d') or key == ord('D'):
                    self.adjust_servo(2, -self.angle_step)

                # Servo 3 - Elbow (I/K)
                elif key == ord('i') or key == ord('I'):
                    self.adjust_servo(3, self.angle_step)
                elif key == ord('k') or key == ord('K'):
                    self.adjust_servo(3, -self.angle_step)

                # Servo 4 - Wrist pitch (J/L)
                elif key == ord('j') or key == ord('J'):
                    self.adjust_servo(4, self.angle_step)
                elif key == ord('l') or key == ord('L'):
                    self.adjust_servo(4, -self.angle_step)

                # Servo 5 - Wrist roll (U/O)
                elif key == ord('u') or key == ord('U'):
                    self.adjust_servo(5, self.angle_step)
                elif key == ord('o') or key == ord('O'):
                    self.adjust_servo(5, -self.angle_step)

                # Servo 6 - Gripper (P/;)
                elif key == ord('p') or key == ord('P'):
                    self.adjust_servo(6, self.angle_step)
                elif key == ord(';') or key == ord(':'):
                    self.adjust_servo(6, -self.angle_step)

                # Step size adjustment
                elif key == ord('+') or key == ord('='):
                    self.adjust_step_size(1)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_step_size(-1)

                # Speed adjustment
                elif key == ord('[') or key == ord('{'):
                    self.change_speed('down')
                elif key == ord(']') or key == ord('}'):
                    self.change_speed('up')

                elif key == ord(' '):
                    # Attempt calibration
                    print("\n[SPACE pressed] Attempting calibration...")

                    if not row1_visible or not row2_visible:
                        print("✗ Cannot calibrate - both tags must be visible!")
                        print(f"  Row 1 (ID {ROW1_TAG_ID}): {'✓ Visible' if row1_visible else '✗ Not visible'}")
                        print(f"  Row 2 (ID {ROW2_TAG_ID}): {'✓ Visible' if row2_visible else '✗ Not visible'}")
                        continue

                    # Capture calibration - find the detections with pose data
                    row1_detection = None
                    row2_detection = None

                    for d in detections:
                        if d.tag_id == ROW1_TAG_ID and d.pose_t is not None:
                            row1_detection = d
                        elif d.tag_id == ROW2_TAG_ID and d.pose_t is not None:
                            row2_detection = d

                    if row1_detection and row2_detection:
                        print("\n✓ Both tags detected with valid pose!")
                        print(f"  Row 1 Tag (ID {ROW1_TAG_ID}): Distance = {np.linalg.norm(row1_detection.pose_t):.3f}m")
                        print(f"  Row 2 Tag (ID {ROW2_TAG_ID}): Distance = {np.linalg.norm(row2_detection.pose_t):.3f}m")

                        # Calculate slot positions
                        slot_positions = self.calculate_slot_positions(row1_detection, row2_detection)

                        # Build configuration
                        self.shelf_config = {
                            "calibration_date": datetime.now().isoformat(),
                            "tag_family": TAG_FAMILY,
                            "tag_size_m": TAG_SIZE,
                            "cube_width_m": CUBE_WIDTH,
                            "cube_spacing_m": CUBE_SPACING,
                            "camera_params": self.camera_params,
                            "row1_tag": {
                                "id": ROW1_TAG_ID,
                                "pose_t": row1_detection.pose_t.tolist(),
                                "pose_R": row1_detection.pose_R.tolist()
                            },
                            "row2_tag": {
                                "id": ROW2_TAG_ID,
                                "pose_t": row2_detection.pose_t.tolist(),
                                "pose_R": row2_detection.pose_R.tolist()
                            },
                            "slots": slot_positions
                        }

                        # Save configuration
                        self.save_config()

                        print("\n" + "="*60)
                        print("✓✓✓ CALIBRATION COMPLETE! ✓✓✓")
                        print("="*60)
                        print(f"✓ Configuration saved to: {OUTPUT_FILE}")
                        print(f"✓ {len(slot_positions)} slots configured:")
                        for slot_name, slot_data in slot_positions.items():
                            print(f"  - {slot_name}: Row {slot_data['row']}, Col {slot_data['col']}")
                        print("\nNote: Ingredients are NOT pre-assigned.")
                        print("They will be detected dynamically during patrol with YOLO.")
                        print("="*60)

                        break
                    else:
                        print("✗ Tags detected but pose estimation failed!")
                        print("  Try improving lighting or tag visibility")

        finally:
            self.close_camera()
            cv2.destroyAllWindows()
            if self.arm:
                del self.arm

    def save_config(self):
        """Save calibration configuration to JSON file"""
        if self.shelf_config is None:
            print("ERROR: No configuration to save")
            return

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(self.shelf_config, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main calibration routine"""
    calibrator = ShelfCalibrator(camera_id=0)
    calibrator.calibrate_shelf()


if __name__ == "__main__":
    main()
