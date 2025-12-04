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
from datetime import datetime
try:
    from pupil_apriltags import Detector
except ImportError:
    print("ERROR: pupil-apriltags not installed!")
    print("Install with: pip install pupil-apriltags")
    exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

# AprilTag Configuration
TAG_FAMILY = "tag36h11"  # Standard family
TAG_SIZE = 0.050  # 50mm in meters

# Row reference tags
ROW1_TAG_ID = 0
ROW2_TAG_ID = 1

# Shelf layout (6 cube slots in 2 rows, 3 columns)
SLOTS = {
    "R1C1": {"row": 1, "col": 1, "ingredient": "cheese"},
    "R1C2": {"row": 1, "col": 2, "ingredient": "chicken"},
    "R1C3": {"row": 1, "col": 3, "ingredient": "fresh_tomato"},
    "R2C1": {"row": 2, "col": 1, "ingredient": "anchovies"},
    "R2C2": {"row": 2, "col": 2, "ingredient": "basil"},
    "R2C3": {"row": 2, "col": 3, "ingredient": "shrimp"},
}

# Physical measurements (in meters) - YOU WILL NEED TO MEASURE THESE
CUBE_WIDTH = 0.08  # 80mm cube width (adjust to your actual cube size)
CUBE_SPACING = 0.01  # 10mm spacing between cubes

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

        # Initialize AprilTag detector
        self.detector = Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
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
                    "ingredient": slot_data["ingredient"]
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
                    "ingredient": slot_data["ingredient"]
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
                detections = self.detect_tags(gray)

                # Draw detections
                display_frame = frame.copy()
                self.draw_detections(display_frame, detections)

                # Check which tags are detected
                tags_detected = {d.tag_id for d in detections}
                row1_visible = ROW1_TAG_ID in tags_detected
                row2_visible = ROW2_TAG_ID in tags_detected

                # Display status
                status_text = f"Row 1 Tag (ID {ROW1_TAG_ID}): {'✓ DETECTED' if row1_visible else '✗ NOT FOUND'}"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0) if row1_visible else (0, 0, 255), 2)

                status_text = f"Row 2 Tag (ID {ROW2_TAG_ID}): {'✓ DETECTED' if row2_visible else '✗ NOT FOUND'}"
                cv2.putText(display_frame, status_text, (10, 60),
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

                elif key == ord(' ') and row1_visible and row2_visible:
                    # Capture calibration
                    for d in detections:
                        if d.tag_id == ROW1_TAG_ID:
                            row1_detection = d
                        elif d.tag_id == ROW2_TAG_ID:
                            row2_detection = d

                    if row1_detection and row2_detection:
                        print("\n✓ Both tags detected!")
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

                        print("\n✓ Calibration complete!")
                        print(f"✓ Configuration saved to: {OUTPUT_FILE}")
                        print(f"✓ {len(slot_positions)} slots configured:")
                        for slot_name, slot_data in slot_positions.items():
                            print(f"  - {slot_name}: {slot_data['ingredient']}")

                        break

        finally:
            self.close_camera()
            cv2.destroyAllWindows()

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
