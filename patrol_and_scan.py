#!/usr/bin/env python3
#coding=utf-8
"""
Patrol and Scan Script for ChefMate DOFBOT
==========================================

This script patrols through saved slot positions and uses YOLO to detect ingredients.
When it finds the requested ingredient with sufficient confidence, it can grab it.

Workflow:
1. Load slot positions from slot_positions.json (manually saved positions)
2. Load AprilTag calibration from apriltag_shelf_config.json (for position correction)
3. Enter patrol mode: visit each slot, detect ingredient with YOLO
4. When target ingredient found (confidence > threshold), grab it
"""

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from threading import Thread, Lock
from Arm_Lib import Arm_Device

# Try to import ONNX Runtime for object detection
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Object detection disabled.")

# Try to import AprilTag detector (optional - for position correction)
try:
    from pupil_apriltags import Detector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("Warning: pupil-apriltags not available. Position correction disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
SLOT_POSITIONS_FILE = "slot_positions.json"
APRILTAG_CONFIG_FILE = "apriltag_shelf_config.json"
MODEL_PATH = "models/best.onnx"

# Robot configuration
NUM_SERVOS = 6
MOVE_SPEED = 500  # Movement speed for patrol

# Detection configuration
CONFIDENCE_THRESHOLD = 0.5
DETECTION_FRAME_SKIP = 2  # Run detection every N frames
DETECTION_SAMPLES = 3  # Number of consecutive detections required to confirm

# Class names - ChefMate Dataset (must match training data)
CLASS_NAMES = [
    'anchovies',
    'basil',
    'cheese',
    'chicken',
    'fresh_tomato',
    'shrimp'
]

# Gripper positions
GRIPPER_OPEN = 180
GRIPPER_CLOSED = 90


# ============================================================================
# PATROL SCANNER
# ============================================================================

class PatrolScanner:
    """Patrol through slots and detect ingredients with YOLO"""

    def __init__(self):
        """Initialize patrol scanner"""
        print("\n" + "="*60)
        print("CHEFMATE PATROL & SCAN SYSTEM")
        print("="*60)

        # Initialize robot arm
        print("\nInitializing DOFBOT...")
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
        print("✓ Robot arm initialized")

        # Initialize camera
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Fix Qt platform warning
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Camera initialized")

        # Load slot positions
        self.slot_positions = self.load_slot_positions()
        if not self.slot_positions:
            print("\n✗ ERROR: No slot positions found!")
            print("  Please run slot_position_configurator.py first to save slot positions.")
            exit(1)

        print(f"✓ Loaded {len(self.slot_positions)} slot positions")

        # Load AprilTag calibration (optional)
        self.apriltag_config = self.load_apriltag_config()
        if self.apriltag_config:
            print("✓ AprilTag calibration loaded (position correction enabled)")
        else:
            print("⚠ AprilTag calibration not found (position correction disabled)")

        # Initialize YOLO model
        self.model = None
        self.model_loaded = False
        if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                print(f"\nLoading YOLO model: {MODEL_PATH}...")
                self.model = ort.InferenceSession(MODEL_PATH)
                self.input_name = self.model.get_inputs()[0].name
                self.input_shape = self.model.get_inputs()[0].shape
                self.model_loaded = True
                print(f"✓✓✓ YOLO MODEL LOADED! ✓✓✓")
                print(f"Model input: {self.input_name}, shape: {self.input_shape}")
            except Exception as e:
                print(f"✗ Failed to load YOLO model: {e}")
        else:
            if not ONNX_AVAILABLE:
                print("\n✗ ONNX Runtime not available")
            elif not os.path.exists(MODEL_PATH):
                print(f"\n✗ Model file not found: {MODEL_PATH}")

        # Detection tracking
        self.detection_frame_counter = 0
        self.last_detections = []

        print("="*60)

    def load_slot_positions(self):
        """Load saved slot positions from JSON"""
        if not os.path.exists(SLOT_POSITIONS_FILE):
            return {}
        try:
            with open(SLOT_POSITIONS_FILE, 'r') as f:
                positions = json.load(f)
            return positions
        except Exception as e:
            print(f"Error loading slot positions: {e}")
            return {}

    def load_apriltag_config(self):
        """Load AprilTag calibration configuration"""
        if not os.path.exists(APRILTAG_CONFIG_FILE):
            return None
        try:
            with open(APRILTAG_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading AprilTag config: {e}")
            return None

    def move_to_slot(self, slot_name):
        """Move arm to a specific slot position"""
        if slot_name not in self.slot_positions:
            print(f"✗ Slot {slot_name} not found in saved positions")
            return False

        slot_data = self.slot_positions[slot_name]
        angles = slot_data.get('angles', {})

        print(f"\nMoving to {slot_name}...")

        # Move each servo to saved position
        for servo_key, angle in angles.items():
            servo_id = int(servo_key.split('_')[1])
            self.move_servo(servo_id, angle)
            time.sleep(0.05)

        # Wait for movement to complete
        time.sleep(1.0)
        print(f"✓ Arrived at {slot_name}")
        return True

    def move_servo(self, servo_id, angle):
        """Move a specific servo to an angle"""
        angle = max(0, min(180, angle))
        try:
            self.arm.Arm_serial_servo_write(servo_id, angle, MOVE_SPEED)
            self.current_angles[servo_id] = angle
            return True
        except Exception as e:
            print(f"Error moving servo {servo_id}: {e}")
            return False

    def preprocess_frame(self, frame):
        """Preprocess frame for YOLO model"""
        input_height = self.input_shape[2] if len(self.input_shape) > 2 else 640
        input_width = self.input_shape[3] if len(self.input_shape) > 3 else 640

        resized = cv2.resize(frame, (input_width, input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(transposed, axis=0)

        return input_tensor

    def postprocess_detections(self, outputs, frame_shape):
        """Post-process YOLO outputs"""
        detections = []

        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs

        if len(output.shape) == 3:
            if output.shape[1] < output.shape[2]:
                output = output[0].T
            else:
                output = output[0]

        for detection in output:
            if len(detection) >= 4:
                x_center, y_center, width, height = detection[:4]
                class_probs = detection[4:]
                class_id = np.argmax(class_probs)
                confidence = float(class_probs[class_id])

                if confidence > CONFIDENCE_THRESHOLD:
                    h, w = frame_shape[:2]

                    if x_center <= 1.0 and y_center <= 1.0:
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                    else:
                        input_h = self.input_shape[2] if len(self.input_shape) > 2 else 640
                        input_w = self.input_shape[3] if len(self.input_shape) > 3 else 640
                        scale_x = w / input_w
                        scale_y = h / input_h
                        x1 = int((x_center - width / 2) * scale_x)
                        y1 = int((y_center - height / 2) * scale_y)
                        x2 = int((x_center + width / 2) * scale_x)
                        y2 = int((y_center + height / 2) * scale_y)

                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': int(class_id),
                        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'Class_{class_id}'
                    })

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections

    def detect_ingredient(self, num_samples=DETECTION_SAMPLES):
        """
        Detect ingredient at current position
        Takes multiple samples to ensure consistency
        Returns: (ingredient_name, confidence) or (None, 0) if nothing detected
        """
        if not self.model_loaded:
            print("✗ YOLO model not loaded, cannot detect")
            return None, 0.0

        print(f"  Scanning (taking {num_samples} samples)...", end='', flush=True)

        detection_votes = {}  # Count detections for each ingredient

        for sample in range(num_samples):
            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                input_tensor = self.preprocess_frame(frame)
                outputs = self.model.run(None, {self.input_name: input_tensor})
                detections = self.postprocess_detections(outputs, frame.shape)

                # Record the top detection
                if detections:
                    top_detection = detections[0]
                    ingredient = top_detection['class_name']
                    confidence = top_detection['confidence']

                    if ingredient not in detection_votes:
                        detection_votes[ingredient] = {'count': 0, 'total_conf': 0.0}

                    detection_votes[ingredient]['count'] += 1
                    detection_votes[ingredient]['total_conf'] += confidence

            except Exception as e:
                print(f"\n  Detection error: {e}")

            time.sleep(0.1)  # Small delay between samples

        print(" done!")

        # Find ingredient with most votes
        if detection_votes:
            best_ingredient = max(detection_votes.items(),
                                 key=lambda x: (x[1]['count'], x[1]['total_conf']))
            ingredient_name = best_ingredient[0]
            avg_confidence = best_ingredient[1]['total_conf'] / best_ingredient[1]['count']
            vote_count = best_ingredient[1]['count']

            print(f"  ✓ Detected: {ingredient_name} (confidence: {avg_confidence:.2f}, {vote_count}/{num_samples} samples)")
            return ingredient_name, avg_confidence
        else:
            print("  ✗ Nothing detected")
            return None, 0.0

    def grab_ingredient(self):
        """Execute grab sequence"""
        print("  Grabbing ingredient...")

        # Open gripper
        self.move_servo(6, GRIPPER_OPEN)
        time.sleep(0.5)

        # Close gripper
        self.move_servo(6, GRIPPER_CLOSED)
        time.sleep(0.5)

        print("  ✓ Grabbed!")

    def patrol_and_find(self, target_ingredient):
        """
        Patrol through all slots looking for target ingredient
        Returns: slot_name if found, None otherwise
        """
        print("\n" + "="*60)
        print(f"STARTING PATROL - Looking for: {target_ingredient}")
        print("="*60)

        # Get sorted slot names (R1C1, R1C2, ..., R2C3)
        slot_names = sorted(self.slot_positions.keys())

        for slot_name in slot_names:
            print(f"\n[{slot_name}]")

            # Move to slot
            if not self.move_to_slot(slot_name):
                continue

            # Detect what's there
            detected_ingredient, confidence = self.detect_ingredient()

            # Check if we found what we're looking for
            if detected_ingredient == target_ingredient and confidence >= CONFIDENCE_THRESHOLD:
                print(f"\n🎯 FOUND {target_ingredient} at {slot_name}!")
                return slot_name

            # Not found, continue patrol
            print(f"  ➜ Not here, continuing patrol...")

        print("\n" + "="*60)
        print(f"✗ {target_ingredient} not found in any slot")
        print("="*60)
        return None

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        del self.arm
        print("Done!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main patrol routine"""
    scanner = PatrolScanner()

    try:
        # Interactive mode
        while True:
            print("\n" + "="*60)
            print("PATROL MODE - MENU")
            print("="*60)
            print("Available ingredients:")
            for i, ingredient in enumerate(CLASS_NAMES, 1):
                print(f"  {i}. {ingredient}")
            print("\nOptions:")
            print("  Enter ingredient name to search for it")
            print("  Type 'list' to see saved slots")
            print("  Type 'quit' to exit")
            print("="*60)

            user_input = input("\nWhat are you looking for? ").strip().lower()

            if user_input == 'quit':
                break
            elif user_input == 'list':
                print("\nSaved slot positions:")
                for slot_name in sorted(scanner.slot_positions.keys()):
                    print(f"  - {slot_name}")
                continue
            elif user_input in [name.lower() for name in CLASS_NAMES]:
                # Find the matching class name (with correct case)
                target = next(name for name in CLASS_NAMES if name.lower() == user_input)

                # Start patrol
                found_slot = scanner.patrol_and_find(target)

                if found_slot:
                    grab = input(f"\nGrab {target} from {found_slot}? (y/n): ").strip().lower()
                    if grab == 'y':
                        scanner.grab_ingredient()
                        print(f"✓ {target} grabbed successfully!")
            else:
                print(f"✗ Unknown ingredient: {user_input}")
                print("Please choose from the available ingredients list.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        scanner.cleanup()


if __name__ == "__main__":
    main()
