#!/usr/bin/env python3
#coding=utf-8
"""
Test Grab System V2 - Multi-Waypoint Execution
===============================================
Test the configured grab sequences with proper waypoint execution.

Grab Sequence:
1. Move to scan position (detect ingredient)
2. Move to Waypoint 1 (Approach - above cube)
3. Open gripper
4. Move to Waypoint 2 (Grab - at cube level)
5. Close gripper
6. Move to Waypoint 3 (Lift - after grabbing)
7. Item stays in gripper for pick-and-place
"""

import time
import cv2
import json
import os
import numpy as np
from Arm_Lib import Arm_Device

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Object detection disabled.")

# Configuration
SLOT_POSITIONS_FILE = "slot_positions.json"
GRAB_POSITIONS_FILE = "grab_positions_v2.json"
MODEL_PATH = "models/best.onnx"
NUM_SERVOS = 6

# Movement speeds (higher value = slower)
MOVE_SPEED = 800       # Normal movement
SAFE_LEVEL_SPEED = 1000  # Moving to safe level position
GRAB_SPEED = 1000      # Grabbing motion (very slow)

# Delays between movements
MOVEMENT_DELAY = 3.0   # 3 seconds between movements

CONFIDENCE_THRESHOLD = 0.5
DETECTION_SAMPLES = 3

# Class names
CLASS_NAMES = [
    'anchovies',
    'basil',
    'cheese',
    'chicken',
    'fresh_tomato',
    'shrimp'
]

# Gripper positions (servo 6 is reversed!)
GRIPPER_OPEN = 0      # 0° = OPEN
# GRIPPER_CLOSED is dynamically determined from waypoint_2 servo_6 angle


class GrabTesterV2:
    def __init__(self):
        print("\n" + "="*60)
        print("GRAB SYSTEM TEST V2 - MULTI-WAYPOINT")
        print("="*60)

        # Initialize robot arm
        print("\nInitializing DOFBOT...")
        self.arm = Arm_Device()
        time.sleep(0.1)
        print("✓ Robot arm initialized")

        # Initialize camera
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Camera initialized")

        # Load positions
        self.slot_positions = self.load_json(SLOT_POSITIONS_FILE)
        self.grab_positions = self.load_json(GRAB_POSITIONS_FILE)

        if not self.slot_positions:
            print("\n✗ ERROR: No slot positions found!")
            print("  Run slot_position_configurator.py first")
            exit(1)

        if not self.grab_positions:
            print("\n✗ ERROR: No grab positions found!")
            print("  Run grab_position_configurator_v2.py first")
            exit(1)

        print(f"✓ Loaded {len(self.slot_positions)} slot positions")
        print(f"✓ Loaded {len(self.grab_positions)} grab sequences")

        # Current angles
        self.current_angles = {}
        self.read_current_angles()

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
                print("✓✓✓ YOLO MODEL LOADED! ✓✓✓")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
        else:
            print("\n✗ YOLO model not available - detection disabled")

        print("="*60)

    def load_json(self, filepath):
        """Load JSON file"""
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}

    def read_current_angles(self):
        """Read current servo angles"""
        for servo_id in range(1, NUM_SERVOS + 1):
            try:
                angle = self.arm.Arm_serial_servo_read(servo_id)
                if angle is not None:
                    self.current_angles[servo_id] = angle
            except:
                pass

    def move_servo(self, servo_id, angle, speed=MOVE_SPEED):
        """Move a specific servo to an angle"""
        angle = max(0, min(180, angle))
        try:
            self.arm.Arm_serial_servo_write(servo_id, angle, speed)
            self.current_angles[servo_id] = angle
            return True
        except Exception as e:
            print(f"Error moving servo {servo_id}: {e}")
            return False

    def move_to_position(self, angles_dict, speed=MOVE_SPEED, description="position", exclude_gripper=False):
        """Move to a position defined by angles dictionary"""
        print(f"  Moving to {description}...")

        # Log which servos will move
        servos_to_move = []
        for servo_key, angle in angles_dict.items():
            servo_id = int(servo_key.split('_')[1])
            if exclude_gripper and servo_id == 6:
                print(f"    SKIP Servo {servo_id} (gripper excluded, stays at {self.current_angles.get(6, 'unknown')}°)")
            else:
                servos_to_move.append((servo_id, angle))

        # Execute movements
        for servo_id, angle in servos_to_move:
            print(f"    MOVE Servo {servo_id} → {angle}°")
            self.move_servo(servo_id, angle, speed)
            time.sleep(0.05)

        time.sleep(0.5)
        print(f"  ✓ Reached {description}")

    def move_to_slot(self, slot_number):
        """Move to slot scanning position"""
        slot_key = f"slot_{slot_number}"
        if slot_key not in self.slot_positions:
            print(f"✗ Slot {slot_number} not found in slot positions")
            return False

        slot_data = self.slot_positions[slot_key]
        angles = slot_data.get('angles', {})
        # EXCLUDE GRIPPER when moving to scan position - we'll control it manually in grab sequence
        self.move_to_position(angles, MOVE_SPEED, f"Slot {slot_number} scan position", exclude_gripper=True)
        return True

    def preprocess_frame(self, frame):
        """Preprocess frame for YOLO"""
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
                    detections.append({
                        'confidence': confidence,
                        'class_id': int(class_id),
                        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'Class_{class_id}'
                    })

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections

    def detect_ingredient(self, num_samples=DETECTION_SAMPLES):
        """Detect ingredient at current position"""
        if not self.model_loaded:
            print("  ✗ YOLO model not loaded, skipping detection")
            return None, 0.0

        print(f"  Scanning (taking {num_samples} samples)...", end='', flush=True)

        detection_votes = {}

        for sample in range(num_samples):
            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                input_tensor = self.preprocess_frame(frame)
                outputs = self.model.run(None, {self.input_name: input_tensor})
                detections = self.postprocess_detections(outputs, frame.shape)

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

            time.sleep(0.1)

        print(" done!")

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

    def execute_grab_sequence(self, slot_number):
        """Execute 3-waypoint grab and pickup sequence"""
        slot_key = f"slot_{slot_number}"
        if slot_key not in self.grab_positions:
            print(f"✗ Grab sequence for Slot {slot_number} not configured")
            return False

        grab_data = self.grab_positions[slot_key]

        # Get waypoint 2 to extract the closed gripper angle
        waypoint_2 = grab_data.get('waypoint_2_grab')
        if not waypoint_2:
            print("✗ Waypoint 2 not found!")
            return False

        # Get the gripper closed angle from waypoint 2
        gripper_closed_angle = waypoint_2.get('servo_6', 136)  # Default to 136 if not found
        print(f"\n  Gripper closed angle from WP2: {gripper_closed_angle}°")

        print("\n  Executing grab sequence:")

        # Step 1: Open gripper FIRST (before any movement)
        print("\n  [1/5] Opening gripper...")
        current_gripper = self.current_angles.get(6, 'unknown')
        print(f"        Current gripper angle: {current_gripper}°")
        print(f"        Moving gripper to: {GRIPPER_OPEN}° (OPEN)")
        self.move_servo(6, GRIPPER_OPEN, GRAB_SPEED)
        time.sleep(1.0)
        actual_gripper = self.current_angles.get(6, 'unknown')
        print(f"        Gripper now at: {actual_gripper}°")

        # Step 2: Move to Waypoint 1 (Safe Level Position - outside shelf)
        # EXCLUDE GRIPPER from this movement
        print("\n  [2/5] Moving to Safe Level Position (Waypoint 1)...")
        print("        (Outside shelf, at cube height)")
        waypoint_1 = grab_data.get('waypoint_1_safe_level')
        if not waypoint_1:
            print("✗ Waypoint 1 not found!")
            return False
        print("        Waypoint 1 Target Angles:")
        for servo_key, angle in sorted(waypoint_1.items()):
            servo_id = int(servo_key.split('_')[1])
            print(f"          {servo_key}: {angle}°")
        self.move_to_position(waypoint_1, SAFE_LEVEL_SPEED, "safe level position", exclude_gripper=True)
        print(f"  Waiting {MOVEMENT_DELAY} seconds...")
        time.sleep(MOVEMENT_DELAY)

        # Step 3: Move to Waypoint 2 (Grab Position - inside shelf)
        # EXCLUDE GRIPPER - keep it open!
        print("\n  [3/5] Moving to Grab Position (Waypoint 2)...")
        print("        (Inside shelf, at cube)")
        print("        (Gripper will stay OPEN - servo 6 excluded)")

        print("        Waypoint 2 Target Angles:")
        for servo_key, angle in sorted(waypoint_2.items()):
            servo_id = int(servo_key.split('_')[1])
            status = "(EXCLUDED)" if servo_id == 6 else ""
            print(f"          {servo_key}: {angle}° {status}")

        self.move_to_position(waypoint_2, GRAB_SPEED, "grab position", exclude_gripper=True)
        print(f"  Waiting {MOVEMENT_DELAY} seconds...")
        time.sleep(MOVEMENT_DELAY)

        # Step 4: Close gripper to grab (use angle from waypoint 2)
        print(f"\n  [4/5] Closing gripper to grab (angle: {gripper_closed_angle}°)...")
        self.move_servo(6, gripper_closed_angle, GRAB_SPEED)
        time.sleep(2.0)

        # Step 5: Pickup sequence - return to Waypoint 1 with gripper CLOSED
        print("\n  [5/5] Pickup - returning to Safe Level Position (Waypoint 1)...")
        print("        (Gripper stays CLOSED - servo 6 excluded)")
        print("        Waypoint 1 Target Angles:")
        for servo_key, angle in sorted(waypoint_1.items()):
            servo_id = int(servo_key.split('_')[1])
            status = "(EXCLUDED - stays closed)" if servo_id == 6 else ""
            print(f"          {servo_key}: {angle}° {status}")
        self.move_to_position(waypoint_1, SAFE_LEVEL_SPEED, "safe level position (pickup)", exclude_gripper=True)
        print(f"  Waiting {MOVEMENT_DELAY} seconds...")
        time.sleep(MOVEMENT_DELAY)

        # Final step: Open gripper after reaching safe position
        print("\n  [FINAL] Opening gripper after reaching safe position...")
        print(f"        Moving gripper to: {GRIPPER_OPEN}° (OPEN)")
        self.move_servo(6, GRIPPER_OPEN, GRAB_SPEED)
        time.sleep(1.0)
        actual_gripper = self.current_angles.get(6, 'unknown')
        print(f"        Gripper now at: {actual_gripper}°")

        print("\n  ✓ Grab and pickup sequence complete!")
        print("  ℹ Item picked up and gripper opened at safe position")
        return True

    def release_sequence(self):
        """Release the grabbed item"""
        print("  Releasing item...")
        self.move_servo(6, GRIPPER_OPEN, GRAB_SPEED)
        time.sleep(0.5)
        print("  ✓ Released!")

    def test_slot(self, slot_number):
        """Test grabbing from a specific slot"""
        print("\n" + "="*60)
        print(f"TESTING SLOT {slot_number}")
        print("="*60)

        # Step 1: Move to scan position
        print(f"\n[STEP 1/3] Moving to scan position...")
        if not self.move_to_slot(slot_number):
            return False

        # Step 2: Detect ingredient
        print(f"\n[STEP 2/3] Detecting ingredient...")
        ingredient, confidence = self.detect_ingredient()

        if ingredient:
            print(f"  Found: {ingredient}")
        else:
            print("  No ingredient detected (continuing anyway)")

        # Step 3: Execute grab sequence
        print(f"\n[STEP 3/3] Executing grab sequence...")
        if not self.execute_grab_sequence(slot_number):
            return False

        print("\n" + "="*60)
        print(f"✓ SLOT {slot_number} GRAB TEST COMPLETE!")
        print("  Item is now in gripper - ready for pick-and-place to tray")
        print("="*60)

        return True

    def interactive_test(self):
        """Interactive testing mode"""
        print("\n" + "="*60)
        print("INTERACTIVE GRAB TEST MODE")
        print("="*60)

        # Show available slots
        print("\nConfigured slots:")
        slot_numbers = []
        for slot_key in sorted(self.slot_positions.keys()):
            slot_num = slot_key.split('_')[1]
            has_grab = slot_key in self.grab_positions
            status = "✓ Ready" if has_grab else "✗ No grab sequence"
            print(f"  Slot {slot_num}: {status}")
            if has_grab:
                slot_numbers.append(int(slot_num))

        if not slot_numbers:
            print("\n✗ No slots with grab sequences configured!")
            return

        while True:
            print("\n" + "="*60)
            print("MENU")
            print("="*60)
            print("  1-6: Test grab from slot 1-6")
            print("  r:   Release gripper (drop item)")
            print("  q:   Quit")
            print("="*60)

            choice = input("\nYour choice: ").strip().lower()

            if choice == 'q':
                print("Exiting...")
                break
            elif choice == 'r':
                self.release_sequence()
            elif choice.isdigit() and 1 <= int(choice) <= 6:
                slot_num = int(choice)
                if slot_num in slot_numbers:
                    self.test_slot(slot_num)
                    print("\n  ℹ Item remains in gripper (ready for pick-and-place)")
                else:
                    print(f"✗ Slot {slot_num} does not have a grab sequence configured")
            else:
                print("Invalid choice!")

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        del self.arm
        print("Done!")


def main():
    tester = GrabTesterV2()

    try:
        tester.interactive_test()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
