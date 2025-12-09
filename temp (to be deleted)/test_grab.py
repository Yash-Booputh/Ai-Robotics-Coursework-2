#!/usr/bin/env python3
#coding=utf-8
"""
Test Grab System for DOFBOT
============================
Test the configured grab positions by actually grabbing ingredients.

This script:
1. Loads slot positions (for scanning)
2. Loads grab positions (for grabbing)
3. Moves to a slot, detects ingredient
4. Executes grab sequence using saved grab position
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
GRAB_POSITIONS_FILE = "grab_positions.json"
MODEL_PATH = "models/best.onnx"
NUM_SERVOS = 6
MOVE_SPEED = 1000  # Slowed down for smoother movement
GRAB_SPEED = 1000  # Slower speed for precise grabbing
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

# Gripper positions
GRIPPER_OPEN = 180
GRIPPER_CLOSED = 90

class GrabTester:
    def __init__(self):
        print("\n" + "="*60)
        print("GRAB SYSTEM TEST")
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
            print("  Run grab_position_configurator.py first")
            exit(1)

        print(f"✓ Loaded {len(self.slot_positions)} slot positions")
        print(f"✓ Loaded {len(self.grab_positions)} grab positions")

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

    def move_to_position(self, angles_dict, speed=MOVE_SPEED, description="position"):
        """Move to a position defined by angles dictionary"""
        print(f"  Moving to {description}...")
        for servo_key, angle in angles_dict.items():
            servo_id = int(servo_key.split('_')[1])
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
        self.move_to_position(angles, MOVE_SPEED, f"Slot {slot_number} scan position")
        return True

    def move_to_grab_position(self, slot_number):
        """Move to grab position for a slot"""
        slot_key = f"slot_{slot_number}"
        if slot_key not in self.grab_positions:
            print(f"✗ Grab position for Slot {slot_number} not configured")
            return False

        grab_data = self.grab_positions[slot_key]
        angles = grab_data.get('angles', {})
        self.move_to_position(angles, GRAB_SPEED, f"Slot {slot_number} grab position")
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

    def grab_sequence(self):
        """Execute grab sequence - open, close, lift"""
        print("  Executing grab sequence...")

        # Step 1: Open gripper
        print("    1. Opening gripper...")
        self.move_servo(6, GRIPPER_OPEN, GRAB_SPEED)
        time.sleep(0.5)

        # Step 2: Close gripper to grab
        print("    2. Closing gripper to grab...")
        self.move_servo(6, GRIPPER_CLOSED, GRAB_SPEED)
        time.sleep(1.0)

        # Step 3: Lift slightly (move servo 3 - elbow up by 10 degrees)
        print("    3. Lifting...")
        current_elbow = self.current_angles.get(3, 90)
        self.move_servo(3, current_elbow - 10, GRAB_SPEED)
        time.sleep(0.5)

        print("  ✓ Grab sequence complete!")

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
        print(f"\n[1/4] Moving to scan position...")
        if not self.move_to_slot(slot_number):
            return False

        # Step 2: Detect ingredient
        print(f"\n[2/4] Detecting ingredient...")
        ingredient, confidence = self.detect_ingredient()

        if ingredient:
            print(f"  Found: {ingredient}")
        else:
            print("  No ingredient detected (continuing anyway)")

        # Step 3: Move to grab position
        print(f"\n[3/4] Moving to grab position...")
        if not self.move_to_grab_position(slot_number):
            return False

        # Step 4: Execute grab
        print(f"\n[4/4] Grabbing...")
        self.grab_sequence()

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
            status = "✓ Ready" if has_grab else "✗ No grab position"
            print(f"  Slot {slot_num}: {status}")
            if has_grab:
                slot_numbers.append(int(slot_num))

        if not slot_numbers:
            print("\n✗ No slots with grab positions configured!")
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
                    print(f"✗ Slot {slot_num} does not have a grab position configured")
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
    tester = GrabTester()

    try:
        tester.interactive_test()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
