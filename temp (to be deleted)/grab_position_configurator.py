#!/usr/bin/env python3
#coding=utf-8
"""
Grab Position Configurator for DOFBOT (Keyboard Control Version)
=================================================================
Configure precise grab positions for each slot using KEYBOARD CONTROL.

This tool:
1. Loads existing slot positions from slot_positions.json
2. Press F1-F6 to move to that slot position automatically
3. Use KEYBOARD to fine-tune the grab position and gripper rotation
4. Press SPACE to save the grab position

Keyboard Controls (Same as slot_position_configurator):
- W/S: Servo 1 (Base)
- A/D: Servo 2 (Shoulder)
- I/K: Servo 3 (Elbow)
- J/L: Servo 4 (Wrist pitch)
- U/O: Servo 5 (Wrist roll/GRIPPER ROTATION) ← IMPORTANT FOR GRABBING
- P/;: Servo 6 (Gripper open/close)
"""

import time
import cv2
import json
import os
import numpy as np
from datetime import datetime
from Arm_Lib import Arm_Device

# Try to import ONNX Runtime for object detection
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Object detection disabled.")

# Configuration
SLOT_POSITIONS_FILE = "slot_positions.json"
GRAB_POSITIONS_FILE = "grab_positions.json"
NUM_SERVOS = 6

# Movement configurations (same as slot_position_configurator)
ANGLE_STEP = 2
MOVE_SPEEDS = {
    'slow': 500,
    'normal': 300,
    'fast': 150,
    'instant': 50
}

MODEL_PATH = "models/best.onnx"
CONFIDENCE_THRESHOLD = 0.5
DETECTION_FRAME_SKIP = 5

# Class names
CLASS_NAMES = [
    'anchovies',
    'basil',
    'cheese',
    'chicken',
    'fresh_tomato',
    'shrimp'
]

class GrabConfigurator:
    def __init__(self):
        # Initialize DOFBOT
        print("Initializing DOFBOT...")
        self.arm = Arm_Device()
        time.sleep(0.1)

        # Initialize camera
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Current servo angles (initialize by reading current positions)
        self.current_angles = {}
        for i in range(NUM_SERVOS):
            servo_id = i + 1
            try:
                angle = self.arm.Arm_serial_servo_read(servo_id)
                self.current_angles[servo_id] = angle if angle is not None else 90
            except:
                self.current_angles[servo_id] = 90
            time.sleep(0.01)

        # Movement speed setting
        self.speed_mode = 'normal'
        self.current_speed = MOVE_SPEEDS['normal']

        # Angle step adjustment
        self.angle_step = ANGLE_STEP

        # Current slot being programmed
        self.current_slot = 1

        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Detection tracking
        self.detection_enabled = False
        self.detection_frame_counter = 0
        self.last_detections = []
        self.top_detections = []

        # Load slot positions
        self.slot_positions = self.load_slot_positions()
        if not self.slot_positions:
            print("\n✗ ERROR: No slot positions found!")
            print("  Please run slot_position_configurator.py first.")
            exit(1)

        print(f"✓ Loaded {len(self.slot_positions)} slot positions")

        # Load existing grab positions if available
        self.grab_positions = self.load_grab_positions()

        # Initialize object detection model
        self.model = None
        self.model_loaded = False

        print("\n" + "="*60)
        print("CHECKING MODEL FILES")
        print("="*60)

        if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                print(f"Loading ONNX model: {MODEL_PATH}...")
                self.model = ort.InferenceSession(MODEL_PATH)
                self.input_name = self.model.get_inputs()[0].name
                self.input_shape = self.model.get_inputs()[0].shape
                self.model_loaded = True
                print(f"✓✓✓ MODEL LOADED! ✓✓✓")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
        else:
            if not ONNX_AVAILABLE:
                print("✗ ONNX Runtime not available")
            elif not os.path.exists(MODEL_PATH):
                print(f"✗ Model file not found: {MODEL_PATH}")

        print("="*60)

        print("\n" + "="*60)
        print("GRAB POSITION CONFIGURATOR (KEYBOARD MODE)")
        print("="*60)
        print(f"Selected Slot: {self.current_slot}")

    def load_slot_positions(self):
        """Load slot positions from JSON"""
        if not os.path.exists(SLOT_POSITIONS_FILE):
            return {}
        try:
            with open(SLOT_POSITIONS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading slot positions: {e}")
            return {}

    def load_grab_positions(self):
        """Load grab positions from JSON"""
        if not os.path.exists(GRAB_POSITIONS_FILE):
            return {}
        try:
            with open(GRAB_POSITIONS_FILE, 'r') as f:
                positions = json.load(f)
            print(f"Loaded existing grab positions from {GRAB_POSITIONS_FILE}")
            return positions
        except Exception as e:
            print(f"Error loading grab positions: {e}")
            return {}

    def save_grab_positions(self):
        """Save grab positions to JSON file"""
        try:
            with open(GRAB_POSITIONS_FILE, 'w') as f:
                json.dump(self.grab_positions, f, indent=4)
            print(f"\n✓ Grab positions saved to {GRAB_POSITIONS_FILE}")
            return True
        except Exception as e:
            print(f"\n✗ Error saving grab positions: {e}")
            return False

    def move_servo(self, servo_id, angle):
        """Move a specific servo to an angle"""
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
        current = self.current_angles.get(servo_id, 90)
        new_angle = current + delta
        self.move_servo(servo_id, new_angle)

    def move_to_slot(self, slot_number):
        """Move to a slot position"""
        slot_key = f"slot_{slot_number}"

        if slot_key not in self.slot_positions:
            print(f"\n✗ Slot {slot_number} not found in slot positions!")
            return False

        print(f"\nMoving to Slot {slot_number} position...")

        slot_data = self.slot_positions[slot_key]
        angles = slot_data.get('angles', {})

        # Move each servo
        for servo_key, angle in angles.items():
            servo_id = int(servo_key.split('_')[1])
            self.move_servo(servo_id, angle)
            time.sleep(0.05)

        time.sleep(0.5)
        print(f"✓ Arrived at Slot {slot_number}")
        print("Now use keyboard to fine-tune the grab position!")
        print("IMPORTANT: Use U/O to adjust SERVO 5 (gripper rotation)")

        return True

    def change_speed(self, direction):
        """Change movement speed"""
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
        self.angle_step = max(1, min(10, self.angle_step + delta))
        print(f"Angle step size: {self.angle_step}°")

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
        self.top_detections = detections[:3]
        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            color = tuple([int(c) for c in np.random.RandomState(det['class_id']).randint(0, 255, 3)])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)

            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def display_current_angles(self, frame):
        """Overlay current servo angles and settings on camera frame"""
        h, w = frame.shape[:2]

        # FPS Display - Top Center
        fps_color = (0, 255, 0) if self.fps > 10 else (0, 165, 255) if self.fps > 5 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w//2 - 50, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # LEFT COLUMN - Servo angles
        y_offset = 20
        for servo_id in range(1, NUM_SERVOS + 1):
            angle = self.current_angles.get(servo_id, "N/A")
            # Highlight servo 5 (gripper rotation) and servo 6 (gripper)
            if servo_id == 5:
                color = (0, 255, 255)  # Cyan for servo 5
                thickness = 2
                text = f">S{servo_id}:{angle:3}° GRIP-ROT"
            elif servo_id == 6:
                color = (255, 0, 255)  # Magenta for servo 6
                thickness = 2
                text = f">S{servo_id}:{angle:3}° GRIPPER"
            else:
                color = (0, 255, 0)
                thickness = 1
                text = f" S{servo_id}:{angle:3}°"

            cv2.putText(frame, text, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
            y_offset += 18

        # RIGHT COLUMN - Settings
        right_x = w - 160
        y_offset = 20

        # Current slot indicator
        slot_color = (0, 255, 255)
        cv2.putText(frame, f"SLOT:{self.current_slot}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, slot_color, 2)
        y_offset += 22

        # Saved status
        slot_key = f"slot_{self.current_slot}"
        if slot_key in self.grab_positions:
            cv2.putText(frame, "SAVED", (right_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NOT SAVED", (right_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        y_offset += 20

        # Movement settings
        settings_color = (255, 165, 0)
        cv2.putText(frame, f"Spd:{self.speed_mode[:3].upper()}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, settings_color, 1)
        y_offset += 18
        cv2.putText(frame, f"Step:{self.angle_step}°", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, settings_color, 1)
        y_offset += 18

        # Detection status
        if self.model_loaded:
            det_color = (0, 255, 0) if self.detection_enabled else (100, 100, 100)
            det_text = "DET:ON" if self.detection_enabled else "DET:OFF"
            cv2.putText(frame, det_text, (right_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, det_color, 2)

        # TOP 3 PREDICTIONS
        if self.detection_enabled and self.top_detections:
            y_offset = h - 80
            cv2.putText(frame, "TOP 3 PREDICTIONS:", (w//2 - 80, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
            y_offset += 20

            for i, det in enumerate(self.top_detections):
                text = f"{i+1}. {det['class_name']}: {det['confidence']:.2f}"
                color = tuple([int(c) for c in np.random.RandomState(det['class_id']).randint(0, 255, 3)])
                cv2.putText(frame, text, (w//2 - 80, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                y_offset += 18

        return frame

    def display_instructions(self, frame):
        """Display controls on the camera frame"""
        h, w = frame.shape[:2]

        instructions = [
            "F1-F6:GoToSlot | WASD/IKJLUO/P;:Move",
            "U/O:Servo5(GRIP-ROT) | SPACE:SaveGrab",
            "T:Detect | +/-:Step | []:Speed | Q:Quit"
        ]

        y_offset = h - 55
        for instruction in instructions:
            cv2.putText(frame, instruction, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 16

        return frame

    def save_grab_position(self, slot_number):
        """Save current grab position for a specific slot"""
        print(f"\nSaving grab position for Slot {slot_number}...")

        # Convert current angles to expected format
        angles = {f"servo_{i}": self.current_angles[i] for i in range(1, NUM_SERVOS + 1)}

        # Save with timestamp
        slot_key = f"slot_{slot_number}"
        self.grab_positions[slot_key] = {
            "angles": angles,
            "gripper_rotation": self.current_angles.get(5, 90),  # Servo 5 = gripper rotation
            "gripper_close": self.current_angles.get(6, 90),     # Servo 6 = gripper open/close
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n✓ Grab position saved for Slot {slot_number}:")
        for servo, angle in angles.items():
            print(f"  {servo}: {angle}°")
        print(f"  *** Gripper rotation (S5): {self.current_angles.get(5, 90)}° ***")
        print(f"  *** Gripper close (S6): {self.current_angles.get(6, 90)}° ***")

        return self.save_grab_positions()

    def run(self):
        """Main loop with camera view and keyboard control"""
        print("\nStarting camera feed...")
        print("\n" + "="*60)
        print("GRAB POSITION CONFIGURATION - KEYBOARD CONTROL")
        print("="*60)
        print("\nWORKFLOW:")
        print("  1. Press F1-F6 to automatically move to that slot")
        print("  2. Use keyboard to fine-tune position:")
        print("     W/S - Servo 1 (Base)")
        print("     A/D - Servo 2 (Shoulder)")
        print("     I/K - Servo 3 (Elbow)")
        print("     J/L - Servo 4 (Wrist pitch)")
        print("     U/O - Servo 5 (GRIPPER ROTATION) ← IMPORTANT!")
        print("     P/; - Servo 6 (Gripper open/close)")
        print("  3. Press SPACE to save grab position for that slot")
        print("  4. Repeat for all 6 slots")
        print("\nSETTINGS:")
        print("  +/-     - Adjust step size")
        print("  [ / ]   - Slower / Faster speed")
        print("  T       - Toggle detection")
        print("  Q       - Quit")
        print("\nCurrent slot: {}".format(self.current_slot))
        print("="*60)

        try:
            while True:
                # Read camera frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Failed to read camera frame")
                    time.sleep(0.01)
                    continue

                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.fps_start_time
                if elapsed > 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = current_time

                # Run object detection if enabled
                if self.detection_enabled and self.model:
                    self.detection_frame_counter += 1

                    if self.detection_frame_counter >= DETECTION_FRAME_SKIP:
                        self.detection_frame_counter = 0
                        try:
                            input_tensor = self.preprocess_frame(frame)
                            outputs = self.model.run(None, {self.input_name: input_tensor})
                            self.last_detections = self.postprocess_detections(outputs, frame.shape)
                        except Exception as e:
                            print(f"Detection error: {e}")

                    if self.last_detections:
                        frame = self.draw_detections(frame, self.last_detections)
                else:
                    self.last_detections = []
                    self.top_detections = []

                # Display overlays
                frame = self.display_current_angles(frame)
                frame = self.display_instructions(frame)

                # Show frame
                cv2.imshow("Grab Position Configurator - KEYBOARD MODE", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                # === SLOT SELECTION (F1-F6) ===
                if key >= 190 and key <= 195:
                    self.current_slot = key - 189  # F1=1, F2=2, ..., F6=6
                    print(f"\n{'='*60}")
                    print(f"Selected Slot {self.current_slot}")
                    self.move_to_slot(self.current_slot)

                # === SERVO MOVEMENT (Video Game Style) ===
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

                # Servo 5 - Wrist roll/GRIPPER ROTATION (U/O) ← IMPORTANT
                elif key == ord('u') or key == ord('U'):
                    self.adjust_servo(5, self.angle_step)
                    print(f"  Gripper rotation: {self.current_angles.get(5, 90)}°")
                elif key == ord('o') or key == ord('O'):
                    self.adjust_servo(5, -self.angle_step)
                    print(f"  Gripper rotation: {self.current_angles.get(5, 90)}°")

                # Servo 6 - Gripper (P/;)
                elif key == ord('p') or key == ord('P'):
                    self.adjust_servo(6, self.angle_step)
                elif key == ord(';') or key == ord(':'):
                    self.adjust_servo(6, -self.angle_step)

                # === SETTINGS ADJUSTMENT ===
                # Step size
                elif key == ord('+') or key == ord('='):
                    self.adjust_step_size(1)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_step_size(-1)

                # Speed
                elif key == ord('[') or key == ord('{'):
                    self.change_speed('down')
                elif key == ord(']') or key == ord('}'):
                    self.change_speed('up')

                # Toggle detection
                elif key == ord('t') or key == ord('T'):
                    if self.model:
                        self.detection_enabled = not self.detection_enabled
                        status = "ON" if self.detection_enabled else "OFF"
                        print(f"Object detection: {status}")
                    else:
                        print("Object detection model not available")

                # === SAVE GRAB POSITION (SPACE) ===
                elif key == 32:  # SPACE bar
                    self.save_grab_position(self.current_slot)

                # === QUIT (Q) ===
                elif key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break

        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        del self.arm
        print("Done!")


def main():
    configurator = GrabConfigurator()
    configurator.run()


if __name__ == "__main__":
    main()
