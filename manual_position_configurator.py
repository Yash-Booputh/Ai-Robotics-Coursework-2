#!/usr/bin/env python3
#coding=utf-8
"""
Manual Position Configurator for DOFBOT
========================================
This tool allows you to physically move the robot arm by hand (torque off) and save positions.
Much more intuitive than keyboard control for precise positioning!

Features:
- Torque OFF mode - move arm freely by hand
- Live camera view with detection
- Save positions to slots with a key press
- View current servo angles in real-time
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
SLOTS_FILE = "slot_positions.json"
NUM_SERVOS = 6
MODEL_PATH = "models/best.onnx"
CONFIDENCE_THRESHOLD = 0.5
DETECTION_FRAME_SKIP = 5  # Run detection less frequently since we're manual

# Class names - ChefMate Dataset
CLASS_NAMES = [
    'anchovies',
    'basil',
    'cheese',
    'chicken',
    'fresh_tomato',
    'shrimp'
]

class ManualConfigurator:
    def __init__(self):
        # Initialize DOFBOT
        print("Initializing DOFBOT...")
        self.arm = Arm_Device()
        time.sleep(0.1)

        # Turn OFF torque on all servos - this allows manual movement!
        print("Turning torque OFF - you can now move the arm by hand!")
        try:
            self.arm.Arm_serial_set_torque(0)  # 0 = torque OFF, 1 = torque ON
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not disable torque: {e}")

        print("✓ Robot arm initialized - TORQUE OFF MODE")

        # Initialize camera
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Fix Qt platform warning
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Camera initialized")

        # Current servo angles (read in real-time)
        self.current_angles = {}

        # Current slot being programmed
        self.current_slot = 1  # Default to slot 1

        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Detection tracking
        self.detection_enabled = False
        self.detection_frame_counter = 0
        self.last_detections = []
        self.top_detections = []

        # Initialize YOLO model
        self.model = None
        self.model_loaded = False

        print("\n" + "="*60)
        print("CHECKING MODEL FILES")
        print("="*60)

        if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                print(f"Loading YOLO model: {MODEL_PATH}...")
                self.model = ort.InferenceSession(MODEL_PATH)
                self.input_name = self.model.get_inputs()[0].name
                self.input_shape = self.model.get_inputs()[0].shape
                self.model_loaded = True
                print(f"✓✓✓ MODEL LOADED! ✓✓✓")
                print(f"Model input: {self.input_name}, shape: {self.input_shape}")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
        else:
            if not ONNX_AVAILABLE:
                print("✗ ONNX Runtime not available")
            elif not os.path.exists(MODEL_PATH):
                print(f"✗ Model file not found: {MODEL_PATH}")

        print("="*60)

        # Load existing slot positions if available
        self.slot_positions = self.load_positions()

        print("\n" + "="*60)
        print("MANUAL POSITION CONFIGURATOR")
        print("="*60)
        print(f"Selected Slot: {self.current_slot}")

    def load_positions(self):
        """Load saved slot positions from JSON file"""
        if os.path.exists(SLOTS_FILE):
            try:
                with open(SLOTS_FILE, 'r') as f:
                    positions = json.load(f)
                print(f"Loaded existing positions from {SLOTS_FILE}")
                return positions
            except Exception as e:
                print(f"Error loading positions: {e}")
                return {}
        return {}

    def save_positions(self):
        """Save slot positions to JSON file"""
        try:
            with open(SLOTS_FILE, 'w') as f:
                json.dump(self.slot_positions, f, indent=4)
            print(f"\n✓ Positions saved to {SLOTS_FILE}")
            return True
        except Exception as e:
            print(f"\n✗ Error saving positions: {e}")
            return False

    def read_current_angles(self):
        """Read current servo angles from the robot"""
        for servo_id in range(1, NUM_SERVOS + 1):
            try:
                angle = self.arm.Arm_serial_servo_read(servo_id)
                if angle is not None:
                    self.current_angles[servo_id] = angle
            except:
                pass  # Ignore read errors

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

    def display_overlay(self, frame):
        """Overlay servo angles and info on camera frame"""
        h, w = frame.shape[:2]

        # FPS Display - Top Center
        fps_color = (0, 255, 0) if self.fps > 10 else (0, 165, 255) if self.fps > 5 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w//2 - 50, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # LEFT COLUMN - Servo angles
        y_offset = 20
        for servo_id in range(1, NUM_SERVOS + 1):
            angle = self.current_angles.get(servo_id, "N/A")
            text = f"S{servo_id}:{angle:3}°"
            cv2.putText(frame, text, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 18

        # RIGHT COLUMN - Status
        right_x = w - 180
        y_offset = 20

        # Torque status (prominent warning)
        cv2.putText(frame, "TORQUE: OFF", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        y_offset += 22

        # Current slot
        slot_color = (0, 255, 255)
        cv2.putText(frame, f"SLOT:{self.current_slot}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, slot_color, 2)
        y_offset += 22

        # Detection status
        if self.model_loaded:
            det_color = (0, 255, 0) if self.detection_enabled else (100, 100, 100)
            det_text = "DET:ON" if self.detection_enabled else "DET:OFF"
            cv2.putText(frame, det_text, (right_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, det_color, 2)

        # TOP 3 PREDICTIONS - Center bottom
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
            "MOVE ARM BY HAND!",
            "F1-F6:Slot | SPACE:Save",
            "T:Detect | Q:Quit"
        ]

        y_offset = h - 55
        for instruction in instructions:
            cv2.putText(frame, instruction, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 16

        return frame

    def save_slot_position(self, slot_number):
        """Save current position for a specific slot"""
        print(f"\nReading current arm position for Slot {slot_number}...")

        # Read current angles one more time to be sure
        self.read_current_angles()

        # Convert current angles to the format expected
        angles = {f"servo_{i}": self.current_angles[i] for i in range(1, NUM_SERVOS + 1)}

        # Save with timestamp
        self.slot_positions[f"slot_{slot_number}"] = {
            "angles": angles,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n✓ Slot {slot_number} position saved:")
        for servo, angle in angles.items():
            print(f"  {servo}: {angle}°")

        return self.save_positions()

    def run(self):
        """Main loop with camera view"""
        print("\nStarting camera feed...")
        print("\n" + "="*60)
        print("MANUAL POSITIONING MODE - TORQUE OFF!")
        print("="*60)
        print("\nCONTROLS:")
        print("  MOVE ARM      - Physically move it with your hands!")
        print("  F1-F6         - Select slot 1-6")
        print("  SPACE         - Save current position to selected slot")
        print("  T             - Toggle object detection ON/OFF")
        print("  Q             - Quit")
        print("\nCurrent slot: {}".format(self.current_slot))

        if self.model_loaded:
            print("\n✓ Object detection available - press T to enable")
        else:
            print("\n✗ Object detection NOT available")

        print("="*60)
        print("\nCamera window is now open...")

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

                # Read current servo angles
                self.read_current_angles()

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
                frame = self.display_overlay(frame)
                frame = self.display_instructions(frame)

                # Show frame
                cv2.imshow("Manual Position Configurator - TORQUE OFF MODE", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                # Select slot (F1-F6)
                if key >= 190 and key <= 195:
                    self.current_slot = key - 189  # F1=1, F2=2, ..., F6=6
                    print(f"Selected Slot {self.current_slot}")

                # Save current position (SPACE)
                elif key == 32:  # SPACE bar
                    self.save_slot_position(self.current_slot)
                    print(f"✓ Saved current position to Slot {self.current_slot}")

                # Toggle detection (T)
                elif key == ord('t') or key == ord('T'):
                    if self.model:
                        self.detection_enabled = not self.detection_enabled
                        status = "ON" if self.detection_enabled else "OFF"
                        print(f"Object detection: {status}")
                    else:
                        print("Object detection model not available")

                # Quit (Q)
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

        # Re-enable torque before cleanup
        print("Re-enabling torque...")
        try:
            self.arm.Arm_serial_set_torque(1)  # 1 = torque ON
            time.sleep(0.1)
            print("✓ Torque re-enabled")
        except Exception as e:
            print(f"Warning: Could not re-enable torque: {e}")

        del self.arm
        print("Done!")


def main():
    configurator = ManualConfigurator()
    configurator.run()


if __name__ == "__main__":
    main()
