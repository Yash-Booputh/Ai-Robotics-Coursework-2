#!/usr/bin/env python3
#coding=utf-8
"""
Slot Position Configurator for DOFBOT
This tool allows keyboard-controlled positioning of the robot arm and saving servo angles for each slot.
Use this to configure slot positions without hard-coding angles.
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

# Movement configurations
ANGLE_STEP = 2        # Angle step per key press
MOVE_SPEEDS = {
    'slow': 500,
    'normal': 300,
    'fast': 150,
    'instant': 50
}

# Object detection model paths
MODEL_PATH = "best.onnx"  # ONNX model path
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
DETECTION_FRAME_SKIP = 2  # Run detection every N frames (1=every frame, 3=every 3rd frame)

# Class names - ChefMate Dataset (must match data.yaml order)
CLASS_NAMES = [
    'anchovies',
    'basil',
    'cheese',
    'chicken',
    'fresh_tomato',
    'shrimp'
]

class SlotConfigurator:
    def __init__(self):
        # Initialize DOFBOT
        print("Initializing DOFBOT...")
        self.arm = Arm_Device()
        time.sleep(0.1)

        # Initialize camera with environment variable for display
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 instead of Wayland

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Camera settings
        self.brightness = 0  # -100 to 100
        self.contrast = 0    # -100 to 100
        self.saturation = 0  # -100 to 100
        self.detection_enabled = False  # Toggle for object detection

        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Frame skipping for detection
        self.detection_frame_counter = 0
        self.last_detections = []  # Cache last detection results

        # Current servo angles (initialize by reading current positions)
        self.current_angles = {}
        self.selected_servo = 1  # Currently selected servo for adjustment

        # Movement speed setting
        self.speed_mode = 'normal'
        self.current_speed = MOVE_SPEEDS['normal']

        # Angle step adjustment
        self.angle_step = ANGLE_STEP

        # Initialize object detection model
        self.model = None
        self.model_loaded = False
        self.top_detections = []  # Store top 3 detections

        print("\n" + "="*60)
        print("CHECKING MODEL FILES")
        print("="*60)

        # Check for best.pt
        if os.path.exists("best.pt"):
            print("✓ best.pt found")
        else:
            print("✗ best.pt NOT found")

        # Check for best.onnx
        if os.path.exists(MODEL_PATH):
            print(f"✓ {MODEL_PATH} found")
        else:
            print(f"✗ {MODEL_PATH} NOT found")

        if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                print(f"\nLoading ONNX model: {MODEL_PATH}...")
                self.model = ort.InferenceSession(MODEL_PATH)
                self.input_name = self.model.get_inputs()[0].name
                self.input_shape = self.model.get_inputs()[0].shape
                self.model_loaded = True
                print(f"✓✓✓ MODEL LOADED SUCCESSFULLY! ✓✓✓")
                print(f"Model input name: {self.input_name}")
                print(f"Model input shape: {self.input_shape}")
            except Exception as e:
                print(f"✗✗✗ FAILED TO LOAD MODEL ✗✗✗")
                print(f"Error: {e}")
                self.model = None
                self.model_loaded = False
        else:
            if not ONNX_AVAILABLE:
                print("\n✗ ONNX Runtime not available - install with:")
                print("  pip install onnxruntime --break-system-packages")
            elif not os.path.exists(MODEL_PATH):
                print(f"\n✗ Model file not found: {MODEL_PATH}")
                print("  Please place best.onnx in the project directory")

        print("="*60)

        # Initialize current angles
        print("Reading initial servo positions...")
        for i in range(NUM_SERVOS):
            servo_id = i + 1
            try:
                angle = self.arm.Arm_serial_servo_read(servo_id)
                self.current_angles[servo_id] = angle if angle is not None else 90
            except:
                self.current_angles[servo_id] = 90  # Default to 90 if read fails
            time.sleep(0.01)

        # Load existing slot positions if available
        self.slot_positions = self.load_positions()

        print("\n" + "="*60)
        print("DOFBOT Slot Position Configurator")
        print("="*60)
        print(f"Selected Servo: {self.selected_servo}")

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

    def move_servo(self, servo_id, angle):
        """Move a specific servo to an angle"""
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
        current = self.current_angles.get(servo_id, 90)
        new_angle = current + delta
        self.move_servo(servo_id, new_angle)

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

    def adjust_brightness(self, delta):
        """Adjust camera brightness"""
        self.brightness = max(-100, min(100, self.brightness + delta))
        print(f"Brightness: {self.brightness}")

    def adjust_contrast(self, delta):
        """Adjust camera contrast"""
        self.contrast = max(-100, min(100, self.contrast + delta))
        print(f"Contrast: {self.contrast}")

    def adjust_saturation(self, delta):
        """Adjust camera saturation"""
        self.saturation = max(-100, min(100, self.saturation + delta))
        print(f"Saturation: {self.saturation}")

    def apply_camera_adjustments(self, frame):
        """Apply brightness, contrast, and saturation adjustments"""
        # Convert to float for processing
        adjusted = frame.astype(np.float32)

        # Apply brightness (simple addition)
        if self.brightness != 0:
            adjusted = adjusted + (self.brightness * 1.5)

        # Apply contrast (alpha adjustment)
        if self.contrast != 0:
            alpha = 1.0 + (self.contrast / 100.0)
            adjusted = adjusted * alpha

        # Apply saturation
        if self.saturation != 0:
            hsv = cv2.cvtColor(np.clip(adjusted, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + self.saturation / 100.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted

    def preprocess_frame(self, frame):
        """Preprocess frame for YOLO model"""
        # Get input size from model
        input_height = self.input_shape[2] if len(self.input_shape) > 2 else 640
        input_width = self.input_shape[3] if len(self.input_shape) > 3 else 640

        # Resize frame
        resized = cv2.resize(frame, (input_width, input_height))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and transpose to CHW format
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        input_tensor = np.expand_dims(transposed, axis=0)

        return input_tensor

    def postprocess_detections(self, outputs, frame_shape, conf_threshold=CONFIDENCE_THRESHOLD):
        """Post-process YOLO outputs to get bounding boxes - YOLOv8 format"""
        detections = []

        # Handle different YOLO output formats
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs

        # YOLOv8 ONNX output is typically (1, 84+num_classes, 8400) or (1, num_predictions, 84+num_classes)
        # We need to transpose to (num_predictions, 84+num_classes)

        if len(output.shape) == 3:
            # If shape is (1, features, predictions), transpose to (predictions, features)
            if output.shape[1] < output.shape[2]:
                output = output[0].T  # Transpose to (predictions, features)
            else:
                output = output[0]

        # YOLOv8 format: [x_center, y_center, width, height, class1_prob, class2_prob, ...]
        # No objectness score in YOLOv8
        for detection in output:
            if len(detection) >= 4:
                x_center, y_center, width, height = detection[:4]

                # Get class probabilities (everything after bbox coords)
                class_probs = detection[4:]
                class_id = np.argmax(class_probs)
                confidence = float(class_probs[class_id])

                if confidence > conf_threshold:
                    # Convert to original frame coordinates
                    h, w = frame_shape[:2]

                    # YOLOv8 uses normalized coordinates (0-1) or pixel coordinates
                    # Check if values are normalized
                    if x_center <= 1.0 and y_center <= 1.0:
                        # Normalized coordinates
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                    else:
                        # Pixel coordinates - scale from model input size to frame size
                        input_h = self.input_shape[2] if len(self.input_shape) > 2 else 640
                        input_w = self.input_shape[3] if len(self.input_shape) > 3 else 640

                        scale_x = w / input_w
                        scale_y = h / input_h

                        x1 = int((x_center - width / 2) * scale_x)
                        y1 = int((y_center - height / 2) * scale_y)
                        x2 = int((x_center + width / 2) * scale_x)
                        y2 = int((y_center + height / 2) * scale_y)

                    # Clamp to frame boundaries
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

        # Sort by confidence and get top detections
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        # Store top 3 for display
        self.top_detections = detections[:3]

        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']

            # Random color based on class_id
            color = tuple([int(c) for c in np.random.RandomState(det['class_id']).randint(0, 255, 3)])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def display_current_angles(self, frame):
        """Overlay current servo angles and settings on camera frame - organized layout"""
        h, w = frame.shape[:2]

        # FPS Display - Top Center
        fps_color = (0, 255, 0) if self.fps > 10 else (0, 165, 255) if self.fps > 5 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w//2 - 50, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        # LEFT COLUMN - Servo angles (compact)
        y_offset = 20
        for servo_id in range(1, NUM_SERVOS + 1):
            angle = self.current_angles.get(servo_id, "N/A")
            color = (0, 255, 255) if servo_id == self.selected_servo else (0, 255, 0)
            thickness = 2 if servo_id == self.selected_servo else 1
            prefix = ">" if servo_id == self.selected_servo else " "
            text = f"{prefix}S{servo_id}:{angle:3}°"
            cv2.putText(frame, text, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
            y_offset += 18

        # RIGHT COLUMN - Settings
        right_x = w - 160
        y_offset = 20

        # Movement settings
        settings_color = (255, 165, 0)  # Orange
        cv2.putText(frame, f"Spd:{self.speed_mode[:3].upper()}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, settings_color, 1)
        y_offset += 18
        cv2.putText(frame, f"Step:{self.angle_step}°", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, settings_color, 1)
        y_offset += 18

        # Camera settings
        camera_color = (147, 20, 255)  # Pink/Purple
        cv2.putText(frame, f"Brt:{self.brightness:+3}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, camera_color, 1)
        y_offset += 16
        cv2.putText(frame, f"Con:{self.contrast:+3}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, camera_color, 1)
        y_offset += 16
        cv2.putText(frame, f"Sat:{self.saturation:+3}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, camera_color, 1)
        y_offset += 20

        # Detection status
        if self.model_loaded:
            det_color = (0, 255, 0) if self.detection_enabled else (100, 100, 100)
            det_text = "DET:ON" if self.detection_enabled else "DET:OFF"
            cv2.putText(frame, det_text, (right_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, det_color, 2)

        # TOP 3 PREDICTIONS - Center bottom area
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
        """Display compact controls on the camera frame - bottom left"""
        h, w = frame.shape[:2]

        # Simplified controls at bottom left
        instructions = [
            "WASD/IKJLUO/Arrows:Move",
            "1-6:Camera | T:Detect",
            "0,7-9:Save | R:Load | Q:Quit"
        ]

        y_offset = h - 55
        for instruction in instructions:
            cv2.putText(frame, instruction, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 16

        return frame

    def save_slot_position(self, slot_number):
        """Save current position for a specific slot"""
        print(f"\nSaving current position to Slot {slot_number}...")

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

    def recall_slot_position(self):
        """Recall and move to a saved slot position"""
        if not self.slot_positions:
            print("\n✗ No saved slots available.")
            return

        print("\n" + "="*60)
        print("Available slots:")
        for slot_name in sorted(self.slot_positions.keys()):
            slot_num = slot_name.split('_')[1]
            print(f"  Slot {slot_num}")

        try:
            slot_num = input("\nEnter slot number to recall (or 'c' to cancel): ").strip()
            if slot_num.lower() == 'c':
                return

            slot_key = f"slot_{slot_num}"
            if slot_key in self.slot_positions:
                angles = self.slot_positions[slot_key]['angles']
                print(f"\nMoving to Slot {slot_num} position...")

                for servo_key, angle in angles.items():
                    servo_id = int(servo_key.split('_')[1])
                    self.move_servo(servo_id, angle)
                    time.sleep(0.1)

                print("✓ Position recalled successfully!")
            else:
                print(f"✗ Slot {slot_num} not found.")
        except Exception as e:
            print(f"Error: {e}")

    def list_saved_slots(self):
        """Display all saved slot positions"""
        print("\n" + "="*60)
        print("SAVED SLOT POSITIONS")
        print("="*60)

        if not self.slot_positions:
            print("No slots configured yet.")
            return

        for slot_name, data in sorted(self.slot_positions.items()):
            slot_num = slot_name.split('_')[1]
            timestamp = data.get('timestamp', 'Unknown')
            print(f"\nSlot {slot_num} (saved: {timestamp}):")
            for servo, angle in data['angles'].items():
                servo_num = servo.split('_')[1]
                print(f"  Servo {servo_num}: {angle}°")

    def delete_slot(self):
        """Delete a saved slot position"""
        self.list_saved_slots()
        if not self.slot_positions:
            return

        try:
            slot_num = input("\nEnter slot number to delete (or 'c' to cancel): ").strip()
            if slot_num.lower() == 'c':
                return

            slot_key = f"slot_{slot_num}"
            if slot_key in self.slot_positions:
                del self.slot_positions[slot_key]
                self.save_positions()
                print(f"✓ Slot {slot_num} deleted.")
            else:
                print(f"✗ Slot {slot_num} not found.")
        except Exception as e:
            print(f"Error: {e}")

    def run(self):
        """Main loop with camera view and keyboard control"""
        print("\nStarting camera feed...")
        print("\n" + "="*60)
        print("VIDEO GAME STYLE CONTROLS!")
        print("="*60)
        print("\nMOVEMENT (each key controls one servo):")
        print("  W/S         - Servo 1 (Base rotation)")
        print("  A/D         - Servo 2 (Shoulder)")
        print("  I/K         - Servo 3 (Elbow)")
        print("  J/L         - Servo 4 (Wrist pitch)")
        print("  U/O         - Servo 5 (Wrist roll)")
        print("  UP/DOWN     - Servo 6 (Gripper open/close)")
        print("\nCAMERA CONTROLS:")
        print("  1/2         - Decrease/Increase brightness")
        print("  3/4         - Decrease/Increase contrast")
        print("  5/6         - Decrease/Increase saturation")
        print("  T           - Toggle object detection ON/OFF")
        print("\nSETTINGS:")
        print("  +/-         - Increase/decrease step size (1-10°)")
        print("  [ / ]       - Slower / Faster speed")
        print("\nSLOT MANAGEMENT:")
        print("  0-9         - Save current position to slot")
        print("  R           - Recall/load a saved slot")
        print("  Q           - Quit")
        print("\nCurrent settings: Speed={}, Step={}°".format(self.speed_mode, self.angle_step))

        # Show model status
        print("\n" + "="*60)
        if self.model_loaded:
            print("✓✓✓ OBJECT DETECTION MODEL READY ✓✓✓")
            print("Press 'T' to toggle detection ON/OFF")
        else:
            print("✗ Object detection NOT available")
            print("Make sure best.onnx is in the project directory")
        print("="*60)

        print("\nCamera window is now open...")

        try:
            while True:
                # Read camera frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read camera frame")
                    break

                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.fps_start_time
                if elapsed > 1.0:  # Update FPS every second
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = current_time

                # Apply camera adjustments (only if non-zero to save processing)
                if self.brightness != 0 or self.contrast != 0 or self.saturation != 0:
                    frame = self.apply_camera_adjustments(frame)

                # Run object detection with frame skipping
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
                    # Reset when detection is off
                    self.last_detections = []
                    self.top_detections = []

                # Display current servo angles on frame
                frame = self.display_current_angles(frame)

                # Display instructions
                frame = self.display_instructions(frame)

                # Show frame
                cv2.imshow("DOFBOT Slot Configurator - Video Game Controls", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                # === SERVO MOVEMENT (Video Game Style) ===
                # Servo 1 - Base rotation (W/S)
                if key == ord('w') or key == ord('W'):
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

                # Servo 6 - Gripper (UP/DOWN arrows)
                elif key == 82:  # Up arrow
                    self.adjust_servo(6, self.angle_step)
                elif key == 84:  # Down arrow
                    self.adjust_servo(6, -self.angle_step)

                # === CAMERA CONTROLS ===
                # Brightness controls (1/2)
                elif key == ord('1'):
                    self.adjust_brightness(-5)
                elif key == ord('2'):
                    self.adjust_brightness(5)

                # Contrast controls (3/4)
                elif key == ord('3'):
                    self.adjust_contrast(-5)
                elif key == ord('4'):
                    self.adjust_contrast(5)

                # Saturation controls (5/6)
                elif key == ord('5'):
                    self.adjust_saturation(-5)
                elif key == ord('6'):
                    self.adjust_saturation(5)

                # Toggle object detection (T)
                elif key == ord('t') or key == ord('T'):
                    if self.model:
                        self.detection_enabled = not self.detection_enabled
                        status = "ON" if self.detection_enabled else "OFF"
                        print(f"Object detection: {status}")
                    else:
                        print("Object detection model not available")

                # === SETTINGS ADJUSTMENT ===
                # Increase step size (+/=)
                elif key == ord('+') or key == ord('='):
                    self.adjust_step_size(1)

                # Decrease step size (-)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_step_size(-1)

                # Speed slower ([)
                elif key == ord('[') or key == ord('{'):
                    self.change_speed('down')

                # Speed faster (])
                elif key == ord(']') or key == ord('}'):
                    self.change_speed('up')

                # === SLOT MANAGEMENT ===
                # Save slot positions (keys 0-9) - but not camera controls
                elif key >= ord('7') and key <= ord('9'):
                    slot_number = key - ord('0')
                    self.save_slot_position(slot_number)

                # Slot 0
                elif key == ord('0'):
                    self.save_slot_position(0)

                # Recall slot position (R)
                elif key == ord('r') or key == ord('R'):
                    cv2.destroyAllWindows()
                    self.recall_slot_position()
                    print("\nResuming... Camera window reopening...")
                    time.sleep(0.5)

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
        del self.arm
        print("Done!")

def main():
    configurator = SlotConfigurator()
    configurator.run()

if __name__ == "__main__":
    main()
