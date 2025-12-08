#!/usr/bin/env python3
#coding=utf-8
"""
Grab Position Configurator for DOFBOT - WITH GAME CONTROLLER SUPPORT (PYGAME)
==============================================================================
Configure precise grab positions for each slot using GAME CONTROLLER.

CONTROLLER MAPPING (PlayStation/Xbox style):
- Left Joystick: Servo 1 (horizontal) & Servo 2 (vertical) - Base & Shoulder
- Right Joystick: Servo 5 (vertical) & Servo 6 (horizontal) - Gripper Rotation & Gripper
- L1 (Button 4): Servo 3 decrease (Elbow)
- L2 (Button 6): Servo 3 increase (Elbow)
- R1 (Button 5): Servo 4 decrease (Wrist pitch)
- R2 (Button 7): Servo 4 increase (Wrist pitch)
- SELECT (Button 8): Reset all servos to 90°
- START (Button 9): Save grab position for current slot
- Triangle (Button 2): Increase step size
- Cross (Button 0): Decrease step size

KEYBOARD CONTROLS (still available):
- W/S/A/D/I/K/J/L/U/O/P/;: Move servos (U/O = GRIPPER ROTATION)
- F1-F6: Go to slot 1-6 position
- SPACE: Save grab position
- +/-: Step size
- [/]: Speed
- Q: Quit

WORKFLOW:
1. Press F1-F6 to automatically move to that slot
2. Use controller/keyboard to fine-tune grab position
3. Press START (or SPACE) to save grab position

INSTALLATION:
pip install pygame
"""

import time
import cv2
import json
import os
import numpy as np
from datetime import datetime
from Arm_Lib import Arm_Device

# Try to import pygame for controller support
try:
    import pygame
    import pygame.joystick
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    print("Warning: pygame not available. Controller support disabled.")
    print("Install with: pip install pygame")

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

# Movement configurations
ANGLE_STEP = 2
MOVE_SPEEDS = {
    'slow': 500,
    'normal': 300,
    'fast': 150,
    'instant': 50
}

# Controller deadzone
CONTROLLER_DEADZONE = 0.15

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

        # Initialize pygame and controller
        self.joystick = None
        self.controller_enabled = False

        if CONTROLLER_AVAILABLE:
            try:
                pygame.init()
                pygame.joystick.init()

                # Check for connected joysticks
                joystick_count = pygame.joystick.get_count()
                if joystick_count > 0:
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                    self.controller_enabled = True
                    print(f"✓ Controller connected: {self.joystick.get_name()}")
                    print(f"  Axes: {self.joystick.get_numaxes()}")
                    print(f"  Buttons: {self.joystick.get_numbuttons()}")
                else:
                    print("✗ No controller detected")
            except Exception as e:
                print(f"✗ Failed to initialize controller: {e}")
                self.controller_enabled = False

        # Button debounce tracking
        self.button_pressed = {}
        self.last_button_time = {}

        # Initialize camera
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Current servo angles
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
        print("GRAB POSITION CONFIGURATOR - CONTROLLER MODE")
        print("="*60)
        print(f"Controller: {'ENABLED' if self.controller_enabled else 'DISABLED (keyboard only)'}")
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
        print("Now use controller/keyboard to fine-tune the grab position!")
        print("IMPORTANT: Use right stick vertical / U/O for SERVO 5 (gripper rotation)")

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

        # Controller status
        if self.controller_enabled:
            cv2.putText(frame, "CTRL:ON", (right_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "CTRL:OFF", (right_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        y_offset += 20

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

        instructions = []
        if self.controller_enabled:
            instructions.extend([
                "L-Stick:S1/S2 R-Stick:S5(ROT)/S6",
                "L1/L2:S3 R1/R2:S4 START:SaveGrab"
            ])
        else:
            instructions.extend([
                "F1-F6:GoToSlot | U/O:S5(GRIP-ROT)",
                "SPACE:SaveGrab | Q:Quit"
            ])

        y_offset = h - 40
        for instruction in instructions:
            cv2.putText(frame, instruction, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 16

        return frame

    def save_grab_position(self, slot_number):
        """Save current grab position for a specific slot"""
        print(f"\nSaving grab position for Slot {slot_number}...")

        angles = {f"servo_{i}": self.current_angles[i] for i in range(1, NUM_SERVOS + 1)}

        slot_key = f"slot_{slot_number}"
        self.grab_positions[slot_key] = {
            "angles": angles,
            "gripper_rotation": self.current_angles.get(5, 90),
            "gripper_close": self.current_angles.get(6, 90),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n✓ Grab position saved for Slot {slot_number}:")
        for servo, angle in angles.items():
            print(f"  {servo}: {angle}°")
        print(f"  *** Gripper rotation (S5): {self.current_angles.get(5, 90)}° ***")
        print(f"  *** Gripper close (S6): {self.current_angles.get(6, 90)}° ***")

        return self.save_grab_positions()

    def is_button_pressed(self, button_id, debounce_time=0.3):
        """Check if button is pressed with debouncing"""
        current_time = time.time()

        if button_id not in self.last_button_time:
            self.last_button_time[button_id] = 0

        if current_time - self.last_button_time[button_id] > debounce_time:
            if self.joystick.get_button(button_id):
                self.last_button_time[button_id] = current_time
                return True

        return False

    def handle_controller_input(self):
        """Handle game controller input using pygame"""
        if not self.controller_enabled or self.joystick is None:
            return

        # Process pygame events (required for joystick updates)
        pygame.event.pump()

        s_step = self.angle_step

        # Left joystick - Servo 1 (horizontal) and Servo 2 (vertical)
        try:
            axis0_val = self.joystick.get_axis(0)  # Left stick horizontal
            axis1_val = self.joystick.get_axis(1)  # Left stick vertical

            # Servo 1 - Base rotation
            if abs(axis0_val) > CONTROLLER_DEADZONE:
                if axis0_val > CONTROLLER_DEADZONE:
                    self.adjust_servo(1, -s_step)
                else:
                    self.adjust_servo(1, s_step)

            # Servo 2 - Shoulder
            if abs(axis1_val) > CONTROLLER_DEADZONE:
                if axis1_val > CONTROLLER_DEADZONE:
                    self.adjust_servo(2, s_step)
                else:
                    self.adjust_servo(2, -s_step)

            # Right joystick - Servo 5 (GRIPPER ROTATION) and Servo 6
            if self.joystick.get_numaxes() >= 4:
                axis2_val = self.joystick.get_axis(2)
                axis3_val = self.joystick.get_axis(3)

                # Try axis 4 and 5 for some controllers
                if self.joystick.get_numaxes() >= 6:
                    axis4_val = self.joystick.get_axis(4)
                    axis5_val = self.joystick.get_axis(5)

                    # Servo 6 - Gripper
                    if abs(axis4_val) > CONTROLLER_DEADZONE:
                        if axis4_val > CONTROLLER_DEADZONE:
                            self.adjust_servo(6, -s_step)
                        else:
                            self.adjust_servo(6, s_step)

                    # Servo 5 - GRIPPER ROTATION (IMPORTANT!)
                    if abs(axis5_val) > CONTROLLER_DEADZONE:
                        if axis5_val > CONTROLLER_DEADZONE:
                            self.adjust_servo(5, s_step)
                        else:
                            self.adjust_servo(5, -s_step)
                else:
                    # Fallback to axis 2/3
                    if abs(axis2_val) > CONTROLLER_DEADZONE:
                        if axis2_val > CONTROLLER_DEADZONE:
                            self.adjust_servo(6, -s_step)
                        else:
                            self.adjust_servo(6, s_step)

                    # Servo 5 - GRIPPER ROTATION
                    if abs(axis3_val) > CONTROLLER_DEADZONE:
                        if axis3_val > CONTROLLER_DEADZONE:
                            self.adjust_servo(5, s_step)
                        else:
                            self.adjust_servo(5, -s_step)

            # Shoulder buttons for Servo 3 and 4
            num_buttons = self.joystick.get_numbuttons()

            # Servo 3 - Elbow (L1=4, L2=6)
            if num_buttons > 4 and self.joystick.get_button(4):
                self.adjust_servo(3, -s_step)
            if num_buttons > 6 and self.joystick.get_button(6):
                self.adjust_servo(3, s_step)

            # Servo 4 - Wrist pitch (R1=5, R2=7)
            if num_buttons > 5 and self.joystick.get_button(5):
                self.adjust_servo(4, -s_step)
            if num_buttons > 7 and self.joystick.get_button(7):
                self.adjust_servo(4, s_step)

            # SELECT button (8) - Reset to 90 degrees
            if num_buttons > 8 and self.is_button_pressed(8):
                print("\nResetting all servos to 90°...")
                for i in range(1, NUM_SERVOS + 1):
                    self.current_angles[i] = 90
                self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
                time.sleep(1)

            # START button (9) - Save grab position
            if num_buttons > 9 and self.is_button_pressed(9):
                self.save_grab_position(self.current_slot)

            # Buttons for adjustments
            if num_buttons > 0 and self.is_button_pressed(0):  # Cross/A - decrease step
                self.adjust_step_size(-1)
            if num_buttons > 2 and self.is_button_pressed(2):  # Triangle/Y - increase step
                self.adjust_step_size(1)

        except Exception as e:
            print(f"Controller input error: {e}")

    def run(self):
        """Main loop with camera view and keyboard/controller control"""
        print("\nStarting camera feed...")
        print("\n" + "="*60)
        if self.controller_enabled:
            print("CONTROLLER MODE ENABLED!")
            print("="*60)
            print("\nCONTROLLER CONTROLS:")
            print("  Left Stick      - Servo 1 (Base) & Servo 2 (Shoulder)")
            print("  Right Stick     - Servo 5 (GRIP-ROT) & Servo 6 (Gripper)")
            print("  L1/L2           - Servo 3 (Elbow)")
            print("  R1/R2           - Servo 4 (Wrist pitch)")
            print("  SELECT (B8)     - Reset all servos to 90°")
            print("  START (B9)      - Save grab position")
            print("  Cross (B0)      - Decrease step size")
            print("  Triangle (B2)   - Increase step size")
        else:
            print("KEYBOARD MODE (Controller not available)")
            print("="*60)

        print("\nKEYBOARD CONTROLS:")
        print("  F1-F6           - Go to slot position")
        print("  W/S/A/D/I/K/J/L - Move servos")
        print("  U/O             - Servo 5 (GRIPPER ROTATION) ← IMPORTANT!")
        print("  P/;             - Servo 6 (Gripper open/close)")
        print("  SPACE           - Save grab position")
        print("  +/-             - Step size")
        print("  [ / ]           - Speed")
        print("  T               - Toggle detection")
        print("  Q               - Quit")
        print("\nCurrent slot: {}".format(self.current_slot))
        print("="*60)

        try:
            while True:
                # Handle controller input
                if self.controller_enabled:
                    self.handle_controller_input()

                # Read camera frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
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

                cv2.imshow("Grab Position Configurator - Controller Mode", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                # Slot selection (F1-F6)
                if key >= 190 and key <= 195:
                    self.current_slot = key - 189
                    print(f"\n{'='*60}")
                    print(f"Selected Slot {self.current_slot}")
                    self.move_to_slot(self.current_slot)

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

                # Servo 5 - GRIPPER ROTATION (U/O) ← IMPORTANT
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

                # Settings
                elif key == ord('+') or key == ord('='):
                    self.adjust_step_size(1)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_step_size(-1)
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

                # Save grab position (SPACE)
                elif key == 32:
                    self.save_grab_position(self.current_slot)

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

        if self.controller_enabled and self.joystick:
            self.joystick.quit()

        if CONTROLLER_AVAILABLE:
            pygame.quit()

        del self.arm
        print("Done!")


def main():
    """Main entry point"""
    configurator = GrabConfigurator()
    configurator.run()


if __name__ == "__main__":
    main()
