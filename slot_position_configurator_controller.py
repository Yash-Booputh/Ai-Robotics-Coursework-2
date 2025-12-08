#!/usr/bin/env python3
#coding=utf-8
"""
Slot Position Configurator for DOFBOT - WITH GAME CONTROLLER SUPPORT (PYGAME)
===============================================================================
This tool allows GAME CONTROLLER positioning of the robot arm and saving servo angles for each slot.

CONTROLLER MAPPING (PlayStation/Xbox style):
- Left Joystick: Servo 1 (horizontal) & Servo 2 (vertical) - Base & Shoulder
- Right Joystick: Servo 5 (vertical) & Servo 6 (horizontal) - Wrist & Gripper
- L1 (Button 4): Servo 3 decrease (Elbow)
- L2 (Button 6): Servo 3 increase (Elbow)
- R1 (Button 5): Servo 4 decrease (Wrist pitch)
- R2 (Button 7): Servo 4 increase (Wrist pitch)
- SELECT (Button 8): Reset all servos to 90°
- START (Button 9): Save current position to selected slot
- D-Pad Left/Right: Change slot selection
- Triangle (Button 2): Increase step size
- Cross (Button 0): Decrease step size

KEYBOARD CONTROLS (still available):
- W/S/A/D/I/K/J/L/U/O/P/;: Move servos
- F1-F6: Select slot 1-6
- SPACE: Save position
- +/-: Step size
- [/]: Speed
- T: Toggle detection
- Q: Quit

INSTALLATION:
pip install pygame
"""

import time
import cv2
import json
import os
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from queue import Queue
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
SLOTS_FILE = "slot_positions.json"
NUM_SERVOS = 6

# Movement configurations
ANGLE_STEP = 2
MOVE_SPEEDS = {
    'slow': 500,
    'normal': 300,
    'fast': 150,
    'instant': 50
}

# Controller deadzone (to prevent drift)
CONTROLLER_DEADZONE = 0.15
CONTROLLER_POLL_RATE = 0.01  # Poll controller every 10ms

# Object detection model paths
MODEL_PATH = "models/best.onnx"
CONFIDENCE_THRESHOLD = 0.5
DETECTION_FRAME_SKIP = 2

# Performance settings
USE_THREADING = True
CAMERA_BUFFER_SIZE = 1

# Class names
CLASS_NAMES = [
    'anchovies',
    'basil',
    'cheese',
    'chicken',
    'fresh_tomato',
    'shrimp'
]

class ThreadedCamera:
    """Threaded camera capture for improved performance"""
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.lock = Lock()
        self.stopped = False

    def start(self):
        """Start the camera capture thread"""
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        """Continuously read frames from camera in background"""
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.001)

    def read(self):
        """Get the latest frame"""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return False, None

    def stop(self):
        """Stop the camera thread"""
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()

class DetectionThread:
    """Threaded object detection for improved performance"""
    def __init__(self, model, input_name, input_shape):
        self.model = model
        self.input_name = input_name
        self.input_shape = input_shape

        self.frame_queue = Queue(maxsize=CAMERA_BUFFER_SIZE)
        self.result_lock = Lock()
        self.detections = []
        self.top_detections = []
        self.stopped = False
        self.processing = False

    def start(self):
        """Start the detection processing thread"""
        Thread(target=self.process, daemon=True).start()
        return self

    def process(self):
        """Process detection frames in background"""
        while not self.stopped:
            if not self.frame_queue.empty():
                try:
                    input_tensor, frame_shape = self.frame_queue.get()
                    self.processing = True

                    outputs = self.model.run(None, {self.input_name: input_tensor})
                    detections = self._postprocess(outputs, frame_shape)

                    with self.result_lock:
                        self.detections = detections
                        self.top_detections = detections[:3]

                    self.processing = False
                except Exception as e:
                    print(f"Detection error: {e}")
                    self.processing = False
            else:
                time.sleep(0.01)

    def _postprocess(self, outputs, frame_shape):
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

    def submit_frame(self, input_tensor, frame_shape):
        """Submit a frame for detection processing"""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass

        try:
            self.frame_queue.put_nowait((input_tensor, frame_shape))
        except:
            pass

    def get_detections(self):
        """Get latest detection results"""
        with self.result_lock:
            return self.detections.copy(), self.top_detections.copy()

    def stop(self):
        """Stop the detection thread"""
        self.stopped = True

class SlotConfigurator:
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

        # Initialize camera with environment variable for display
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

        if USE_THREADING:
            self.cap = ThreadedCamera(src=0, width=640, height=480).start()
            print("✓ Using threaded camera capture")
            time.sleep(0.5)
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Camera settings
        self.brightness = 0
        self.contrast = 0
        self.saturation = 0
        self.detection_enabled = False

        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Frame skipping for detection
        self.detection_frame_counter = 0
        self.last_detections = []

        # Current servo angles
        self.current_angles = {}
        self.selected_servo = 1

        # Movement speed setting
        self.speed_mode = 'normal'
        self.current_speed = MOVE_SPEEDS['normal']

        # Angle step adjustment
        self.angle_step = ANGLE_STEP

        # Current slot being programmed
        self.current_slot = 1

        # Initialize object detection model
        self.model = None
        self.model_loaded = False
        self.top_detections = []
        self.detection_thread = None

        print("\n" + "="*60)
        print("CHECKING MODELS FILES")
        print("="*60)

        if os.path.exists("models/best.pt"):
            print("✓ models/best.pt found")
        else:
            print("✗ models/best.pt NOT found")

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

                if USE_THREADING:
                    self.detection_thread = DetectionThread(self.model, self.input_name, self.input_shape).start()
                    print("✓ Using threaded object detection")
            except Exception as e:
                print(f"✗✗✗ FAILED TO LOAD MODEL ✗✗✗")
                print(f"Error: {e}")
                self.model = None
                self.model_loaded = False
        else:
            if not ONNX_AVAILABLE:
                print("\n✗ ONNX Runtime not available")
            elif not os.path.exists(MODEL_PATH):
                print(f"\n✗ Model file not found: {MODEL_PATH}")

        print("="*60)

        # Initialize current angles
        print("Reading initial servo positions...")
        for i in range(NUM_SERVOS):
            servo_id = i + 1
            try:
                angle = self.arm.Arm_serial_servo_read(servo_id)
                self.current_angles[servo_id] = angle if angle is not None else 90
            except:
                self.current_angles[servo_id] = 90
            time.sleep(0.01)

        # Load existing slot positions if available
        self.slot_positions = self.load_positions()

        print("\n" + "="*60)
        print("DOFBOT Slot Position Configurator - CONTROLLER MODE")
        print("="*60)
        print(f"Controller: {'ENABLED' if self.controller_enabled else 'DISABLED (keyboard only)'}")
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
        adjusted = frame.astype(np.float32)

        if self.brightness != 0:
            adjusted = adjusted + (self.brightness * 1.5)

        if self.contrast != 0:
            alpha = 1.0 + (self.contrast / 100.0)
            adjusted = adjusted * alpha

        if self.saturation != 0:
            hsv = cv2.cvtColor(np.clip(adjusted, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + self.saturation / 100.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted

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

    def postprocess_detections(self, outputs, frame_shape, conf_threshold=CONFIDENCE_THRESHOLD):
        """Post-process YOLO outputs to get bounding boxes"""
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

                if confidence > conf_threshold:
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

        # Movement settings
        settings_color = (255, 165, 0)
        cv2.putText(frame, f"Spd:{self.speed_mode[:3].upper()}", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, settings_color, 1)
        y_offset += 18
        cv2.putText(frame, f"Step:{self.angle_step}°", (right_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, settings_color, 1)
        y_offset += 18

        # Camera settings
        camera_color = (147, 20, 255)
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
                "L-Stick:S1/S2 R-Stick:S5/S6",
                "L1/L2:S3 R1/R2:S4 START:Save"
            ])
        else:
            instructions.extend([
                "WASD/IKJLUO/P;:Move",
                "F1-F6:Slot | SPACE:Save | Q:Quit"
            ])

        y_offset = h - 40
        for instruction in instructions:
            cv2.putText(frame, instruction, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 16

        return frame

    def save_slot_position(self, slot_number):
        """Save current position for a specific slot"""
        print(f"\nSaving current position to Slot {slot_number}...")

        angles = {f"servo_{i}": self.current_angles[i] for i in range(1, NUM_SERVOS + 1)}

        self.slot_positions[f"slot_{slot_number}"] = {
            "angles": angles,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"\n✓ Slot {slot_number} position saved:")
        for servo, angle in angles.items():
            print(f"  {servo}: {angle}°")

        return self.save_positions()

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

            # Right joystick - Servo 5 and Servo 6
            if self.joystick.get_numaxes() >= 4:
                axis2_val = self.joystick.get_axis(2)  # Right stick horizontal (sometimes axis 3)
                axis3_val = self.joystick.get_axis(3)  # Right stick vertical

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

                    # Servo 5 - Wrist roll
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

            # START button (9) - Save position
            if num_buttons > 9 and self.is_button_pressed(9):
                self.save_slot_position(self.current_slot)

            # D-Pad or buttons for slot selection
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
            print("  Right Stick     - Servo 5 (Wrist) & Servo 6 (Gripper)")
            print("  L1/L2           - Servo 3 (Elbow)")
            print("  R1/R2           - Servo 4 (Wrist pitch)")
            print("  SELECT (B8)     - Reset all servos to 90°")
            print("  START (B9)      - Save current position")
            print("  Cross (B0)      - Decrease step size")
            print("  Triangle (B2)   - Increase step size")
        else:
            print("KEYBOARD MODE (Controller not available)")
            print("="*60)

        print("\nKEYBOARD CONTROLS:")
        print("  W/S/A/D/I/K/J/L/U/O/P/; - Move servos")
        print("  F1-F6           - Select slot 1-6")
        print("  SPACE           - Save position")
        print("  +/-             - Step size")
        print("  [ / ]           - Speed")
        print("  T               - Toggle detection")
        print("  Q               - Quit")
        print("="*60)

        try:
            if USE_THREADING:
                retry_count = 0
                while retry_count < 10:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.1)
                    retry_count += 1
                if not ret or frame is None:
                    print("Failed to initialize camera")
                    return

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

                # Apply camera adjustments
                adjusted_frame = frame
                if self.brightness != 0 or self.contrast != 0 or self.saturation != 0:
                    adjusted_frame = self.apply_camera_adjustments(frame)

                # Run object detection
                if self.detection_enabled and self.model:
                    if USE_THREADING and self.detection_thread:
                        self.detection_frame_counter += 1

                        if self.detection_frame_counter >= DETECTION_FRAME_SKIP:
                            self.detection_frame_counter = 0
                            input_tensor = self.preprocess_frame(adjusted_frame)
                            self.detection_thread.submit_frame(input_tensor, adjusted_frame.shape)

                        detections, top_detections = self.detection_thread.get_detections()
                        if detections:
                            self.last_detections = detections
                            self.top_detections = top_detections
                            adjusted_frame = self.draw_detections(adjusted_frame, self.last_detections)
                    else:
                        self.detection_frame_counter += 1

                        if self.detection_frame_counter >= DETECTION_FRAME_SKIP:
                            self.detection_frame_counter = 0
                            try:
                                input_tensor = self.preprocess_frame(adjusted_frame)
                                outputs = self.model.run(None, {self.input_name: input_tensor})
                                self.last_detections = self.postprocess_detections(outputs, adjusted_frame.shape)
                            except Exception as e:
                                print(f"Detection error: {e}")

                        if self.last_detections:
                            adjusted_frame = self.draw_detections(adjusted_frame, self.last_detections)
                else:
                    self.last_detections = []
                    self.top_detections = []

                # Display overlays
                adjusted_frame = self.display_current_angles(adjusted_frame)
                adjusted_frame = self.display_instructions(adjusted_frame)

                cv2.imshow("DOFBOT Slot Configurator - Controller Mode", adjusted_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

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

                # Servo 6 - Gripper (P/;)
                elif key == ord('p') or key == ord('P'):
                    self.adjust_servo(6, self.angle_step)
                elif key == ord(';') or key == ord(':'):
                    self.adjust_servo(6, -self.angle_step)

                # Camera controls
                elif key == ord('1'):
                    self.adjust_brightness(-5)
                elif key == ord('2'):
                    self.adjust_brightness(5)
                elif key == ord('3'):
                    self.adjust_contrast(-5)
                elif key == ord('4'):
                    self.adjust_contrast(5)
                elif key == ord('5'):
                    self.adjust_saturation(-5)
                elif key == ord('6'):
                    self.adjust_saturation(5)

                # Toggle detection
                elif key == ord('t') or key == ord('T'):
                    if self.model:
                        self.detection_enabled = not self.detection_enabled
                        status = "ON" if self.detection_enabled else "OFF"
                        print(f"Object detection: {status}")

                # Settings
                elif key == ord('+') or key == ord('='):
                    self.adjust_step_size(1)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_step_size(-1)
                elif key == ord('[') or key == ord('{'):
                    self.change_speed('down')
                elif key == ord(']') or key == ord('}'):
                    self.change_speed('up')

                # Slot selection (F1-F6)
                elif key >= 190 and key <= 195:
                    self.current_slot = key - 189
                    print(f"Selected Slot {self.current_slot}")

                # Save position (SPACE)
                elif key == 32:
                    self.save_slot_position(self.current_slot)

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

        if USE_THREADING:
            if hasattr(self.cap, 'stop'):
                self.cap.stop()
            if self.detection_thread:
                self.detection_thread.stop()
        else:
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
    configurator = SlotConfigurator()
    configurator.run()

if __name__ == "__main__":
    main()
