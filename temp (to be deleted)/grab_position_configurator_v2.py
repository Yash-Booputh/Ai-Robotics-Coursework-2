#!/usr/bin/env python3
#coding=utf-8
"""
Grab Position Configurator V2 - Multi-Waypoint System
======================================================
Configure grab sequences with multiple waypoints for safe grabbing in tight shelf.

Workflow for each slot:
1. Press F1-F6 to select slot
2. Configure WAYPOINT 1 (Safe Level Position - outside shelf, at cube height) - Press '1' to save
3. Configure WAYPOINT 2 (Grab Position - inside shelf, at cube) - Press '2' to save
4. Press SPACE to save complete sequence

IMPORTANT:
- Waypoint 1: Position arm OUTSIDE the shelf, at the SAME HEIGHT as the cube
- Waypoint 2: The actual grab position (endpoint) INSIDE the shelf, around the cube
- The robot will move horizontally from WP1 to WP2 (minimal movement)
"""

import time
import cv2
import json
import os
from Arm_Lib import Arm_Device

# Configuration
SLOT_POSITIONS_FILE = "slot_positions.json"
GRAB_POSITIONS_FILE = "grab_positions_v2.json"
NUM_SERVOS = 6
ANGLE_STEP = 2

# Slow speed for precise configuration (higher value = slower)
DEFAULT_SPEED = 1000

MOVE_SPEEDS = {
    'slow': 1000,
    'normal': 800,
    'fast': 500,
    'instant': 300
}

# Delay between movements (seconds)
MOVEMENT_DELAY = 3.0

# Gripper positions
GRIPPER_OPEN = 180
GRIPPER_CLOSED = 90


class GrabConfiguratorV2:
    def __init__(self):
        print("\n" + "="*60)
        print("GRAB POSITION CONFIGURATOR V2 - MULTI-WAYPOINT SYSTEM")
        print("="*60)

        # Initialize DOFBOT
        print("\nInitializing DOFBOT...")
        self.arm = Arm_Device()
        time.sleep(0.1)

        # Initialize camera
        print("Initializing camera...")
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✓ Camera initialized")

        # Read current servo angles
        self.current_angles = {}
        for servo_id in range(1, NUM_SERVOS + 1):
            try:
                angle = self.arm.Arm_serial_servo_read(servo_id)
                self.current_angles[servo_id] = angle if angle is not None else 90
            except:
                self.current_angles[servo_id] = 90
            time.sleep(0.01)
        print("✓ Robot arm initialized")

        # Movement settings (start with slow speed)
        self.speed_mode = 'slow'
        self.current_speed = MOVE_SPEEDS['slow']
        self.angle_step = ANGLE_STEP

        # Load slot positions
        self.slot_positions = self.load_json(SLOT_POSITIONS_FILE)
        if not self.slot_positions:
            print("\n✗ ERROR: No slot positions found!")
            print("  Run slot_position_configurator.py first")
            exit(1)
        print(f"✓ Loaded {len(self.slot_positions)} slot positions")

        # Load existing grab positions
        self.grab_positions = self.load_json(GRAB_POSITIONS_FILE)
        print(f"✓ Loaded {len(self.grab_positions)} existing grab sequences")

        # Current slot and waypoints
        self.current_slot = None
        self.waypoints = {
            1: None,  # Safe Level Position (outside shelf, at cube height)
            2: None   # Grab Position (inside shelf, at cube endpoint)
        }

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

    def save_json(self, filepath, data):
        """Save JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
            return False

    def move_servo(self, servo_id, angle):
        """Move a servo to specific angle"""
        angle = max(0, min(180, angle))
        try:
            self.arm.Arm_serial_servo_write(servo_id, angle, self.current_speed)
            self.current_angles[servo_id] = angle
            return True
        except Exception as e:
            print(f"Error moving servo {servo_id}: {e}")
            return False

    def adjust_servo(self, servo_id, delta):
        """Adjust servo by delta"""
        current = self.current_angles.get(servo_id, 90)
        new_angle = current + delta
        self.move_servo(servo_id, new_angle)

    def move_to_slot(self, slot_number):
        """Move to slot scan position with delay"""
        slot_key = f"slot_{slot_number}"
        if slot_key not in self.slot_positions:
            print(f"✗ Slot {slot_number} not found")
            return False

        print(f"\nMoving to Slot {slot_number} scan position (SLOW)...")
        slot_data = self.slot_positions[slot_key]
        angles = slot_data.get('angles', {})

        for servo_key, angle in angles.items():
            servo_id = int(servo_key.split('_')[1])
            self.move_servo(servo_id, angle)
            time.sleep(0.1)

        print(f"✓ Arrived at Slot {slot_number}")
        print(f"  Waiting {MOVEMENT_DELAY} seconds to stabilize...")
        time.sleep(MOVEMENT_DELAY)
        print("  Ready for configuration!")
        return True

    def get_current_position(self):
        """Get current position as angles dict"""
        return {f"servo_{i}": self.current_angles[i] for i in range(1, NUM_SERVOS + 1)}

    def save_waypoint(self, waypoint_num):
        """Save current position as waypoint"""
        if self.current_slot is None:
            print("✗ No slot selected! Press F1-F6 first")
            return False

        if waypoint_num not in [1, 2]:
            print(f"✗ Invalid waypoint number: {waypoint_num}")
            return False

        angles = self.get_current_position()
        self.waypoints[waypoint_num] = angles

        waypoint_names = {
            1: "Safe Level Position (outside shelf)",
            2: "Grab Position (inside shelf)"
        }
        print(f"\n" + "="*60)
        print(f"✓ WAYPOINT {waypoint_num} SAVED!")
        print(f"  {waypoint_names[waypoint_num]}")
        print("="*60)

        # Show servo angles
        print("  Servo Angles:")
        for servo_id in range(1, NUM_SERVOS + 1):
            angle = self.current_angles[servo_id]
            if servo_id == 5:
                print(f"    Servo {servo_id}: {angle}° (Gripper Rotation)")
            elif servo_id == 6:
                print(f"    Servo {servo_id}: {angle}° (Gripper Open/Close)")
            else:
                print(f"    Servo {servo_id}: {angle}°")

        print("="*60)
        return True

    def save_complete_sequence(self):
        """Save complete grab sequence for current slot"""
        if self.current_slot is None:
            print("✗ No slot selected!")
            return False

        # Check all waypoints are configured
        missing = [i for i in [1, 2] if self.waypoints[i] is None]
        if missing:
            print(f"\n✗ Missing waypoints: {missing}")
            print("  Configure waypoints 1 and 2 before saving")
            if 1 in missing:
                print("  Waypoint 1: Safe Level Position (outside shelf, at cube height)")
            if 2 in missing:
                print("  Waypoint 2: Grab Position (inside shelf, at cube)")
            return False

        slot_key = f"slot_{self.current_slot}"
        self.grab_positions[slot_key] = {
            "slot_number": self.current_slot,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "waypoint_1_safe_level": self.waypoints[1],
            "waypoint_2_grab": self.waypoints[2]
        }

        if self.save_json(GRAB_POSITIONS_FILE, self.grab_positions):
            print("\n" + "="*60)
            print(f"✓✓✓ SLOT {self.current_slot} GRAB SEQUENCE SAVED! ✓✓✓")
            print("="*60)
            print(f"Saved to: {GRAB_POSITIONS_FILE}")
            print("\nSequence configured:")
            print("  1. Safe Level Position (outside shelf, at cube height)")
            print("  2. Grab Position (inside shelf, at cube)")
            print("\nExecution will:")
            print("  - Move to WP1 (safe level position)")
            print("  - Open gripper")
            print("  - Move to WP2 (grab position)")
            print("  - Close gripper")
            print("="*60)
            return True
        else:
            print("✗ Failed to save!")
            return False

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
        print(f"Speed: {self.speed_mode.upper()} ({self.current_speed}ms)")

    def adjust_step_size(self, delta):
        """Adjust angle step size"""
        self.angle_step = max(1, min(10, self.angle_step + delta))
        print(f"Step size: {self.angle_step}°")

    def display_overlay(self, frame):
        """Display servo info and instructions on frame"""
        h, w = frame.shape[:2]

        # Title
        cv2.putText(frame, "GRAB CONFIGURATOR V2", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Current slot
        if self.current_slot:
            slot_text = f"SLOT {self.current_slot}"
            cv2.putText(frame, slot_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No slot selected (F1-F6)", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Waypoint status
        y_offset = 80
        for wp_num in [1, 2]:
            status = "✓" if self.waypoints[wp_num] else "✗"
            color = (0, 255, 0) if self.waypoints[wp_num] else (0, 0, 255)
            wp_names = {1: "Safe Level", 2: "Grab"}
            text = f"WP{wp_num} ({wp_names[wp_num]}): {status}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25

        # Servo angles (left side)
        y_offset = 170
        for servo_id in range(1, NUM_SERVOS + 1):
            angle = self.current_angles.get(servo_id, 0)

            # Highlight servo 5 (gripper rotation) and servo 6 (gripper)
            if servo_id == 5:
                color = (0, 255, 255)  # Cyan
                text = f">S{servo_id}:{angle:3}° GRIP-ROT<"
            elif servo_id == 6:
                color = (255, 0, 255)  # Magenta
                text = f">S{servo_id}:{angle:3}° GRIPPER<"
            else:
                color = (0, 255, 0)
                text = f"S{servo_id}:{angle:3}°"

            cv2.putText(frame, text, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18

        # Speed and step (right side)
        right_x = w - 130
        cv2.putText(frame, f"Speed:{self.speed_mode[:3].upper()}", (right_x, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        cv2.putText(frame, f"Step:{self.angle_step}°", (right_x, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        # Controls (bottom)
        controls = [
            "F1-F6:Select Slot | 1:WP1 | 2:WP2",
            "WASD/IKJL/UO/P;:Move | SPACE:Save",
            "+/-:Step | []:Speed | Q:Quit"
        ]
        y_offset = h - 60
        for control in controls:
            cv2.putText(frame, control, (5, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 16

        return frame

    def run(self):
        """Main configuration loop"""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS")
        print("="*60)
        print("SELECT SLOT:")
        print("  F1-F6       - Select and move to slot 1-6")
        print("\nCONFIGURE WAYPOINTS:")
        print("  1           - Save Waypoint 1 (Safe Level - OUTSIDE shelf, at cube height)")
        print("  2           - Save Waypoint 2 (Grab Position - INSIDE shelf, at cube)")
        print("\nMOVE ROBOT:")
        print("  W/S         - Servo 1 (Base)")
        print("  A/D         - Servo 2 (Shoulder)")
        print("  I/K         - Servo 3 (Elbow)")
        print("  J/L         - Servo 4 (Wrist pitch)")
        print("  U/O         - Servo 5 (Gripper rotation) ← IMPORTANT")
        print("  P/;         - Servo 6 (Gripper open/close)")
        print("\nADJUST:")
        print("  +/-         - Change step size")
        print("  [ / ]       - Slower / Faster speed")
        print("\nSAVE:")
        print("  SPACE       - Save complete sequence (waypoints 1 & 2)")
        print("  Q           - Quit")
        print("="*60)
        print(f"\nSPEED: {self.speed_mode.upper()} ({self.current_speed}ms)")
        print(f"MOVEMENT DELAY: {MOVEMENT_DELAY} seconds between movements")
        print("="*60)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Add overlay
                display_frame = self.display_overlay(frame)
                cv2.imshow("Grab Configurator V2", display_frame)

                key = cv2.waitKey(1) & 0xFF

                # Quit
                if key == ord('q') or key == ord('Q'):
                    break

                # Select slot (F1-F6)
                elif key == 0xFF & 0xFFBE:  # F1
                    self.current_slot = 1
                    self.waypoints = {1: None, 2: None}
                    self.move_to_slot(1)
                elif key == 0xFF & 0xFFBF:  # F2
                    self.current_slot = 2
                    self.waypoints = {1: None, 2: None}
                    self.move_to_slot(2)
                elif key == 0xFF & 0xFFC0:  # F3
                    self.current_slot = 3
                    self.waypoints = {1: None, 2: None}
                    self.move_to_slot(3)
                elif key == 0xFF & 0xFFC1:  # F4
                    self.current_slot = 4
                    self.waypoints = {1: None, 2: None}
                    self.move_to_slot(4)
                elif key == 0xFF & 0xFFC2:  # F5
                    self.current_slot = 5
                    self.waypoints = {1: None, 2: None}
                    self.move_to_slot(5)
                elif key == 0xFF & 0xFFC3:  # F6
                    self.current_slot = 6
                    self.waypoints = {1: None, 2: None}
                    self.move_to_slot(6)

                # Save waypoints
                elif key == ord('1'):
                    self.save_waypoint(1)
                elif key == ord('2'):
                    self.save_waypoint(2)

                # Save complete sequence
                elif key == ord(' '):
                    self.save_complete_sequence()

                # Servo controls
                elif key == ord('w') or key == ord('W'):
                    self.adjust_servo(1, self.angle_step)
                elif key == ord('s') or key == ord('S'):
                    self.adjust_servo(1, -self.angle_step)
                elif key == ord('a') or key == ord('A'):
                    self.adjust_servo(2, self.angle_step)
                elif key == ord('d') or key == ord('D'):
                    self.adjust_servo(2, -self.angle_step)
                elif key == ord('i') or key == ord('I'):
                    self.adjust_servo(3, self.angle_step)
                elif key == ord('k') or key == ord('K'):
                    self.adjust_servo(3, -self.angle_step)
                elif key == ord('j') or key == ord('J'):
                    self.adjust_servo(4, self.angle_step)
                elif key == ord('l') or key == ord('L'):
                    self.adjust_servo(4, -self.angle_step)
                elif key == ord('u') or key == ord('U'):
                    self.adjust_servo(5, self.angle_step)
                elif key == ord('o') or key == ord('O'):
                    self.adjust_servo(5, -self.angle_step)
                elif key == ord('p') or key == ord('P'):
                    self.adjust_servo(6, self.angle_step)
                elif key == ord(';') or key == ord(':'):
                    self.adjust_servo(6, -self.angle_step)

                # Adjust settings
                elif key == ord('+') or key == ord('='):
                    self.adjust_step_size(1)
                elif key == ord('-') or key == ord('_'):
                    self.adjust_step_size(-1)
                elif key == ord('['):
                    self.change_speed('down')
                elif key == ord(']'):
                    self.change_speed('up')

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
    configurator = GrabConfiguratorV2()
    configurator.run()


if __name__ == "__main__":
    main()
