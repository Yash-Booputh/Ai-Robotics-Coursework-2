#!/usr/bin/env python3
"""
ChefMate Robot Assistant - Position Calibration Tool
Helps you calibrate robot positions for each slot
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Arm_Lib import Arm_Device
import time
import json

# Slot names
SLOTS = ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6"]

# Position types for each slot
POSITION_TYPES = ["scout_top", "scout_angle", "approach", "grab", "lift"]

def print_header():
    """Print calibration header"""
    print("\n" + "=" * 70)
    print("  CHEFMATE ROBOT CALIBRATION UTILITY")
    print("=" * 70)
    print("\nThis tool helps you calibrate positions for all 6 slots.")
    print("For each slot, you'll set 5 positions:")
    print("  1. scout_top   - Top-down camera view")
    print("  2. scout_angle - 45° angle camera view (backup)")
    print("  3. approach    - Position above cube")
    print("  4. grab        - Contact position with cube")
    print("  5. lift        - Lifted cube position")
    print("\n" + "=" * 70 + "\n")

def get_current_position(arm):
    """Read current servo positions"""
    positions = []
    for servo_id in range(1, 6):  # Servos 1-5
        attempts = 0
        while attempts < 10:
            angle = arm.Arm_serial_servo_read(servo_id)
            if angle is not None:
                positions.append(angle)
                break
            attempts += 1
            time.sleep(0.01)
        if attempts >= 10:
            print(f"⚠️  Warning: Failed to read Servo {servo_id}")
            positions.append(90)  # Default
    return positions

def print_position(positions):
    """Print positions in readable format"""
    print(f"  [S1:{positions[0]:3d}, S2:{positions[1]:3d}, S3:{positions[2]:3d}, S4:{positions[3]:3d}, S5:{positions[4]:3d}]")

def calibrate_slot(arm, slot_name):
    """Calibrate all positions for a single slot"""
    print(f"\n{'='*70}")
    print(f"  CALIBRATING {slot_name.upper()}")
    print(f"{'='*70}\n")

    slot_positions = {}

    for position_type in POSITION_TYPES:
        print(f"\n→ Position: {position_type}")
        print("-" * 70)

        if position_type == "scout_top":
            print("  Move robot to TOP-DOWN camera view of this slot")
            print("  Camera should see cube clearly from above")
        elif position_type == "scout_angle":
            print("  Move robot to 45° ANGLE camera view of this slot")
            print("  This is backup view if top-down fails")
        elif position_type == "approach":
            print("  Move gripper ABOVE the cube (ready to descend)")
        elif position_type == "grab":
            print("  Move gripper DOWN to TOUCH the cube")
        elif position_type == "lift":
            print("  Move gripper UP with cube LIFTED clear")

        print("\n  Use Yahboom control software to move robot manually.")
        input("  Press ENTER when robot is in position... ")

        # Read current position
        positions = get_current_position(arm)
        print(f"\n  ✓ Recorded: ", end="")
        print_position(positions)

        slot_positions[position_type] = positions

    return slot_positions

def save_to_python_file(all_calibrations):
    """Save calibrations to positions.py file"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    positions_file = os.path.join(config_dir, "positions.py")

    print(f"\n{'='*70}")
    print("  SAVING CALIBRATION DATA")
    print(f"{'='*70}\n")

    # Read existing file
    with open(positions_file, 'r') as f:
        lines = f.readlines()

    # Find SLOT_POSITIONS section and update it
    new_lines = []
    skip_mode = False

    for line in lines:
        if "SLOT_POSITIONS = {" in line:
            skip_mode = True
            new_lines.append(line)

            # Write calibrated positions
            for i, slot_name in enumerate(SLOTS):
                slot_data = all_calibrations[slot_name]
                new_lines.append(f'    "{slot_name}": {{\n')
                new_lines.append(f'        "scout_top": {slot_data["scout_top"]},\n')
                new_lines.append(f'        "scout_angle": {slot_data["scout_angle"]},\n')
                new_lines.append(f'        "approach": {slot_data["approach"]},\n')
                new_lines.append(f'        "grab": {slot_data["grab"]},\n')
                new_lines.append(f'        "lift": {slot_data["lift"]},\n')

                # Add location based on slot
                locations = ["top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"]
                descriptions = ["Top left slot", "Top right slot", "Middle left slot", "Middle right slot", "Bottom left slot", "Bottom right slot"]

                new_lines.append(f'        "location": "{locations[i]}",\n')
                new_lines.append(f'        "description": "{descriptions[i]}"\n')

                if i < len(SLOTS) - 1:
                    new_lines.append('    },\n\n')
                else:
                    new_lines.append('    }\n')

            continue

        if skip_mode and line.strip() == "}":
            skip_mode = False
            new_lines.append(line)
            continue

        if not skip_mode:
            new_lines.append(line)

    # Write back to file
    with open(positions_file, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ Saved calibrations to: {positions_file}")

def main():
    """Main calibration function"""
    print_header()

    # Initialize robot
    print("Initializing robot arm...")
    try:
        arm = Arm_Device()
        time.sleep(0.5)
        print("✅ Robot connected\n")
    except Exception as e:
        print(f"❌ Failed to connect to robot: {e}")
        print("\nMake sure:")
        print("  1. Robot is powered on")
        print("  2. USB cable is connected")
        print("  3. Arm_Lib is installed")
        return

    # Calibrate all slots
    all_calibrations = {}

    try:
        for slot_name in SLOTS:
            all_calibrations[slot_name] = calibrate_slot(arm, slot_name)

            # Ask if user wants to continue
            if slot_name != SLOTS[-1]:  # Not the last slot
                print(f"\n{'='*70}")
                response = input(f"Continue to next slot? (y/n): ").lower()
                if response != 'y':
                    print("\nCalibration incomplete. Exiting...")
                    return

        # Save calibrations
        save_to_python_file(all_calibrations)

        print(f"\n{'='*70}")
        print("  CALIBRATION COMPLETE!")
        print(f"{'='*70}\n")
        print("Next steps:")
        print("  1. Test positions with: python tools/test_positions.py")
        print("  2. Run the main app: python main.py")
        print(f"\n{'='*70}\n")

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during calibration: {e}")
    finally:
        # Move to home
        print("\nMoving to home position...")
        try:
            arm.Arm_serial_servo_write6([90, 164, 18, 0, 90, 90], 1000)
        except:
            pass

if __name__ == "__main__":
    main()
