#!/usr/bin/env python3
"""
ChefMate Robot Assistant - Position Testing Tool
Test calibrated positions before running full program
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import DofbotController
from config.positions import get_all_slot_names, get_slot_position
import time

def test_home_position(robot):
    """Test home position"""
    print("\n" + "="*70)
    print("  TESTING HOME POSITION")
    print("="*70)
    input("\nPress ENTER to move to HOME position...")

    robot.move_to_home()
    print("✅ Moved to HOME")
    time.sleep(2)

def test_slot_positions(robot, slot_name):
    """Test all positions for a slot"""
    print("\n" + "="*70)
    print(f"  TESTING {slot_name.upper()}")
    print("="*70)

    positions_to_test = [
        ("scout_top", "Top-down scout view"),
        ("scout_angle", "45° angle scout view"),
        ("approach", "Approach position (above cube)"),
        ("grab", "Grab position (touching cube)"),
        ("lift", "Lift position (cube lifted)")
    ]

    for position_type, description in positions_to_test:
        print(f"\n→ {description}")
        input("  Press ENTER to move... ")

        position = get_slot_position(slot_name, position_type)

        if position is None:
            print(f"  ❌ Position not found: {position_type}")
            continue

        try:
            robot.move_servos(position, 1000)
            print(f"  ✅ Moved to {position_type}: {position}")
            time.sleep(1.5)
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_delivery_position(robot):
    """Test delivery position"""
    print("\n" + "="*70)
    print("  TESTING DELIVERY POSITION")
    print("="*70)
    input("\nPress ENTER to move to DELIVERY position...")

    robot.move_to_delivery()
    print("✅ Moved to DELIVERY")
    time.sleep(2)

def test_gripper(robot):
    """Test gripper"""
    print("\n" + "="*70)
    print("  TESTING GRIPPER")
    print("="*70)

    input("\nPress ENTER to OPEN gripper...")
    robot.gripper_open()
    print("✅ Gripper OPEN")
    time.sleep(2)

    input("Press ENTER to CLOSE gripper...")
    robot.gripper_close()
    print("✅ Gripper CLOSED")
    time.sleep(2)

    input("Press ENTER to OPEN gripper again...")
    robot.gripper_open()
    print("✅ Gripper OPEN")

def main():
    """Main test function"""
    print("\n" + "="*70)
    print("  CHEFMATE POSITION TEST UTILITY")
    print("="*70)
    print("\nThis tool tests calibrated positions.")
    print("Make sure robot has clear workspace!")
    print("="*70 + "\n")

    # Initialize robot
    print("Initializing robot...")
    robot = DofbotController()

    if not robot.check_connection():
        print("❌ Robot not connected!")
        print("\nRunning in SIMULATION mode (no actual movement)")
        print("Positions will be displayed but not executed.\n")

    try:
        # Test home position
        test_home_position(robot)

        # Test gripper
        test_gripper(robot)

        # Get all slots
        all_slots = get_all_slot_names()

        # Test each slot
        for slot_name in all_slots:
            response = input(f"\nTest {slot_name}? (y/n/q to quit): ").lower()

            if response == 'q':
                break
            elif response == 'y':
                test_slot_positions(robot, slot_name)

                # Return to home after each slot
                print("\nReturning to home...")
                robot.move_to_home()

        # Test delivery position
        response = input("\nTest DELIVERY position? (y/n): ").lower()
        if response == 'y':
            test_delivery_position(robot)

        # Final home
        print("\n" + "="*70)
        print("  TEST COMPLETE")
        print("="*70)
        print("\nReturning to home position...")
        robot.move_to_home()

        print("\n✅ All tests completed successfully!")
        print("\nIf all positions look correct, you can run: python main.py")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
