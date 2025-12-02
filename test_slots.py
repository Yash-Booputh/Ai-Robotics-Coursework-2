#!/usr/bin/env python3
"""
Quick Slot Position Test
Tests if robot can reach all slot positions safely
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot import DofbotController
import time

def test_single_slot(robot, slot_name):
    """Test all positions for one slot"""
    print(f"\n{'='*60}")
    print(f"  Testing {slot_name}")
    print(f"{'='*60}")

    # Test scout_top
    print(f"\n1. Moving to {slot_name} scout_top (camera view)...")
    input("   Press ENTER to continue (or Ctrl+C to stop)...")

    try:
        robot.scout_slot(slot_name, "top")
        print("   ✅ Reached scout_top position")
        time.sleep(2)
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Test scout_angle
    print(f"\n2. Moving to {slot_name} scout_angle (45° view)...")
    input("   Press ENTER to continue...")

    try:
        robot.scout_slot(slot_name, "angle")
        print("   ✅ Reached scout_angle position")
        time.sleep(2)
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Return home
    print(f"\n3. Returning home...")
    robot.move_to_home()
    time.sleep(1)

    print(f"\n✅ {slot_name} positions OK!")
    return True

def main():
    print("\n" + "="*60)
    print("  ChefMate Slot Position Test")
    print("="*60)
    print("\nThis will test scout positions for each slot.")
    print("Watch robot carefully! Press Ctrl+C for emergency stop.")
    print("\nMake sure:")
    print("  - Robot is powered on")
    print("  - Workspace is clear")
    print("  - Emergency stop is accessible")
    print("="*60 + "\n")

    input("Press ENTER to start...")

    # Initialize robot
    print("\nConnecting to robot...")
    robot = DofbotController()

    if not robot.check_connection():
        print("\n❌ Robot not connected!")
        print("\nCheck:")
        print("  - USB cable connected")
        print("  - Robot powered on")
        print("  - Arm_Lib installed")
        return

    print("✅ Robot connected!\n")

    # Test home position first
    print("Moving to HOME position...")
    robot.move_to_home()
    time.sleep(2)
    print("✅ Home position OK\n")

    # Test each slot
    slots = ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6"]
    results = {}

    for slot_name in slots:
        response = input(f"\nTest {slot_name}? (y/n/q to quit): ").lower()

        if response == 'q':
            print("\nQuitting...")
            break
        elif response == 'y':
            results[slot_name] = test_single_slot(robot, slot_name)
        else:
            results[slot_name] = None  # Skipped

    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)

    for slot_name, result in results.items():
        if result is True:
            print(f"  {slot_name}: ✅ PASSED")
        elif result is False:
            print(f"  {slot_name}: ❌ FAILED")
        else:
            print(f"  {slot_name}: ⊘ SKIPPED")

    print("\n" + "="*60)
    print("Returning to home...")
    robot.move_to_home()
    print("="*60 + "\n")

    robot.disconnect()
    print("✅ Test complete!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🚨 EMERGENCY STOP!")
        print("Powering down robot...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
