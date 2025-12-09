#!/usr/bin/env python3
"""
Test complete order execution with IntegratedPatrolGrabSystem
This will actually patrol and try to grab ingredients
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from robot import DofbotController, VisionSystem, PickSequence

print("="*60)
print("TEST: Full Order Execution")
print("="*60)

# Create components
print("\n1. Creating DofbotController...")
robot = DofbotController()

print("\n2. Creating VisionSystem (for display only)...")
vision = VisionSystem()

print("\n3. Creating PickSequence...")
pick_seq = PickSequence(robot, vision)

print("\n4. Starting order for Margherita...")
if pick_seq.start_order("Margherita"):
    print("✅ Order started successfully!")
    print(f"   Ingredients: {pick_seq.ingredients_needed}")

    print("\n5. Executing first ingredient pick...")
    print("   This will actually patrol and grab the ingredient!")
    print("   Press Ctrl+C if you want to stop\n")

    try:
        # Pick just the first ingredient as a test
        success = pick_seq.pick_next_ingredient()

        if success:
            print("\n✅ TEST PASSED - First ingredient picked successfully!")
        else:
            print("\n❌ TEST FAILED - Could not pick ingredient")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        pick_seq.cancel_order()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        pick_seq.cancel_order()
else:
    print("\n❌ TEST FAILED - Order start failed")

# Cleanup
print("\n6. Cleaning up...")
pick_seq.cleanup()
robot.disconnect()

print("\n" + "="*60)
print("Test complete")
print("="*60)
