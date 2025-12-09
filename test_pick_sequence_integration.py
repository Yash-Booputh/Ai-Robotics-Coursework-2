#!/usr/bin/env python3
"""
Test PickSequence with IntegratedPatrolGrabSystem
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
print("TEST: PickSequence + IntegratedPatrolGrabSystem")
print("="*60)

# Create components (VisionSystem won't be used by IntegratedPatrolGrabSystem)
print("\n1. Creating DofbotController...")
robot = DofbotController()

print("\n2. Creating VisionSystem (will be ignored by IntegratedPatrolGrabSystem)...")
vision = VisionSystem()

print("\n3. Creating PickSequence...")
pick_seq = PickSequence(robot, vision)

print("\n4. Starting order for Margherita...")
if pick_seq.start_order("Margherita"):
    print("✅ Order started successfully!")
    print(f"   Ingredients: {pick_seq.ingredients_needed}")

    # Don't actually execute - just test initialization
    print("\n5. Cancelling order (test only)...")
    pick_seq.cancel_order()

    print("\n✅ TEST PASSED - Integration works!")
else:
    print("\n❌ TEST FAILED - Order start failed")

# Cleanup
pick_seq.cleanup()
robot.disconnect()

print("\n" + "="*60)
print("Test complete")
print("="*60)
