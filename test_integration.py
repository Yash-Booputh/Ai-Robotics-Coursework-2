#!/usr/bin/env python3
"""
Test Integration - Diagnostic script to test RobotPatrolSystem integration
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test if all modules can be imported"""
    logger.info("="*60)
    logger.info("TEST 1: Module Imports")
    logger.info("="*60)

    try:
        from robot import DofbotController, VisionSystem, PickSequence, RobotPatrolSystem
        logger.info("✅ All robot modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_vision_system():
    """Test VisionSystem initialization and camera"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: VisionSystem")
    logger.info("="*60)

    try:
        from robot import VisionSystem

        logger.info("Creating VisionSystem...")
        vision = VisionSystem()
        logger.info(f"✓ VisionSystem created")
        logger.info(f"  Camera active: {vision.is_camera_active}")
        logger.info(f"  Model loaded: {vision.is_model_loaded}")

        # Try to start camera
        logger.info("\nStarting camera...")
        if vision.start_camera():
            logger.info("✅ Camera started successfully")
            logger.info(f"  Camera active: {vision.is_camera_active}")

            # Test frame capture
            frame, detection = vision.capture_and_detect()
            if frame is not None:
                logger.info(f"✅ Frame captured: {frame.shape}")
            else:
                logger.warning("⚠️  No frame captured")

            vision.stop_camera()
            logger.info("  Camera stopped")
            return True
        else:
            logger.error("❌ Failed to start camera")
            return False

    except Exception as e:
        logger.error(f"❌ VisionSystem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robot_patrol_system():
    """Test RobotPatrolSystem with VisionSystem"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: RobotPatrolSystem Integration")
    logger.info("="*60)

    try:
        from robot import VisionSystem, RobotPatrolSystem

        logger.info("Creating VisionSystem...")
        vision = VisionSystem()

        logger.info("Starting camera...")
        if not vision.start_camera():
            logger.error("❌ Failed to start camera")
            return False

        logger.info("\nCreating RobotPatrolSystem with active camera...")
        patrol = RobotPatrolSystem(vision)
        logger.info("✅ RobotPatrolSystem created successfully")

        # Check configuration
        logger.info(f"\nConfiguration:")
        logger.info(f"  Slot positions loaded: {len(patrol.slot_positions)}")
        logger.info(f"  Grab positions loaded: {len(patrol.grab_positions)}")

        # Cleanup
        patrol.cleanup()
        vision.stop_camera()

        return True

    except Exception as e:
        logger.error(f"❌ RobotPatrolSystem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pick_sequence():
    """Test PickSequence with lazy initialization"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: PickSequence Lazy Initialization")
    logger.info("="*60)

    try:
        from robot import DofbotController, VisionSystem, PickSequence
        from config.recipes import get_pizza_ingredients

        logger.info("Creating DofbotController...")
        robot = DofbotController()

        logger.info("Creating VisionSystem...")
        vision = VisionSystem()

        logger.info("Starting camera...")
        if not vision.start_camera():
            logger.error("❌ Failed to start camera")
            return False

        logger.info("\nCreating PickSequence (patrol system should NOT initialize yet)...")
        pick_seq = PickSequence(robot, vision)
        logger.info(f"✅ PickSequence created")
        logger.info(f"  Patrol system initialized: {pick_seq.patrol_system_initialized}")
        logger.info(f"  Patrol system object: {pick_seq.patrol_system}")

        logger.info("\nStarting order (patrol system should initialize now)...")
        if pick_seq.start_order("Margherita"):
            logger.info("✅ Order started")
            logger.info(f"  Pizza: {pick_seq.current_pizza}")
            logger.info(f"  Ingredients: {pick_seq.ingredients_needed}")
            logger.info(f"  Total: {pick_seq.total_ingredients}")

            # Cancel immediately to avoid actually running
            pick_seq.cancel_order()
            logger.info("  Order cancelled (test complete)")
        else:
            logger.error("❌ Failed to start order")
            return False

        # Cleanup
        pick_seq.cleanup()
        vision.stop_camera()
        robot.disconnect()

        return True

    except Exception as e:
        logger.error(f"❌ PickSequence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "🧪"*30)
    logger.info("CHEFMATE INTEGRATION DIAGNOSTIC TESTS")
    logger.info("🧪"*30 + "\n")

    tests = [
        ("Module Imports", test_imports),
        ("VisionSystem", test_vision_system),
        ("RobotPatrolSystem", test_robot_patrol_system),
        ("PickSequence", test_pick_sequence),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    logger.info("\n" + "="*60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("="*60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
