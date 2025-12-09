"""
ChefMate Robot Assistant - Robot Patrol System
Integrated patrol and grab system for multiple ingredients using shared camera
"""

import time
import json
import os
import logging
from typing import Optional, List, Tuple
from Arm_Lib import Arm_Device

from .vision_system import VisionSystem
from config.recipes import get_pizza_ingredients, get_ingredient_display_name


# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
SLOT_POSITIONS_FILE = "slot_positions.json"
GRAB_POSITIONS_FILE = "grab_positions.json"

# Robot configuration
NUM_SERVOS = 6

# Movement speeds (higher value = slower)
MOVE_SPEED = 800           # Normal movement
PATROL_SPEED = 500         # Patrol movement (faster)
SAFE_LEVEL_SPEED = 1000    # Moving to safe level position
GRAB_SPEED = 1000          # Grabbing motion (very slow)

# Delays between movements
MOVEMENT_DELAY = 3.0   # 3 seconds between movements

# Detection configuration
CONFIDENCE_THRESHOLD = 0.5
DETECTION_SAMPLES = 3  # Number of consecutive detections required to confirm

# Gripper positions (servo 6 is reversed!)
GRIPPER_OPEN = 0      # 0° = OPEN


# ============================================================================
# ROBOT PATROL SYSTEM
# ============================================================================

class RobotPatrolSystem:
    """
    Integrated patrol and grab system that uses shared VisionSystem camera
    Handles multiple ingredients for a complete pizza order
    """

    def __init__(self, vision_system: VisionSystem):
        """
        Initialize the patrol system with shared vision system

        Args:
            vision_system: Shared VisionSystem instance with active camera
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*60)
        self.logger.info("ROBOT PATROL SYSTEM - Initializing")
        self.logger.info("="*60)

        # Use shared vision system instead of creating our own camera
        self.vision = vision_system

        # Check if camera is active
        if not self.vision.is_camera_active:
            self.logger.warning("⚠️  VisionSystem camera is NOT active!")
            self.logger.warning("    Camera must be started before initializing RobotPatrolSystem")
        else:
            self.logger.info("✓ VisionSystem camera is active")

        # Check if model is loaded
        if not self.vision.is_model_loaded:
            self.logger.warning("⚠️  YOLO model is NOT loaded!")
        else:
            self.logger.info("✓ YOLO model is loaded")

        self.logger.info("✓ Using shared VisionSystem")

        # Initialize robot arm
        self.logger.info("Initializing DOFBOT...")
        self.arm = Arm_Device()
        time.sleep(0.1)
        self.logger.info("✓ Robot arm initialized")

        # Load positions
        self.slot_positions = self.load_json(SLOT_POSITIONS_FILE)
        self.grab_positions = self.load_json(GRAB_POSITIONS_FILE)

        if not self.slot_positions:
            self.logger.error("No slot positions found! Run slot_position_configurator.py first")
            raise RuntimeError("No slot positions configured")

        if not self.grab_positions:
            self.logger.error("No grab positions found! Run grab_position_configurator_v2.py first")
            raise RuntimeError("No grab positions configured")

        self.logger.info(f"✓ Loaded {len(self.slot_positions)} slot positions")
        self.logger.info(f"✓ Loaded {len(self.grab_positions)} grab sequences")

        # Current angles
        self.current_angles = {}
        self.read_current_angles()

        self.logger.info("="*60)
        self.logger.info("✓ ROBOT PATROL SYSTEM READY")
        self.logger.info("="*60)

    def load_json(self, filepath):
        """Load JSON file"""
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return {}

    def read_current_angles(self):
        """Read current servo angles"""
        for servo_id in range(1, NUM_SERVOS + 1):
            try:
                angle = self.arm.Arm_serial_servo_read(servo_id)
                if angle is not None:
                    self.current_angles[servo_id] = angle
            except:
                pass

    def move_servo(self, servo_id, angle, speed=MOVE_SPEED):
        """Move a specific servo to an angle"""
        angle = max(0, min(180, angle))
        try:
            self.arm.Arm_serial_servo_write(servo_id, angle, speed)
            self.current_angles[servo_id] = angle
            return True
        except Exception as e:
            self.logger.error(f"Error moving servo {servo_id}: {e}")
            return False

    def move_to_position(self, angles_dict, speed=MOVE_SPEED, description="position", exclude_gripper=False):
        """Move to a position defined by angles dictionary"""
        self.logger.info(f"  Moving to {description}...")

        # Log which servos will move
        servos_to_move = []
        for servo_key, angle in angles_dict.items():
            servo_id = int(servo_key.split('_')[1])
            if exclude_gripper and servo_id == 6:
                self.logger.debug(f"    SKIP Servo {servo_id} (gripper excluded)")
            else:
                servos_to_move.append((servo_id, angle))

        # Execute movements
        for servo_id, angle in servos_to_move:
            self.logger.debug(f"    MOVE Servo {servo_id} → {angle}°")
            self.move_servo(servo_id, angle, speed)
            time.sleep(0.05)

        time.sleep(0.5)
        self.logger.info(f"  ✓ Reached {description}")

    def move_to_slot(self, slot_key):
        """Move to slot scanning position"""
        if slot_key not in self.slot_positions:
            self.logger.error(f"Slot {slot_key} not found in slot positions")
            return False

        slot_data = self.slot_positions[slot_key]
        angles = slot_data.get('angles', {})
        # EXCLUDE GRIPPER when moving to scan position
        self.move_to_position(angles, PATROL_SPEED, f"{slot_key} scan position", exclude_gripper=True)

        # Extra stabilization delay after reaching slot position
        # This allows the robot arm to fully settle and camera to stabilize
        time.sleep(1.0)

        return True

    def detect_ingredient(self, num_samples=DETECTION_SAMPLES) -> Tuple[Optional[str], float]:
        """
        Detect ingredient at current position using VisionSystem

        Args:
            num_samples: Number of samples to take for voting

        Returns:
            Tuple of (ingredient_name, confidence)
        """
        self.logger.info(f"  📸 Scanning: taking {num_samples} sample images...")

        detection_votes = {}

        for sample in range(num_samples):
            self.logger.debug(f"    Sample {sample + 1}/{num_samples}...")

            # Use VisionSystem to capture and detect
            frame, detection = self.vision.capture_and_detect()

            if detection and detection.get('confidence', 0) >= CONFIDENCE_THRESHOLD:
                ingredient = detection['class_name']
                confidence = detection['confidence']

                self.logger.debug(f"      Detected: {ingredient} ({confidence:.2f})")

                if ingredient not in detection_votes:
                    detection_votes[ingredient] = {'count': 0, 'total_conf': 0.0}

                detection_votes[ingredient]['count'] += 1
                detection_votes[ingredient]['total_conf'] += confidence
            else:
                self.logger.debug(f"      No detection")

            # Small delay between samples to allow for variation
            time.sleep(0.2)

        if detection_votes:
            best_ingredient = max(detection_votes.items(),
                                 key=lambda x: (x[1]['count'], x[1]['total_conf']))
            ingredient_name = best_ingredient[0]
            avg_confidence = best_ingredient[1]['total_conf'] / best_ingredient[1]['count']
            vote_count = best_ingredient[1]['count']

            self.logger.info(f"  ✓ Detected: {ingredient_name} (confidence: {avg_confidence:.2f}, {vote_count}/{num_samples} samples)")
            return ingredient_name, avg_confidence
        else:
            self.logger.info("  ✗ Nothing detected")
            return None, 0.0

    def execute_grab_sequence(self, slot_key):
        """Execute 3-waypoint grab and pickup sequence"""
        if slot_key not in self.grab_positions:
            self.logger.error(f"Grab sequence for {slot_key} not configured")
            return False

        grab_data = self.grab_positions[slot_key]

        # Get waypoint 2 to extract the closed gripper angle
        waypoint_2 = grab_data.get('waypoint_2_grab')
        if not waypoint_2:
            self.logger.error("Waypoint 2 not found!")
            return False

        # Get the gripper closed angle from waypoint 2
        gripper_closed_angle = waypoint_2.get('servo_6', 136)
        self.logger.info(f"  Gripper closed angle from WP2: {gripper_closed_angle}°")

        self.logger.info("  Executing grab sequence:")

        # Step 1: Open gripper FIRST (before any movement)
        self.logger.info("  [1/5] Opening gripper...")
        self.move_servo(6, GRIPPER_OPEN, GRAB_SPEED)
        time.sleep(1.0)

        # Step 2: Move to Waypoint 1 (Safe Level Position - outside shelf)
        self.logger.info("  [2/5] Moving to Safe Level Position (Waypoint 1)...")
        waypoint_1 = grab_data.get('waypoint_1_safe_level')
        if not waypoint_1:
            self.logger.error("Waypoint 1 not found!")
            return False
        self.move_to_position(waypoint_1, SAFE_LEVEL_SPEED, "safe level position", exclude_gripper=True)
        time.sleep(MOVEMENT_DELAY)

        # Step 3: Move to Waypoint 2 (Grab Position - inside shelf)
        self.logger.info("  [3/5] Moving to Grab Position (Waypoint 2)...")
        self.move_to_position(waypoint_2, GRAB_SPEED, "grab position", exclude_gripper=True)
        time.sleep(MOVEMENT_DELAY)

        # Step 4: Close gripper to grab (use angle from waypoint 2)
        self.logger.info(f"  [4/5] Closing gripper to grab (angle: {gripper_closed_angle}°)...")
        self.move_servo(6, gripper_closed_angle, GRAB_SPEED)
        time.sleep(2.0)

        # Step 5: Pickup sequence - return to Waypoint 1 with gripper CLOSED
        self.logger.info("  [5/5] Pickup - returning to Safe Level Position (Waypoint 1)...")
        self.move_to_position(waypoint_1, SAFE_LEVEL_SPEED, "safe level position (pickup)", exclude_gripper=True)
        time.sleep(MOVEMENT_DELAY)

        # Final step: Open gripper after reaching safe position
        self.logger.info("  [FINAL] Opening gripper after reaching safe position...")
        self.move_servo(6, GRIPPER_OPEN, GRAB_SPEED)
        time.sleep(1.0)

        self.logger.info("  ✓ Grab and pickup sequence complete!")
        return True

    def patrol_and_find(self, target_ingredient: str) -> Optional[str]:
        """
        Patrol through all slots looking for target ingredient
        When found with confidence > threshold, automatically execute grab sequence

        Args:
            target_ingredient: Name of ingredient to find

        Returns:
            slot_key if found and grabbed, None otherwise
        """
        self.logger.info("="*60)
        self.logger.info(f"PATROL - Looking for: {target_ingredient}")
        self.logger.info("="*60)

        # Get sorted slot names
        slot_keys = sorted(self.slot_positions.keys())

        for slot_key in slot_keys:
            self.logger.info(f"[{slot_key}]")

            # Move to slot
            if not self.move_to_slot(slot_key):
                continue

            # Detect what's there
            detected_ingredient, confidence = self.detect_ingredient()

            # Check if we found what we're looking for
            if detected_ingredient == target_ingredient and confidence >= CONFIDENCE_THRESHOLD:
                self.logger.info(f"🎯 FOUND {target_ingredient} at {slot_key}!")
                self.logger.info(f"   Confidence: {confidence:.2f}")

                # AUTOMATICALLY execute grab sequence
                self.logger.info("="*60)
                self.logger.info("EXECUTING GRAB SEQUENCE")
                self.logger.info("="*60)

                success = self.execute_grab_sequence(slot_key)

                if success:
                    self.logger.info("="*60)
                    self.logger.info(f"✓✓✓ SUCCESS! {target_ingredient} GRABBED FROM {slot_key}")
                    self.logger.info("="*60)
                    return slot_key
                else:
                    self.logger.error("="*60)
                    self.logger.error(f"✗ FAILED to grab {target_ingredient} from {slot_key}")
                    self.logger.error("="*60)
                    return None

            # Not found, continue patrol
            self.logger.info(f"  ➜ Not here, continuing patrol...")

        self.logger.warning("="*60)
        self.logger.warning(f"✗ {target_ingredient} not found in any slot")
        self.logger.warning("="*60)
        return None

    def process_ingredients_list(self, ingredients: List[str]) -> bool:
        """
        Process a list of ingredients for a complete pizza order

        Args:
            ingredients: List of ingredient names to find and grab

        Returns:
            bool: True if all ingredients successfully grabbed
        """
        self.logger.info("\n" + "="*60)
        self.logger.info(f"STARTING ORDER - {len(ingredients)} INGREDIENTS")
        self.logger.info("="*60)
        self.logger.info(f"Ingredients: {', '.join(ingredients)}")
        self.logger.info("="*60)

        grabbed_ingredients = []

        for idx, ingredient in enumerate(ingredients, 1):
            display_name = get_ingredient_display_name(ingredient)
            self.logger.info(f"\n[{idx}/{len(ingredients)}] Processing: {display_name}")

            found_slot = self.patrol_and_find(ingredient)

            if found_slot:
                grabbed_ingredients.append(ingredient)
                self.logger.info(f"✓ Progress: {len(grabbed_ingredients)}/{len(ingredients)} ingredients grabbed")
            else:
                self.logger.error(f"\n✗ ORDER FAILED - Could not find {display_name}")
                self.logger.error(f"  Grabbed so far: {grabbed_ingredients}")
                return False

            # Brief pause between ingredients
            if idx < len(ingredients):
                self.logger.info("  Preparing for next ingredient...")
                time.sleep(1.0)

        # All ingredients grabbed successfully
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 ORDER COMPLETE - ALL INGREDIENTS GRABBED!")
        self.logger.info("="*60)
        self.logger.info(f"Grabbed: {', '.join([get_ingredient_display_name(i) for i in grabbed_ingredients])}")
        self.logger.info("="*60)

        return True

    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up RobotPatrolSystem...")
        # Don't release camera - it's managed by VisionSystem
        try:
            del self.arm
        except:
            pass
        self.logger.info("Done!")
