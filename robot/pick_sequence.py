"""
ChefMate Robot Assistant - Pick Sequence
Orchestrates the complete ingredient picking sequence with vision verification
"""

import logging
import time
from typing import List, Optional, Callable

from .dofbot_controller import DofbotController
from .vision_system import VisionSystem
from config.recipes import get_pizza_ingredients, get_ingredient_display_name
from config.positions import get_all_slot_names


class PickSequence:
    """
    Manages the complete picking sequence for pizza ingredients
    Coordinates robot movements and vision verification
    """

    def __init__(self,
                 robot: DofbotController,
                 vision: VisionSystem,
                 status_callback: Optional[Callable] = None):
        """
        Initialize pick sequence manager

        Args:
            robot: DofbotController instance
            vision: VisionSystem instance
            status_callback: Optional callback for status updates
        """
        self.logger = logging.getLogger(__name__)
        self.robot = robot
        self.vision = vision
        self.status_callback = status_callback

        self.is_running = False
        self.current_pizza = None
        self.ingredients_needed = []
        self.ingredients_picked = []
        self.current_ingredient_index = 0
        self.total_ingredients = 0

    def _update_status(self, message: str, status_type: str = "info"):
        """
        Send status update via callback

        Args:
            message: Status message
            status_type: Type of status (info, success, error, warning)
        """
        self.logger.info(message)
        if self.status_callback:
            self.status_callback(message, status_type)

    def start_order(self, pizza_name: str) -> bool:
        """
        Start picking ingredients for a pizza order

        Args:
            pizza_name: Name of pizza to make

        Returns:
            bool: True if order started successfully
        """
        if self.is_running:
            self.logger.warning("Pick sequence already running")
            return False

        # Check systems
        if not self.robot.check_connection():
            self._update_status("❌ Robot not connected", "error")
            return False

        if not self.vision.is_camera_active:
            self._update_status("❌ Camera not active", "error")
            return False

        # Get ingredients needed
        self.ingredients_needed = get_pizza_ingredients(pizza_name)
        if not self.ingredients_needed:
            self._update_status(f"❌ Unknown pizza: {pizza_name}", "error")
            return False

        # Initialize order
        self.current_pizza = pizza_name
        self.ingredients_picked = []
        self.current_ingredient_index = 0
        self.total_ingredients = len(self.ingredients_needed)
        self.is_running = True

        self._update_status(f"🍕 Starting order: {pizza_name}", "info")
        self._update_status(f"Ingredients needed: {self.total_ingredients}", "info")

        # Move to home position
        self.robot.move_to_home()

        # Signal start
        self.robot.buzzer_beep(2)

        return True

    def scout_and_find_ingredient(self, ingredient_name: str) -> Optional[str]:
        """
        Scout all slots to find the specified ingredient

        Args:
            ingredient_name: Name of ingredient to find

        Returns:
            slot_name if found, None otherwise
        """
        display_name = get_ingredient_display_name(ingredient_name)
        self._update_status(f"🔍 Scouting for {display_name}...", "info")

        all_slots = get_all_slot_names()

        # First pass: Scout with top-down view
        for slot_name in all_slots:
            self._update_status(f"  Checking {slot_name} (top view)...", "info")

            # Move to scout position
            if not self.robot.scout_slot(slot_name, angle_mode="top"):
                self.logger.warning(f"Failed to move to {slot_name}")
                continue

            # Detect ingredient
            detected_ingredient = self.vision.detect_current_ingredient()

            if detected_ingredient and detected_ingredient == ingredient_name:
                self._update_status(f"✅ Found {display_name} at {slot_name}!", "success")
                return slot_name

        # Second pass: If not found, try 45° angle view
        self._update_status(f"  🔄 Trying angled view for better detection...", "info")

        for slot_name in all_slots:
            self._update_status(f"  Checking {slot_name} (angle view)...", "info")

            # Move to angled scout position
            if not self.robot.scout_slot(slot_name, angle_mode="angle"):
                self.logger.warning(f"Failed to move to {slot_name} (angle)")
                continue

            # Detect ingredient
            detected_ingredient = self.vision.detect_current_ingredient()

            if detected_ingredient and detected_ingredient == ingredient_name:
                self._update_status(f"✅ Found {display_name} at {slot_name} (angle view)!", "success")
                return slot_name

        # Not found
        self._update_status(f"❌ Could not find {display_name} in any slot", "error")
        return None

    def pick_next_ingredient(self) -> bool:
        """
        Pick the next ingredient in the sequence using scout mode

        Returns:
            bool: True if ingredient picked successfully
        """
        if not self.is_running:
            self.logger.error("Pick sequence not started")
            return False

        if self.current_ingredient_index >= self.total_ingredients:
            self.logger.info("All ingredients already picked")
            return False

        # Get next ingredient
        ingredient = self.ingredients_needed[self.current_ingredient_index]
        display_name = get_ingredient_display_name(ingredient)

        self._update_status(
            f"📦 Finding and picking {display_name} ({self.current_ingredient_index + 1}/{self.total_ingredients})",
            "info"
        )

        try:
            # Step 1: Scout all slots to find the ingredient
            found_slot = self.scout_and_find_ingredient(ingredient)

            if found_slot is None:
                self._update_status(
                    f"❌ {display_name} not found in any slot!",
                    "error"
                )
                self.is_running = False
                return False

            # Step 2: Grab the ingredient from the found slot
            self._update_status(f"🤖 Grabbing {display_name} from {found_slot}...", "info")
            success = self.robot.grab_from_slot(found_slot)

            if not success:
                self._update_status(f"❌ Failed to grab {display_name}", "error")
                return False

            self._update_status(f"✅ Grabbed {display_name}", "success")

            # Step 3: Deliver to basket
            self._update_status(f"🚚 Delivering {display_name}...", "info")
            success = self.robot.move_to_delivery()

            if not success:
                self._update_status(f"❌ Failed to deliver {display_name}", "error")
                return False

            self._update_status(f"✅ {display_name} delivered to basket", "success")

            # Step 4: Return to home
            self.robot.move_to_home()

            # Update progress
            self.ingredients_picked.append(ingredient)
            self.current_ingredient_index += 1

            # Small beep for success
            self.robot.buzzer_beep(1)

            return True

        except Exception as e:
            self.logger.error(f"Error picking {ingredient}: {e}")
            self._update_status(f"❌ Error: {str(e)}", "error")
            self.is_running = False
            return False

    def execute_full_order(self) -> bool:
        """
        Execute the complete order - pick all ingredients

        Returns:
            bool: True if order completed successfully
        """
        self._update_status("🚀 Executing full order...", "info")

        while self.current_ingredient_index < self.total_ingredients:
            success = self.pick_next_ingredient()

            if not success:
                self._update_status("❌ Order failed", "error")
                self.complete_order(success=False)
                return False

            # Brief pause between ingredients
            time.sleep(0.5)

        # Order complete
        self.complete_order(success=True)
        return True

    def complete_order(self, success: bool = True):
        """
        Complete the order and cleanup

        Args:
            success: Whether order was successful
        """
        self.is_running = False

        if success:
            self._update_status(
                f"🎉 Order complete! {self.current_pizza} is ready!",
                "success"
            )
            # Victory beeps
            self.robot.buzzer_beep(3)
            time.sleep(0.3)
            self.robot.buzzer_beep(3)
        else:
            self._update_status("❌ Order incomplete", "error")

        # Return to home
        self.robot.move_to_home()

        # Reset state
        self.current_pizza = None
        self.ingredients_needed = []
        self.ingredients_picked = []
        self.current_ingredient_index = 0
        self.total_ingredients = 0

    def cancel_order(self):
        """Cancel the current order"""
        if not self.is_running:
            return

        self._update_status("⚠️  Order cancelled", "warning")
        self.is_running = False

        # Release any held ingredient
        self.robot.gripper_open()

        # Return to home
        self.robot.move_to_home()

        # Reset state
        self.complete_order(success=False)

    def get_progress(self) -> dict:
        """
        Get current progress information

        Returns:
            Dictionary with progress information
        """
        return {
            'is_running': self.is_running,
            'pizza': self.current_pizza,
            'total_ingredients': self.total_ingredients,
            'picked_count': len(self.ingredients_picked),
            'current_index': self.current_ingredient_index,
            'ingredients_needed': self.ingredients_needed,
            'ingredients_picked': self.ingredients_picked,
            'progress_percent': (len(self.ingredients_picked) / self.total_ingredients * 100)
            if self.total_ingredients > 0 else 0
        }

    def emergency_stop(self):
        """Emergency stop for pick sequence"""
        self.logger.warning("🚨 Emergency stop activated")
        self.is_running = False
        self.robot.emergency_stop()
        self._update_status("🚨 EMERGENCY STOP", "error")