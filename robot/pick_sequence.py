"""
ChefMate Robot Assistant - Pick Sequence
Orchestrates the complete ingredient picking sequence with vision verification
"""

import logging
import time
import sys
import os
from typing import List, Optional, Callable

# Add project root to path for integrated_patrol_grab import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .vision_system import VisionSystem
from config.recipes import get_pizza_ingredients, get_ingredient_display_name

# Import the integrated patrol and grab system (working ONNX version)
try:
    from integrated_patrol_grab import IntegratedPatrolGrabSystem
    PATROL_SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"IntegratedPatrolGrabSystem not available: {e}")
    PATROL_SYSTEM_AVAILABLE = False


class PickSequence:
    """
    Manages the complete picking sequence for pizza ingredients
    Coordinates robot movements and vision verification
    """

    def __init__(self,
                 vision: VisionSystem,
                 status_callback: Optional[Callable] = None,
                 audio_system = None):
        """
        Initialize pick sequence manager

        Args:
            vision: VisionSystem instance (only used for UI display)
            status_callback: Optional callback for status updates
            audio_system: Optional PizzaRobotAudio instance for sound effects
        """
        self.logger = logging.getLogger(__name__)
        self.vision = vision
        self.status_callback = status_callback
        self.audio = audio_system  # Store the audio system

        self.is_running = False
        self.current_pizza = None
        self.ingredients_needed = []
        self.ingredients_picked = []
        self.current_ingredient_index = 0
        self.total_ingredients = 0

        # Robot patrol system will be initialized lazily when needed
        self.patrol_system = None
        self.patrol_system_initialized = False

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

    def _initialize_patrol_system(self) -> bool:
        """
        Initialize the patrol system lazily (only when needed)

        Returns:
            bool: True if initialized successfully
        """
        if self.patrol_system_initialized:
            return self.patrol_system is not None

        if not PATROL_SYSTEM_AVAILABLE:
            self.logger.warning("RobotPatrolSystem not available")
            self.patrol_system_initialized = True
            return False

        try:
            self.logger.info("Initializing IntegratedPatrolGrabSystem...")
            self._update_status("🔧 Initializing patrol system...", "info")

            self.patrol_system = IntegratedPatrolGrabSystem()
            self.patrol_system_initialized = True

            self.logger.info("✓ IntegratedPatrolGrabSystem initialized")
            self._update_status("✅ Patrol system ready", "success")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize IntegratedPatrolGrabSystem: {e}")
            self._update_status(f"❌ Patrol system init failed: {e}", "error")
            self.patrol_system = None
            self.patrol_system_initialized = True
            return False

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

        # Initialize patrol system first (needed to check connection)
        if not self.patrol_system_initialized:
            if not self._initialize_patrol_system():
                self._update_status("❌ Failed to initialize patrol system", "error")
                return False

        # Check systems
        if not self.patrol_system.check_connection():
            self._update_status("❌ Robot not connected", "error")
            return False

        # Note: Camera check removed - IntegratedPatrolGrabSystem creates its own camera
        # VisionSystem camera is only used for display, not for order execution

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

        # Move to home position using IntegratedPatrolGrabSystem
        self.patrol_system.move_to_home()

        # Signal start
        self.patrol_system.buzzer_beep(2)

        return True

    def pick_next_ingredient(self) -> bool:
        """
        Pick the next ingredient in the sequence using integrated patrol and grab system

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
            f"Finding and picking {display_name} ({self.current_ingredient_index + 1}/{self.total_ingredients})",
            "info"
        )

        try:
            # Patrol system should already be initialized at start_order()
            # IntegratedPatrolGrabSystem is REQUIRED - no fallback
            if not self.patrol_system:
                self.logger.error("IntegratedPatrolGrabSystem not available - cannot continue")
                self._update_status(
                    "❌ Patrol system not available - order cannot proceed",
                    "error"
                )
                self.is_running = False
                return False

            # Use IntegratedPatrolGrabSystem for all robot control
            self.logger.info(f"Using IntegratedPatrolGrabSystem to find and grab {ingredient}")
            self._update_status(f"Robot patrolling slots to find {display_name}...", "info")

            # Call patrol_and_find which will automatically grab the ingredient when found
            # This method will log detailed progress internally
            found_slot = self.patrol_system.patrol_and_find(ingredient)

            if found_slot:
                self._update_status(f"{display_name} grabbed from {found_slot}", "success")

                if self.audio:
                    time.sleep(0.5)  # Small delay
                    self.audio.play_found_cube()

                # Update progress
                self.ingredients_picked.append(ingredient)
                self.current_ingredient_index += 1

                return True
            else:
                self._update_status(
                    f"{display_name} not found in any slot!",
                    "error"
                )
                self.is_running = False
                return False

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
            if self.patrol_system:
                self.patrol_system.buzzer_beep(3)
                time.sleep(0.3)
                self.patrol_system.buzzer_beep(3)
                if self.audio and self.current_pizza:
                    time.sleep(0.5)
                    self.audio.play_goodbye()

        else:
            self._update_status("❌ Order incomplete", "error")

        # Return to home
        if self.patrol_system:
            self.patrol_system.move_to_home()

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

        # Return to home (IntegratedPatrolGrabSystem handles gripper state)
        if self.patrol_system:
            self.patrol_system.move_to_home()

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
        if self.patrol_system:
            self.patrol_system.move_to_home()  # Safe position
        self._update_status("🚨 EMERGENCY STOP", "error")

    def cleanup(self):
        """Cleanup resources when done"""
        if self.patrol_system:
            try:
                self.logger.info("Cleaning up IntegratedPatrolGrabSystem...")
                self.patrol_system.cleanup()
                self.logger.info("✓ IntegratedPatrolGrabSystem cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up patrol system: {e}")