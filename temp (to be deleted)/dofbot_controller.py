"""
ChefMate Robot Assistant - Dofbot Controller
Controls the Yahboom Dofbot robotic arm
"""

import time
import logging
from typing import List, Optional

try:
    from Arm_Lib import Arm_Device
    ARM_LIB_AVAILABLE = True
except ImportError:
    ARM_LIB_AVAILABLE = False
    # Suppress warning - simulation mode is fine for UI testing
    # Uncomment line below to see warning:
    # print("⚠️  Warning: Arm_Lib not found. Robot control will be simulated.")

# NOTE: This controller is only used for minimal utility functions:
# - check_connection() - Check if robot is connected
# - buzzer_beep() - Audio feedback
#
# ALL robot control (movement, positions, grabbing) is handled by IntegratedPatrolGrabSystem


class DofbotController:
    """
    Controller for Yahboom Dofbot robotic arm
    Handles all robot movements, gripper control, and safety checks
    """

    def __init__(self):
        """Initialize the robot controller"""
        self.logger = logging.getLogger(__name__)
        self.arm = None
        self.is_connected = False
        self.is_busy = False
        self.current_position = None
        self.gripper_state = "open"  # "open" or "closed"

        # Initialize robot
        self.connect()

    def connect(self) -> bool:
        """
        Connect to the robot

        Returns:
            bool: True if connected successfully
        """
        try:
            if ARM_LIB_AVAILABLE:
                self.logger.info("Connecting to Dofbot...")
                self.arm = Arm_Device()
                time.sleep(0.1)
                self.is_connected = True
                self.logger.info("✅ Dofbot connected successfully")
                return True
            else:
                self.logger.warning("Arm_Lib not available - running in simulation mode")
                self.is_connected = False
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect to robot: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from the robot"""
        try:
            if self.is_connected and self.arm:
                self.logger.info("Disconnecting from Dofbot...")
                self.arm = None
                self.is_connected = False
                self.logger.info("Dofbot disconnected")
        except Exception as e:
            self.logger.error(f"Error disconnecting robot: {e}")

    def check_connection(self) -> bool:
        """
        Check if robot is connected

        Returns:
            bool: True if connected
        """
        return self.is_connected and self.arm is not None

    def move_servo(self, servo_id: int, angle: int, speed: int = 1000):
        """
        Move a single servo to specified angle

        Args:
            servo_id: Servo number (1-6)
            angle: Target angle (0-180 or 0-270 for servo 5)
            speed: Movement speed in milliseconds
        """
        if not self.check_connection():
            self.logger.warning(f"Simulating: Servo {servo_id} -> {angle}° at {speed}ms")
            return

        try:
            self.arm.Arm_serial_servo_write(servo_id, angle, speed)
            time.sleep(0.01)
        except Exception as e:
            self.logger.error(f"Error moving servo {servo_id}: {e}")
            raise

    def move_servos(self, positions: List[int], speed: int = 1000):
        """
        Move multiple servos simultaneously

        Args:
            positions: List of 5 or 6 servo angles [S1, S2, S3, S4, S5, (S6)]
            speed: Movement speed in milliseconds
        """
        if not self.check_connection():
            self.logger.warning(f"Simulating: Moving to {positions} at {speed}ms")
            time.sleep(speed / 1000.0)
            return

        try:
            # Move servos 1-5 (or 1-6)
            for i in range(len(positions)):
                servo_id = i + 1
                self.move_servo(servo_id, positions[i], speed)

            # Wait for movement to complete
            time.sleep(speed / 1000.0)
            self.current_position = positions.copy()

        except Exception as e:
            self.logger.error(f"Error moving servos: {e}")
            raise

    def move_servos_array(self, positions: List[int], speed: int = 1000):
        """
        Move all 6 servos using array method (faster)

        Args:
            positions: List of 6 servo angles [S1, S2, S3, S4, S5, S6]
            speed: Movement speed in milliseconds
        """
        if not self.check_connection():
            self.logger.warning(f"Simulating: Array move to {positions} at {speed}ms")
            time.sleep(speed / 1000.0)
            return

        try:
            self.arm.Arm_serial_servo_write6_array(positions, speed)
            time.sleep(speed / 1000.0)
            self.current_position = positions.copy()
        except Exception as e:
            self.logger.error(f"Error in array move: {e}")
            raise

    def move_to_home(self):
        """
        DEPRECATED: This method is not used. IntegratedPatrolGrabSystem handles home position.
        """
        self.logger.error("move_to_home() is deprecated. Use IntegratedPatrolGrabSystem.move_to_home() instead.")
        return False

    def gripper_open(self):
        """
        DEPRECATED: This method is not used. IntegratedPatrolGrabSystem handles gripper control.
        """
        self.logger.error("gripper_open() is deprecated. Use IntegratedPatrolGrabSystem instead.")
        return False

    def gripper_close(self):
        """
        DEPRECATED: This method is not used. IntegratedPatrolGrabSystem handles gripper control.
        """
        self.logger.error("gripper_close() is deprecated. Use IntegratedPatrolGrabSystem instead.")
        return False

    def buzzer_beep(self, duration: int = 1):
        """
        Sound the buzzer

        Args:
            duration: Beep duration (1-255, each unit is ~0.1s)
        """
        if self.check_connection():
            self.arm.Arm_Buzzer_On(duration)
        else:
            self.logger.warning(f"Simulating: Buzzer beep for {duration * 0.1}s")

    def scout_slot(self, slot_name: str, angle_mode: str = "top") -> bool:
        """
        DEPRECATED: This method is not used. IntegratedPatrolGrabSystem handles all scouting.

        Move robot to scout position for a specific slot

        Args:
            slot_name: Name of slot (e.g., "slot_1")
            angle_mode: "top" for top-down view, "angle" for 45° view

        Returns:
            bool: True if movement successful
        """
        self.logger.error("scout_slot() is deprecated. Use IntegratedPatrolGrabSystem instead.")
        return False

    def grab_from_slot(self, slot_name: str) -> bool:
        """
        DEPRECATED: This method is not used. IntegratedPatrolGrabSystem handles all grabbing.

        Grab cube from a specific slot (approach -> grab -> lift sequence)

        Args:
            slot_name: Name of slot where cube is located

        Returns:
            bool: True if grab successful
        """
        self.logger.error("grab_from_slot() is deprecated. Use IntegratedPatrolGrabSystem instead.")
        return False

    def move_to_delivery(self) -> bool:
        """
        DEPRECATED: This method is not used. IntegratedPatrolGrabSystem handles all delivery.

        Move to delivery area and release cube

        Returns:
            bool: True if delivery successful
        """
        self.logger.error("move_to_delivery() is deprecated. Use IntegratedPatrolGrabSystem instead.")
        return False

    def emergency_stop(self):
        """
        DEPRECATED: This method is not used. IntegratedPatrolGrabSystem handles emergency stop.
        """
        self.logger.error("emergency_stop() is deprecated. Use IntegratedPatrolGrabSystem instead.")
        return False

    def test_movement(self):
        """Test basic robot movements"""
        self.logger.info("Starting movement test...")

        try:
            # Test home position
            self.move_to_home()
            time.sleep(1)

            # Test gripper
            self.logger.info("Testing gripper...")
            self.gripper_close()
            time.sleep(1)
            self.gripper_open()
            time.sleep(1)

            # Test buzzer
            self.logger.info("Testing buzzer...")
            self.buzzer_beep(3)

            self.logger.info("✅ Movement test complete")
            return True

        except Exception as e:
            self.logger.error(f"Movement test failed: {e}")
            return False

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.disconnect()