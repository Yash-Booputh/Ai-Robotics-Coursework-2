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
    print("⚠️  Warning: Arm_Lib not found. Robot control will be simulated.")

from config.positions import (
    HOME_POSITION, DELIVERY_POSITION,
    GRIPPER_POSITIONS, MOVEMENT_SPEEDS, SLOT_POSITIONS,
    get_slot_position, get_all_slot_names
)
from config.settings import GRIPPER_DELAY, MAX_RETRIES


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

                # Move to home position
                self.move_to_home()
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
                self.move_to_home()
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
        """Move robot to home/rest position"""
        self.logger.info("Moving to home position...")
        self.move_servos_array(HOME_POSITION, MOVEMENT_SPEEDS["normal"])
        self.gripper_open()
        self.is_busy = False
        self.logger.info("✅ At home position")

    def gripper_open(self):
        """Open the gripper"""
        if self.gripper_state == "open":
            return  # Already open

        self.logger.info("Opening gripper...")
        angle = GRIPPER_POSITIONS["open"]
        servo_id = GRIPPER_POSITIONS["servo_id"]

        if self.check_connection():
            self.arm.Arm_serial_servo_write(servo_id, angle, 400)
        else:
            self.logger.warning(f"Simulating: Gripper open to {angle}°")

        time.sleep(GRIPPER_DELAY)
        self.gripper_state = "open"
        self.logger.info("✅ Gripper opened")

    def gripper_close(self):
        """Close the gripper"""
        if self.gripper_state == "closed":
            return  # Already closed

        self.logger.info("Closing gripper...")
        angle = GRIPPER_POSITIONS["closed"]
        servo_id = GRIPPER_POSITIONS["servo_id"]

        if self.check_connection():
            self.arm.Arm_serial_servo_write(servo_id, angle, 400)
        else:
            self.logger.warning(f"Simulating: Gripper close to {angle}°")

        time.sleep(GRIPPER_DELAY)
        self.gripper_state = "closed"
        self.logger.info("✅ Gripper closed")

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
        Move robot to scout position for a specific slot

        Args:
            slot_name: Name of slot (e.g., "slot_1")
            angle_mode: "top" for top-down view, "angle" for 45° view

        Returns:
            bool: True if movement successful
        """
        position_type = "scout_top" if angle_mode == "top" else "scout_angle"
        position = get_slot_position(slot_name, position_type)

        if position is None:
            self.logger.error(f"Scout position not found for {slot_name} ({angle_mode})")
            return False

        try:
            self.logger.info(f"Scouting {slot_name} ({angle_mode} view)...")
            self.move_servos(position, MOVEMENT_SPEEDS["normal"])
            # Give camera time to stabilize
            time.sleep(0.3)
            return True
        except Exception as e:
            self.logger.error(f"Error scouting {slot_name}: {e}")
            return False

    def grab_from_slot(self, slot_name: str) -> bool:
        """
        Grab cube from a specific slot (approach -> grab -> lift sequence)

        Args:
            slot_name: Name of slot where cube is located

        Returns:
            bool: True if grab successful
        """
        self.logger.info(f"Grabbing cube from {slot_name}")
        self.is_busy = True

        try:
            # Get positions for this slot
            approach_pos = get_slot_position(slot_name, "approach")
            grab_pos = get_slot_position(slot_name, "grab")
            lift_pos = get_slot_position(slot_name, "lift")

            if not all([approach_pos, grab_pos, lift_pos]):
                self.logger.error(f"Positions not defined for {slot_name}")
                return False

            # Ensure gripper is open
            self.gripper_open()

            # 1. Move to approach position (above cube)
            self.logger.info(f"  -> Approaching {slot_name}...")
            self.move_servos(approach_pos, MOVEMENT_SPEEDS["normal"])

            # 2. Move down to grab position
            self.logger.info(f"  -> Descending to grab...")
            self.move_servos(grab_pos, MOVEMENT_SPEEDS["slow"])

            # 3. Close gripper
            self.gripper_close()

            # 4. Lift up
            self.logger.info(f"  -> Lifting cube...")
            self.move_servos(lift_pos, MOVEMENT_SPEEDS["normal"])

            self.logger.info(f"✅ Successfully grabbed cube from {slot_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error grabbing from {slot_name}: {e}")
            return False
        finally:
            self.is_busy = False

    def move_to_delivery(self) -> bool:
        """
        Move to delivery area and release cube

        Returns:
            bool: True if delivery successful
        """
        self.logger.info("Moving to delivery area...")
        self.is_busy = True

        try:
            # Move to delivery position
            self.move_servos(DELIVERY_POSITION, MOVEMENT_SPEEDS["normal"])

            # Release cube
            self.logger.info("Releasing cube...")
            self.gripper_open()

            self.logger.info("✅ Cube delivered")
            return True

        except Exception as e:
            self.logger.error(f"Error during delivery: {e}")
            return False
        finally:
            self.is_busy = False

    def emergency_stop(self):
        """Emergency stop - move to safe position"""
        self.logger.warning("🚨 EMERGENCY STOP")
        self.is_busy = False
        try:
            if self.check_connection():
                # Release gripper
                self.gripper_open()
                # Move to home slowly
                self.move_servos_array(HOME_POSITION, MOVEMENT_SPEEDS["slow"])
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")

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