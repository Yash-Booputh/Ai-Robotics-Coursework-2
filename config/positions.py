"""
ChefMate Robot Assistant - Robot Positions
Predefined servo positions for ingredients, delivery area, and home position
"""

# ============================================================================
# SERVO POSITION DEFINITIONS
# Format: [Servo1, Servo2, Servo3, Servo4, Servo5, Servo6]
# Servo angles range: 0-180 degrees
# ============================================================================

# Home/Rest position - Robot waiting position
HOME_POSITION = [90, 164, 18, 0, 90, 90]

# Delivery area - Where to drop the cubes
DELIVERY_POSITION = [90, 48, 35, 30, 270]

# ============================================================================
# SLOT POSITIONS (FIXED POSITIONS FOR SCOUTING)
# Based on 2x3 shelf layout - these are FIXED slot positions where cubes
# can be randomly placed. The robot will scout each slot to find ingredients.
#
# Layout:
#    [Slot 1]  [Slot 2]
#    [Slot 3]  [Slot 4]
#    [Slot 5]  [Slot 6]
#
# NOTE: These are PLACEHOLDER values!
# You MUST calibrate these positions for your actual setup.
# ============================================================================

SLOT_POSITIONS = {
    "slot_1": {
        "scout_top": [65, 100, 60, 40, 270],  # Top-down view for detection
        "scout_angle": [65, 110, 50, 50, 270],  # 45° angle view (backup)
        "approach": [65, 100, 60, 40, 270],  # Position above cube
        "grab": [65, 22, 64, 56, 270],  # Position to grab cube
        "lift": [65, 80, 50, 50, 270],  # Lift cube up
        "location": "top_left",
        "description": "Top left slot"
    },

    "slot_2": {
        "scout_top": [115, 100, 60, 40, 270],
        "scout_angle": [115, 110, 50, 50, 270],
        "approach": [115, 100, 60, 40, 270],
        "grab": [115, 22, 64, 56, 270],
        "lift": [115, 80, 50, 50, 270],
        "location": "top_right",
        "description": "Top right slot"
    },

    "slot_3": {
        "scout_top": [70, 90, 55, 45, 270],
        "scout_angle": [70, 100, 45, 55, 270],
        "approach": [70, 90, 55, 45, 270],
        "grab": [70, 19, 66, 56, 270],
        "lift": [70, 80, 50, 50, 270],
        "location": "middle_left",
        "description": "Middle left slot"
    },

    "slot_4": {
        "scout_top": [110, 90, 55, 45, 270],
        "scout_angle": [110, 100, 45, 55, 270],
        "approach": [110, 90, 55, 45, 270],
        "grab": [110, 19, 66, 56, 270],
        "lift": [110, 80, 50, 50, 270],
        "location": "middle_right",
        "description": "Middle right slot"
    },

    "slot_5": {
        "scout_top": [60, 85, 50, 50, 270],
        "scout_angle": [60, 95, 40, 60, 270],
        "approach": [60, 85, 50, 50, 270],
        "grab": [60, 66, 20, 29, 270],
        "lift": [60, 80, 50, 50, 270],
        "location": "bottom_left",
        "description": "Bottom left slot"
    },

    "slot_6": {
        "scout_top": [120, 85, 50, 50, 270],
        "scout_angle": [120, 95, 40, 60, 270],
        "approach": [120, 85, 50, 50, 270],
        "grab": [120, 66, 20, 28, 270],
        "lift": [120, 80, 50, 50, 270],
        "location": "bottom_right",
        "description": "Bottom right slot"
    }
}

# Ingredient names (for reference - must match YOLO model classes)
INGREDIENT_NAMES = [
    "anchovies",
    "basil",
    "cheese",
    "chicken",
    "fresh_tomato",
    "shrimp"
]

# ============================================================================
# GRIPPER POSITIONS
# ============================================================================

GRIPPER_POSITIONS = {
    "open": 60,  # Gripper open angle
    "closed": 135,  # Gripper closed angle
    "servo_id": 6  # Gripper servo number
}

# ============================================================================
# MOVEMENT PARAMETERS
# ============================================================================

MOVEMENT_SPEEDS = {
    "very_fast": 300,
    "fast": 500,
    "normal": 1000,
    "slow": 1500,
    "very_slow": 2000
}

# ============================================================================
# CALIBRATION GUIDE
# ============================================================================

CALIBRATION_INSTRUCTIONS = """
CALIBRATION INSTRUCTIONS FOR INGREDIENT POSITIONS:

1. Place all cubes in their designated shelf positions
2. Run the calibration script (tools/calibrate_positions.py)
3. For each ingredient:
   a. Manually move robot to "approach" position (above cube)
   b. Record servo angles
   c. Move to "grab" position (touching cube)
   d. Record servo angles
   e. Move to "lift" position (cube lifted)
   f. Record servo angles
4. Update the INGREDIENT_POSITIONS dictionary with recorded values
5. Test each position before running full program

SAFETY:
- Always start with gripper OPEN
- Move slowly during calibration
- Keep emergency stop accessible
- Never run program with uncalibrated positions
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_slot_position(slot_name, position_type="scout_top"):
    """
    Get servo position for a specific slot

    Args:
        slot_name: Name of slot (e.g., "slot_1", "slot_2")
        position_type: Type of position - "scout_top", "scout_angle", "approach", "grab", or "lift"

    Returns:
        List of servo angles or None if not found
    """
    slot = SLOT_POSITIONS.get(slot_name)
    if slot:
        return slot.get(position_type)
    return None


def get_all_slot_names():
    """Get list of all slot names"""
    return list(SLOT_POSITIONS.keys())


def get_all_ingredient_names():
    """Get list of all ingredient names"""
    return INGREDIENT_NAMES


def validate_slot_name(slot_name):
    """Check if slot name is valid"""
    return slot_name in SLOT_POSITIONS


def get_shelf_layout():
    """Get visual representation of shelf layout"""
    layout = {}
    for slot_name, data in SLOT_POSITIONS.items():
        location = data["location"]
        layout[location] = slot_name
    return layout


def print_calibration_status():
    """Print calibration status for all slots"""
    print("\n" + "=" * 60)
    print("SLOT POSITION CALIBRATION STATUS")
    print("=" * 60)
    for slot_name, data in SLOT_POSITIONS.items():
        print(f"\n{slot_name.upper()}:")
        print(f"  Location:    {data['location']}")
        print(f"  Scout Top:   {data['scout_top']}")
        print(f"  Scout Angle: {data['scout_angle']}")
        print(f"  Approach:    {data['approach']}")
        print(f"  Grab:        {data['grab']}")
        print(f"  Lift:        {data['lift']}")
    print("\n" + "=" * 60)
    print("⚠️  REMINDER: These are PLACEHOLDER values!")
    print("    Run calibration before operating robot!")
    print("=" * 60 + "\n")