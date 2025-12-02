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

# Look-at position - For camera view
CAMERA_VIEW_POSITION = [90, 135, 20, 25, 90, 30]

# Delivery area - Where to drop the cubes
DELIVERY_POSITION = [90, 48, 35, 30, 270]

# Pre-grab position - Position above ingredients before descending
PRE_GRAB_POSITION = [90, 80, 50, 50, 270]

# ============================================================================
# INGREDIENT POSITIONS
# Based on 2x3 shelf layout:
#
#    [anchovies]  [shrimp]
#    [tomato]     [cheese]
#    [basil]      [chicken]
#
# NOTE: These are PLACEHOLDER values!
# You MUST calibrate these positions for your actual setup.
# ============================================================================

INGREDIENT_POSITIONS = {
    "anchovies": {
        "approach": [65, 100, 60, 40, 270],  # Position above cube
        "grab": [65, 22, 64, 56, 270],  # Position to grab cube
        "lift": [65, 80, 50, 50, 270],  # Lift cube up
        "shelf_location": "top_left",
        "description": "Top left position"
    },

    "shrimp": {
        "approach": [115, 100, 60, 40, 270],
        "grab": [115, 22, 64, 56, 270],
        "lift": [115, 80, 50, 50, 270],
        "shelf_location": "top_right",
        "description": "Top right position"
    },

    "fresh_tomato": {
        "approach": [70, 90, 55, 45, 270],
        "grab": [70, 19, 66, 56, 270],
        "lift": [70, 80, 50, 50, 270],
        "shelf_location": "middle_left",
        "description": "Middle left position"
    },

    "cheese": {
        "approach": [110, 90, 55, 45, 270],
        "grab": [110, 19, 66, 56, 270],
        "lift": [110, 80, 50, 50, 270],
        "shelf_location": "middle_right",
        "description": "Middle right position"
    },

    "basil": {
        "approach": [60, 85, 50, 50, 270],
        "grab": [60, 66, 20, 29, 270],
        "lift": [60, 80, 50, 50, 270],
        "shelf_location": "bottom_left",
        "description": "Bottom left position"
    },

    "chicken": {
        "approach": [120, 85, 50, 50, 270],
        "grab": [120, 66, 20, 28, 270],
        "lift": [120, 80, 50, 50, 270],
        "shelf_location": "bottom_right",
        "description": "Bottom right position"
    }
}

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

def get_ingredient_position(ingredient_name, position_type="grab"):
    """
    Get servo position for a specific ingredient

    Args:
        ingredient_name: Name of ingredient (e.g., "cheese")
        position_type: Type of position - "approach", "grab", or "lift"

    Returns:
        List of servo angles or None if not found
    """
    ingredient = INGREDIENT_POSITIONS.get(ingredient_name)
    if ingredient:
        return ingredient.get(position_type)
    return None


def get_all_ingredient_names():
    """Get list of all configured ingredients"""
    return list(INGREDIENT_POSITIONS.keys())


def validate_ingredient_position(ingredient_name):
    """Check if ingredient has configured positions"""
    return ingredient_name in INGREDIENT_POSITIONS


def get_shelf_layout():
    """Get visual representation of shelf layout"""
    layout = {}
    for ingredient, data in INGREDIENT_POSITIONS.items():
        location = data["shelf_location"]
        layout[location] = ingredient
    return layout


def print_calibration_status():
    """Print calibration status for all ingredients"""
    print("\n" + "=" * 60)
    print("INGREDIENT POSITION CALIBRATION STATUS")
    print("=" * 60)
    for ingredient, data in INGREDIENT_POSITIONS.items():
        print(f"\n{ingredient.upper()}:")
        print(f"  Location: {data['shelf_location']}")
        print(f"  Approach: {data['approach']}")
        print(f"  Grab:     {data['grab']}")
        print(f"  Lift:     {data['lift']}")
    print("\n" + "=" * 60)
    print("⚠️  REMINDER: These are PLACEHOLDER values!")
    print("    Run calibration before operating robot!")
    print("=" * 60 + "\n")