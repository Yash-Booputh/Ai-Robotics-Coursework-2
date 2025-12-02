"""
ChefMate Robot Assistant - Configuration Package
Easy imports for all configuration modules
"""

from .settings import *
from .recipes import *
from .positions import *

__all__ = [
    # Settings
    'APP_NAME', 'APP_VERSION', 'WINDOW_TITLE',
    'WINDOW_WIDTH', 'WINDOW_HEIGHT',
    'CAMERA_ID', 'CAMERA_WIDTH', 'CAMERA_HEIGHT',
    'YOLO_MODEL_PATH', 'YOLO_CONFIDENCE_THRESHOLD',
    'COLOR_PRIMARY', 'COLOR_SECONDARY', 'COLOR_SUCCESS',

    # Recipes
    'PIZZA_RECIPES', 'INGREDIENT_INFO',
    'get_pizza_list', 'get_pizza_info', 'get_pizza_ingredients',

    # Positions - Scout Mode
    'HOME_POSITION', 'DELIVERY_POSITION', 'SLOT_POSITIONS',
    'INGREDIENT_NAMES',
    'get_slot_position', 'get_all_slot_names', 'validate_slot_name',
    'get_all_ingredient_names'
]