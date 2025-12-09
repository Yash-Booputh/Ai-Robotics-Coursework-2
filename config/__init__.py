"""
ChefMate Robot Assistant - Configuration Package
Easy imports for all configuration modules

NOTE: Robot positions have been moved to integrated_patrol_grab.py
"""

from .settings import *
from .recipes import *

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
]