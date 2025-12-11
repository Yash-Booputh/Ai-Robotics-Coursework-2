"""
ChefMate Robot Assistant - Application Settings
Global configuration for the application
"""

import os

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_NAME = "ChefMate Robot Assistant"
APP_VERSION = "1.0.0"
WINDOW_TITLE = "ChefMate - Pizza Robot Assistant"

# Window Configuration
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600
WINDOW_RESIZABLE = True

# ============================================================================
# PATHS
# ============================================================================

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

# Asset paths
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
PIZZA_IMAGES_DIR = os.path.join(ASSETS_DIR, "pizza_images")
INGREDIENT_IMAGES_DIR = os.path.join(ASSETS_DIR, "ingredient_images")
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")

# Log paths
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "chefmate.log")

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

CAMERA_ID = 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ============================================================================
# YOLO DETECTION SETTINGS
# ============================================================================

YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection (lowered for better detection)
YOLO_IMAGE_SIZE = 320  # Input size for YOLO
YOLO_MAX_DETECTIONS = 1  # Only detect top ingredient

# Ingredient class names (must match your trained model)
INGREDIENT_CLASSES = [
    'anchovies',
    'basil',
    'cheese',
    'chicken',
    'fresh_tomato',
    'shrimp'
]

# NOTE: Robot control settings have been moved to integrated_patrol_grab.py
# Only integrated_patrol_grab.py and JSON config files should define robot control parameters

# ============================================================================
# UI THEME - ITALIAN PIZZA RESTAURANT THEME
# ============================================================================

# Main colors - Italian Pizza Restaurant Palette
COLOR_PRIMARY = "#7F1100"       # Rich deep red (pizza sauce red)
COLOR_SECONDARY = "#BF9861"     # Warm tan/gold (crust/bread color)
COLOR_SUCCESS = "#02332D"       # Deep teal/forest green (basil/herbs)
COLOR_WARNING = "#BF9861"       # Warm tan (warning accent)
COLOR_DANGER = "#7F1100"        # Deep red (danger)
COLOR_INFO = "#02332D"          # Deep teal (info)
COLOR_PURPLE = "#BF9861"        # Warm gold (special features)

# Background colors - Cozy Italian restaurant feel
COLOR_BG_DARK = "#DACFBD"       # Warm beige/cream (main background)
COLOR_BG_LIGHT = "#DACFBD"      # Warm beige/cream (panels)
COLOR_BG_MEDIUM = "#BF9861"     # Warm tan/gold (card accents)

# Title bar colors
COLOR_TITLE_BAR = "#7F1100"     # Deep red title bar
COLOR_TITLE_TEXT = "#DACFBD"    # Cream text on title
COLOR_TITLE_SUBTITLE = "#BF9861"  # Gold subtitle

# Status bar colors
COLOR_STATUS_BAR = "#02332D"    # Deep teal status bar
COLOR_STATUS_TEXT = "#DACFBD"   # Cream status text

# Text colors
COLOR_TEXT_LIGHT = "#DACFBD"    # Cream text (on dark backgrounds)
COLOR_TEXT_DARK = "#7F1100"     # Deep red text (on light backgrounds)
COLOR_TEXT_GRAY = "#02332D"     # Deep teal text (secondary)
COLOR_TEXT_MUTED = "#BF9861"    # Warm gold (muted text)

# Button colors
BUTTON_PRIMARY = "#7F1100"      # Deep red button
BUTTON_PRIMARY_HOVER = "#5F0D00"
BUTTON_SECONDARY = "#BF9861"    # Warm gold button
BUTTON_SECONDARY_HOVER = "#A67E4D"
BUTTON_SUCCESS = "#02332D"      # Deep teal button
BUTTON_SUCCESS_HOVER = "#012620"
BUTTON_DANGER = "#7F1100"       # Deep red button
BUTTON_DANGER_HOVER = "#5F0D00"
BUTTON_INFO = "#02332D"         # Deep teal button
BUTTON_INFO_HOVER = "#012620"
BUTTON_PURPLE = "#BF9861"       # Warm gold button
BUTTON_PURPLE_HOVER = "#A67E4D"

# ============================================================================
# FONTS - Modern Style
# ============================================================================

FONT_FAMILY = "Segoe UI"        # Modern Windows font
FONT_SIZE_SMALL = 9
FONT_SIZE_NORMAL = 10
FONT_SIZE_LARGE = 14
FONT_SIZE_TITLE = 16
FONT_SIZE_HEADER = 20

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# MISC
# ============================================================================

# Maximum pizzas per order (limited by cube availability)
MAX_PIZZAS_PER_ORDER = 1

# Status messages
STATUS_IDLE = "Ready"
STATUS_BUSY = "Processing..."
STATUS_ERROR = "Error"
STATUS_SUCCESS = "Success"

# File upload supported formats
SUPPORTED_IMAGE_FORMATS = [
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
]