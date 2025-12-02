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

CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ============================================================================
# YOLO DETECTION SETTINGS
# ============================================================================

YOLO_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for detection
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

# ============================================================================
# ROBOT SETTINGS
# ============================================================================

# Movement speeds (milliseconds)
ROBOT_SPEED_FAST = 500
ROBOT_SPEED_NORMAL = 1000
ROBOT_SPEED_SLOW = 1500

# Gripper settings
GRIPPER_SERVO_ID = 6
GRIPPER_OPEN_ANGLE = 60
GRIPPER_CLOSE_ANGLE = 135
GRIPPER_DELAY = 0.5  # seconds to wait after gripper action

# Safety settings
MAX_RETRIES = 1  # Number of retries for failed operations
OPERATION_TIMEOUT = 30  # seconds

# ============================================================================
# UI THEME - PIZZA COLORS
# ============================================================================

# Main colors
COLOR_PRIMARY = "#E74C3C"      # Pizza Red
COLOR_SECONDARY = "#F39C12"     # Pizza Yellow/Orange
COLOR_SUCCESS = "#27AE60"       # Green (basil)
COLOR_WARNING = "#F39C12"       # Orange
COLOR_DANGER = "#E74C3C"        # Red
COLOR_INFO = "#3498DB"          # Blue

# Background colors
COLOR_BG_DARK = "#2C3E50"       # Dark background
COLOR_BG_LIGHT = "#ECF0F1"      # Light background
COLOR_BG_MEDIUM = "#34495E"     # Medium background

# Text colors
COLOR_TEXT_LIGHT = "#FFFFFF"    # White text
COLOR_TEXT_DARK = "#2C3E50"     # Dark text
COLOR_TEXT_GRAY = "#7F8C8D"     # Gray text

# Button colors
BUTTON_PRIMARY = "#E74C3C"      # Red button
BUTTON_PRIMARY_HOVER = "#C0392B"
BUTTON_SECONDARY = "#F39C12"    # Orange button
BUTTON_SECONDARY_HOVER = "#D68910"
BUTTON_SUCCESS = "#27AE60"      # Green button
BUTTON_SUCCESS_HOVER = "#229954"
BUTTON_DANGER = "#E74C3C"       # Red button
BUTTON_DANGER_HOVER = "#C0392B"

# ============================================================================
# FONTS
# ============================================================================

FONT_FAMILY = "Arial"
FONT_SIZE_SMALL = 10
FONT_SIZE_NORMAL = 12
FONT_SIZE_LARGE = 14
FONT_SIZE_TITLE = 18
FONT_SIZE_HEADER = 24

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