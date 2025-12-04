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
# UI THEME - MODERN STYLE (IRIS-inspired)
# ============================================================================

# Main colors - Modern Material Design inspired
COLOR_PRIMARY = "#2196F3"       # Blue (main accent)
COLOR_SECONDARY = "#FF9800"     # Orange (secondary)
COLOR_SUCCESS = "#4CAF50"       # Green (success)
COLOR_WARNING = "#FF9800"       # Orange (warning)
COLOR_DANGER = "#F44336"        # Red (danger)
COLOR_INFO = "#00BCD4"          # Cyan (info)
COLOR_PURPLE = "#9C27B0"        # Purple (video/special)

# Background colors - Softer, more pleasing palette
COLOR_BG_DARK = "#fafafa"       # Very light gray background (main)
COLOR_BG_LIGHT = "#FFFFFF"      # Pure white background (panels)
COLOR_BG_MEDIUM = "#f5f5f5"     # Soft gray (card backgrounds)

# Title bar colors
COLOR_TITLE_BAR = "#1976D2"     # Dark blue title bar
COLOR_TITLE_TEXT = "#FFFFFF"    # White text on title
COLOR_TITLE_SUBTITLE = "#B3E5FC"  # Light blue subtitle

# Status bar colors
COLOR_STATUS_BAR = "#333333"    # Dark status bar
COLOR_STATUS_TEXT = "#FFFFFF"   # White status text

# Text colors
COLOR_TEXT_LIGHT = "#FFFFFF"    # White text
COLOR_TEXT_DARK = "#333333"     # Dark text
COLOR_TEXT_GRAY = "#666666"     # Gray text
COLOR_TEXT_MUTED = "#999999"    # Muted text

# Button colors
BUTTON_PRIMARY = "#2196F3"      # Blue button
BUTTON_PRIMARY_HOVER = "#1976D2"
BUTTON_SECONDARY = "#FF9800"    # Orange button
BUTTON_SECONDARY_HOVER = "#F57C00"
BUTTON_SUCCESS = "#4CAF50"      # Green button
BUTTON_SUCCESS_HOVER = "#388E3C"
BUTTON_DANGER = "#F44336"       # Red button
BUTTON_DANGER_HOVER = "#D32F2F"
BUTTON_INFO = "#00BCD4"         # Cyan button
BUTTON_INFO_HOVER = "#0097A7"
BUTTON_PURPLE = "#9C27B0"       # Purple button
BUTTON_PURPLE_HOVER = "#7B1FA2"

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