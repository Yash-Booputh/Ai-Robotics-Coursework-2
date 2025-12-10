"""
ChefMate Robot Assistant - Main Application
Main controller for the pizza robot assistant application
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    APP_NAME, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT, LOG_FILE, LOG_LEVEL,
    COLOR_BG_DARK, COLOR_TITLE_BAR, COLOR_TITLE_TEXT, COLOR_TITLE_SUBTITLE,
    COLOR_STATUS_BAR, COLOR_STATUS_TEXT, FONT_FAMILY
)
from robot import VisionSystem
from integrated_patrol_grab import IntegratedPatrolGrabSystem
from ui import (
    HomeScreen, MenuScreen, CartScreen, RobotScreen,
    FileUploadScreen, LiveCameraScreen
)


class ChefMateApp(tk.Tk):
    """
    Main application controller
    Manages all screens and coordinates between UI and robot systems
    """

    def __init__(self):
        """Initialize the application"""
        super().__init__()

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info(f"Starting {APP_NAME}")
        self.logger.info("=" * 60)

        # Window configuration
        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.configure(bg=COLOR_BG_DARK)

        # Start maximized (Linux-compatible)
        try:
            self.attributes('-zoomed', True)  # Linux
        except tk.TclError:
            try:
                self.state('zoomed')  # Windows
            except tk.TclError:
                pass  # If neither works, just use the geometry size

        # Configure styles
        self.configure_styles()

        # Initialize vision system (for UI display only)
        self.logger.info("Initializing vision system...")
        self.vision = VisionSystem()
        self.patrol_system = None  # Will be created when needed

        # Application state
        self.current_pizza_order = None
        self.current_screen_name = "Home"

        # Create title bar
        self.create_title_bar()

        # Create main container
        container = tk.Frame(self, bg=COLOR_BG_DARK)
        container.pack(fill=tk.BOTH, expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Dictionary to hold all screens
        self.frames = {}

        # Create all screens
        self.create_screens(container)

        # Create status bar
        self.create_status_bar()

        # Show home screen
        self.show_frame("HomeScreen")

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.logger.info("Application initialized successfully")

    def setup_logging(self):
        """Setup application logging"""
        # Create logs directory if needed
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def configure_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Dark frame style
        style.configure('Dark.TFrame', background=COLOR_BG_DARK)

    def create_title_bar(self):
        """Create modern title bar with app name and HOME button"""
        title_frame = tk.Frame(self, bg=COLOR_TITLE_BAR)
        title_frame.pack(fill=tk.X)

        title_container = tk.Frame(title_frame, bg=COLOR_TITLE_BAR)
        title_container.pack(side=tk.LEFT, padx=20, pady=15)

        title_label = tk.Label(
            title_container,
            text="ChefMate Robot Assistant",
            font=(FONT_FAMILY, 20, 'bold'),
            bg=COLOR_TITLE_BAR,
            fg=COLOR_TITLE_TEXT
        )
        title_label.pack(anchor=tk.W)

        # Mode subtitle label
        self.mode_subtitle_label = tk.Label(
            title_container,
            text="Home",
            font=(FONT_FAMILY, 10),
            bg=COLOR_TITLE_BAR,
            fg=COLOR_TITLE_SUBTITLE
        )
        self.mode_subtitle_label.pack(anchor=tk.W)

        # Home button in title bar
        home_btn = tk.Button(
            title_frame,
            text="HOME",
            command=self.go_home,
            font=(FONT_FAMILY, 10, 'bold'),
            bg='#0D47A1',
            fg=COLOR_TITLE_TEXT,
            relief=tk.FLAT,
            cursor='hand2',
            padx=15,
            pady=8
        )
        home_btn.pack(side=tk.RIGHT, padx=20, pady=10)

    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = tk.Frame(self, bg=COLOR_STATUS_BAR)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text="[READY] Welcome to ChefMate Robot Assistant",
            font=(FONT_FAMILY, 9),
            bg=COLOR_STATUS_BAR,
            fg=COLOR_STATUS_TEXT,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=15, pady=6)

    def update_status(self, message):
        """Update status bar message"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)

    def go_home(self):
        """Navigate to home screen"""
        self.show_frame("HomeScreen")

    def create_screens(self, container):
        """
        Create all application screens

        Args:
            container: Parent container widget
        """
        self.logger.info("Creating UI screens...")

        # List of all screen classes
        screens = [
            ("HomeScreen", HomeScreen),
            ("MenuScreen", MenuScreen),
            ("CartScreen", CartScreen),
            ("RobotScreen", RobotScreen),
            ("FileUploadScreen", FileUploadScreen),
            ("LiveCameraScreen", LiveCameraScreen)
        ]

        # Create each screen
        for screen_name, ScreenClass in screens:
            try:
                frame = ScreenClass(container, self)
                self.frames[screen_name] = frame
                frame.grid(row=0, column=0, sticky="nsew")
                self.logger.info(f"  [OK] {screen_name} created")
            except Exception as e:
                self.logger.error(f"  [ERROR] Failed to create {screen_name}: {e}")
                raise

    def show_frame(self, screen_name):
        """
        Show a specific screen

        Args:
            screen_name: Name of screen to show
        """
        if screen_name not in self.frames:
            self.logger.error(f"Screen not found: {screen_name}")
            return

        self.logger.info(f"Showing screen: {screen_name}")

        # Call on_hide for current screen if it has the method
        for frame in self.frames.values():
            if frame.winfo_viewable() and hasattr(frame, 'on_hide'):
                frame.on_hide()

        # Show requested screen
        frame = self.frames[screen_name]
        frame.tkraise()

        # Update subtitle based on screen
        screen_titles = {
            "HomeScreen": "Home",
            "MenuScreen": "Pizza Menu - Order Mode",
            "CartScreen": "Cart - Review Order",
            "RobotScreen": "Robot Execution",
            "FileUploadScreen": "File Upload - Detection Mode",
            "LiveCameraScreen": "Live Camera - Detection Mode"
        }

        if hasattr(self, 'mode_subtitle_label'):
            self.mode_subtitle_label.config(text=screen_titles.get(screen_name, "Unknown"))

        # Update status
        status_messages = {
            "HomeScreen": "[HOME] Select a mode to begin",
            "MenuScreen": "[MENU] Select a pizza to order",
            "CartScreen": "[CART] Review your order",
            "RobotScreen": "[ROBOT] Executing order...",
            "FileUploadScreen": "[FILE UPLOAD] Upload images for detection",
            "LiveCameraScreen": "[LIVE CAMERA] Real-time detection mode"
        }

        self.update_status(status_messages.get(screen_name, "[READY]"))

        # Call on_show if screen has the method
        if hasattr(frame, 'on_show'):
            frame.on_show()

        # Special handling for menu screen
        if screen_name == "MenuScreen":
            frame.reset_selection()

    # =========================================================================
    # PIZZA ORDER MANAGEMENT
    # =========================================================================

    def set_pizza_order(self, pizza_name):
        """
        Set the current pizza order

        Args:
            pizza_name: Name of ordered pizza
        """
        self.logger.info(f"Pizza order set: {pizza_name}")
        self.current_pizza_order = pizza_name

        # Update cart screen
        cart_screen = self.frames["CartScreen"]
        cart_screen.set_order(pizza_name)

    def start_robot_execution(self, pizza_name):
        """
        Start robot execution for an order

        Args:
            pizza_name: Name of pizza to make
        """
        self.logger.info(f"Starting robot execution for: {pizza_name}")

        # IntegratedPatrolGrabSystem will handle robot connection check
        # VisionSystem camera is only for display on RobotScreen

        # Show robot screen
        robot_screen = self.frames["RobotScreen"]
        self.show_frame("RobotScreen")

        # Start execution
        robot_screen.start_execution(pizza_name)

    def execute_pick_sequence(self, pizza_name, status_callback=None):
        """
        Execute the complete picking sequence using IntegratedPatrolGrabSystem

        Args:
            pizza_name: Name of pizza to make
            status_callback: Callback for status updates

        Returns:
            bool: True if successful
        """
        try:
            # Create patrol system if not already created
            if not self.patrol_system:
                self.logger.info("Initializing IntegratedPatrolGrabSystem...")
                if status_callback:
                    status_callback("🔧 Initializing patrol system...", "info")

                self.patrol_system = IntegratedPatrolGrabSystem()

                self.logger.info("✓ IntegratedPatrolGrabSystem initialized")
                if status_callback:
                    status_callback("✅ Patrol system ready", "success")

            # Check robot connection
            if not self.patrol_system.check_connection():
                self.logger.error("Robot not connected")
                if status_callback:
                    status_callback("❌ Robot not connected", "error")
                return False

            # Execute pizza order
            return self.patrol_system.execute_pizza_order(pizza_name, status_callback)

        except Exception as e:
            self.logger.error(f"Pick sequence error: {e}")
            if status_callback:
                status_callback(f"Error: {str(e)}", "error")
            return False

    def stop_pick_sequence(self):
        """Stop the current pick sequence"""
        if self.patrol_system:
            self.logger.warning("Emergency stop requested")
            self.patrol_system.move_to_home()

    # =========================================================================
    # CAMERA AND DETECTION
    # =========================================================================

    def start_vision_camera(self):
        """
        Start vision camera

        Returns:
            bool: True if started successfully
        """
        return self.vision.start_camera()

    def stop_vision_camera(self):
        """Stop vision camera"""
        self.vision.stop_camera()

    def get_camera_frame(self):
        """
        Get current camera frame with detection

        Returns:
            Tuple of (frame, detection)
        """
        return self.vision.capture_and_detect()

    def get_camera_fps(self):
        """
        Get camera FPS

        Returns:
            float: Current FPS
        """
        return self.vision.get_fps()

    def detect_in_image(self, image):
        """
        Run detection on a static image

        Args:
            image: Image as numpy array

        Returns:
            Tuple of (annotated_image, detection)
        """
        return self.vision.detect_ingredient(image)

    # =========================================================================
    # APPLICATION LIFECYCLE
    # =========================================================================

    def on_closing(self):
        """Handle application closing"""
        self.logger.info("Application closing...")

        # Ask for confirmation if robot might be running
        if self.patrol_system:
            response = messagebox.askyesno(
                "Confirm Exit",
                "Are you sure you want to exit?\n\nThis will stop any running operations.",
                icon="warning"
            )
            if not response:
                return

            # Stop robot
            self.logger.info("Stopping robot...")
            try:
                self.patrol_system.move_to_home()
            except:
                pass

        # Cleanup
        self.cleanup()

        # Close window
        self.destroy()

    def cleanup(self):
        """Cleanup resources before exit"""
        self.logger.info("Cleaning up resources...")

        try:
            # Cleanup patrol system
            if self.patrol_system:
                self.patrol_system.cleanup()

            # Stop camera
            if self.vision:
                self.vision.stop_camera()

            self.logger.info("Cleanup complete")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def run(self):
        """Run the application"""
        self.logger.info(f"{APP_NAME} started")
        self.mainloop()


def main():
    """Main entry point"""
    try:
        # Create and run application
        app = ChefMateApp()
        app.run()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Fatal error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()