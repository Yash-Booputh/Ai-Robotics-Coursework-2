"""
ChefMate Robot Assistant - Live Camera Screen
Real-time ingredient detection (IRIS-style modern design)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time

from .widgets import ModernButton
from config.settings import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_INFO,
    COLOR_BG_DARK, COLOR_BG_LIGHT, COLOR_BG_MEDIUM, COLOR_TEXT_DARK, COLOR_TEXT_GRAY,
    COLOR_TEXT_LIGHT, FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL
)


class LiveCameraScreen(ttk.Frame):
    """
    Live camera screen for real-time detection
    No robot control - just detection testing
    """

    def __init__(self, parent, controller):
        """
        Initialize live camera screen

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')

        self.camera_active = False
        self.camera_thread = None

        # Latest frame data (thread-safe access)
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Captured image for detection
        self.captured_frame = None
        self.detection_result = None

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets - modern IRIS style"""
        # Main container
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header = tk.Frame(main_frame, bg=COLOR_BG_LIGHT)
        header.pack(fill=tk.X, padx=0, pady=(0, 10))

        title = tk.Label(
            header,
            text="Live Camera Detection",
            font=(FONT_FAMILY, 18, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_INFO
        )
        title.pack(pady=12)

        # Use PanedWindow for responsive layout
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, bg=COLOR_BG_DARK,
                               sashwidth=5, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        # LEFT panel - LARGE Camera display (70% of space)
        left_panel = tk.Frame(paned, bg=COLOR_TEXT_GRAY)
        paned.add(left_panel, minsize=500, stretch="always")

        # Camera display
        self.camera_label = tk.Label(
            left_panel,
            text="Camera Feed\n\nClick 'Start Camera' to begin",
            font=(FONT_FAMILY, 14),
            bg=COLOR_TEXT_GRAY,
            fg=COLOR_TEXT_LIGHT
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # RIGHT panel - Compact Controls (30% of space)
        right_panel = tk.Frame(paned, bg=COLOR_BG_LIGHT)
        paned.add(right_panel, minsize=300, stretch="never")

        # Controls header
        controls_label = tk.Label(
            right_panel,
            text="Controls",
            font=(FONT_FAMILY, 13, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        controls_label.pack(pady=(10, 8))

        # Camera controls
        camera_controls = tk.Frame(right_panel, bg=COLOR_BG_LIGHT)
        camera_controls.pack(pady=5, padx=10, fill=tk.X)

        self.start_btn = ModernButton(
            camera_controls,
            text="START CAMERA",
            command=self.start_camera,
            bg=COLOR_SUCCESS
        )
        self.start_btn.pack(pady=3, fill=tk.X)

        self.capture_btn = ModernButton(
            camera_controls,
            text="CAPTURE & DETECT",
            command=self.capture_and_detect,
            bg=COLOR_INFO
        )
        self.capture_btn.pack(pady=3, fill=tk.X)
        self.capture_btn.config(state=tk.DISABLED)

        self.stop_btn = ModernButton(
            camera_controls,
            text="STOP CAMERA",
            command=self.stop_camera,
            bg=COLOR_DANGER
        )
        self.stop_btn.pack(pady=3, fill=tk.X)
        self.stop_btn.config(state=tk.DISABLED)

        # Separator
        tk.Frame(right_panel, bg=COLOR_BG_MEDIUM, height=1).pack(fill=tk.X, padx=15, pady=10)

        # Info section
        tk.Label(
            right_panel,
            text="Detection Info",
            font=(FONT_FAMILY, 11, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        ).pack(pady=(10, 5))

        # Current detection frame
        current_frame = tk.Frame(right_panel, bg=COLOR_BG_MEDIUM, relief=tk.SUNKEN, borderwidth=2)
        current_frame.pack(fill=tk.X, padx=15, pady=8)

        tk.Label(
            current_frame,
            text="Current Detection",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_DARK
        ).pack(pady=(8, 3))

        self.current_detection_label = tk.Label(
            current_frame,
            text="No detection",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, 'bold'),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_GRAY
        )
        self.current_detection_label.pack(pady=(3, 5))

        self.confidence_label = tk.Label(
            current_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_GRAY
        )
        self.confidence_label.pack(pady=(0, 8))

        # Model Status (NEW - shows if YOLO is loaded)
        model_status_frame = tk.Frame(right_panel, bg=COLOR_BG_LIGHT, relief=tk.SUNKEN, borderwidth=1)
        model_status_frame.pack(fill=tk.X, padx=15, pady=8)

        model_loaded = self.controller.vision.is_model_loaded
        model_text = "YOLO Model: LOADED" if model_loaded else "YOLO Model: NOT LOADED"
        model_color = COLOR_SUCCESS if model_loaded else COLOR_DANGER

        self.model_status_label = tk.Label(
            model_status_frame,
            text=model_text,
            font=(FONT_FAMILY, 9, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=model_color
        )
        self.model_status_label.pack(pady=5)

        # FPS and Status
        status_frame = tk.Frame(right_panel, bg=COLOR_BG_LIGHT)
        status_frame.pack(fill=tk.X, padx=15, pady=5)

        self.fps_label = tk.Label(
            status_frame,
            text="FPS: 0",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        )
        self.fps_label.pack(side=tk.LEFT)

        self.status_label = tk.Label(
            status_frame,
            text="● Camera Off",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        )
        self.status_label.pack(side=tk.RIGHT)

        # Detection history
        tk.Label(
            right_panel,
            text="Recent Detections",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        ).pack(pady=(15, 5))

        history_frame = tk.Frame(right_panel, bg=COLOR_BG_LIGHT)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        history_scrollbar = ttk.Scrollbar(history_frame)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_listbox = tk.Listbox(
            history_frame,
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK,
            font=(FONT_FAMILY, 9),
            yscrollcommand=history_scrollbar.set,
            selectmode=tk.SINGLE,
            borderwidth=1,
            relief=tk.SUNKEN
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.config(command=self.history_listbox.yview)

        # Info text
        info_text = tk.Label(
            right_panel,
            text="1. Start Camera\n2. Position ingredients\n3. Click Capture & Detect",
            font=(FONT_FAMILY, 9),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY,
            justify=tk.CENTER
        )
        info_text.pack(pady=(5, 10))

    def start_camera(self):
        """Start camera feed"""
        if self.camera_active:
            return

        try:
            # Start camera through controller
            success = self.controller.start_vision_camera()

            if not success:
                messagebox.showerror("Camera Error", "Failed to start camera")
                return

            self.camera_active = True
            self.start_btn.config(state=tk.DISABLED)
            self.capture_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.configure(text="● Camera Active", fg=COLOR_SUCCESS)

            # Clear any previous detection
            self.captured_frame = None
            self.detection_result = None

            # Start camera feed thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            # Start display update loop
            self.display_loop()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")

    def capture_and_detect(self):
        """Capture current frame and run detection"""
        if not self.camera_active:
            return

        try:
            # Get current frame
            with self.frame_lock:
                frame = self.latest_frame

            if frame is None:
                messagebox.showwarning("No Frame", "No camera frame available")
                return

            # Disable capture button while processing
            self.capture_btn.config(state=tk.DISABLED)
            self.status_label.configure(text="● Processing...", fg=COLOR_INFO)

            # Run detection on captured frame (in separate thread to not freeze UI)
            def detect_thread():
                try:
                    # Run detection with all ingredients
                    annotated_frame, detections = self.controller.vision.detect_all_ingredients(frame)

                    # Store results
                    self.captured_frame = annotated_frame
                    self.detection_result = detections

                    # Update UI in main thread
                    self.after(0, self.update_detection_results)

                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Detection Error", f"Failed to detect:\n{str(e)}"))
                    self.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
                    self.after(0, lambda: self.status_label.configure(text="● Camera Active", fg=COLOR_SUCCESS))

            threading.Thread(target=detect_thread, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture:\n{str(e)}")
            self.capture_btn.config(state=tk.NORMAL)

    def update_detection_results(self):
        """Update UI with detection results"""
        self.capture_btn.config(state=tk.NORMAL)
        self.status_label.configure(text="● Detection Complete", fg=COLOR_SUCCESS)

        if self.detection_result and len(self.detection_result) > 0:
            # Update detection info with first detection
            self.update_detection(self.detection_result[0])

            # Add all detections to history
            timestamp = time.strftime("%H:%M:%S")
            for det in self.detection_result:
                class_name = det['class_name']
                confidence = det['confidence']
                history_entry = f"[{timestamp}] {class_name} - {confidence:.1%}"
                self.history_listbox.insert(0, history_entry)

            # Limit history
            while self.history_listbox.size() > 20:
                self.history_listbox.delete(tk.END)

    def stop_camera(self):
        """Stop camera display and release the VisionSystem camera"""
        # Signal thread to stop
        self.camera_active = False

        # Wait for camera thread to stop
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
            self.camera_thread = None

        # Stop the shared VisionSystem camera
        self.controller.stop_vision_camera()

        # Update UI state
        self.start_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.configure(text="● Camera Off", fg=COLOR_TEXT_GRAY)

        # Reset display
        self.camera_label.configure(
            image="",
            text="Camera Off\n\nClick 'Start Camera' to begin"
        )
        self.camera_label.image = None
        self.current_detection_label.configure(text="No detection", fg=COLOR_TEXT_GRAY)
        self.confidence_label.configure(text="")
        self.fps_label.configure(text="FPS: 0")

    def camera_loop(self):
        """Simple camera loop - just capture frames"""
        while self.camera_active:
            try:
                # Read frame from camera
                frame = self.controller.vision.read_frame()

                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame

            except Exception as e:
                print(f"Camera loop error: {e}")
                time.sleep(0.1)

    def display_loop(self):
        """Update UI with camera feed or captured result"""
        def update_ui():
            if not self.camera_active:
                return

            try:
                # If we have a captured frame with detection, show that
                # Otherwise show live camera feed
                if self.captured_frame is not None:
                    display_frame = self.captured_frame
                else:
                    with self.frame_lock:
                        display_frame = self.latest_frame

                if display_frame is not None:
                    # Convert to RGB and display at higher resolution
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    # Display at full camera resolution (no resize needed if already 640x480)
                    # or scale up if needed for better quality
                    photo = ImageTk.PhotoImage(img)

                    # Update camera label
                    self.camera_label.configure(image=photo, text="")
                    self.camera_label.image = photo

                    # Update FPS
                    fps = self.controller.get_camera_fps()
                    self.fps_label.configure(text=f"FPS: {fps:.1f}")

            except Exception as e:
                print(f"Display loop error: {e}")

            # Schedule next update - 30ms for ~33 FPS
            if self.camera_active:
                self.after(30, update_ui)

        # Start the update loop
        update_ui()

    def update_detection(self, detection):
        """
        Update detection display

        Args:
            detection: Detection dictionary
        """
        class_name = detection['class_name']
        confidence = detection['confidence']

        # Update current detection
        self.current_detection_label.configure(
            text=class_name.replace("_", " ").title(),
            fg=COLOR_SUCCESS
        )

        # Confidence with color coding
        if confidence >= 0.8:
            conf_text = f"Confidence: {confidence:.1%} (Excellent)"
            conf_color = COLOR_SUCCESS
        elif confidence >= 0.6:
            conf_text = f"Confidence: {confidence:.1%} (Good)"
            conf_color = COLOR_SECONDARY
        else:
            conf_text = f"Confidence: {confidence:.1%} (Low)"
            conf_color = COLOR_DANGER

        self.confidence_label.configure(text=conf_text, fg=conf_color)

    def on_show(self):
        """Called when screen is shown"""
        pass

    def on_hide(self):
        """Called when screen is hidden"""
        # Stop the camera when leaving the screen
        if self.camera_active:
            self.stop_camera()
