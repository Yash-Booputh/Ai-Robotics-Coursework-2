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
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_INFO,
    COLOR_BG_DARK, COLOR_BG_LIGHT, COLOR_BG_MEDIUM, COLOR_TEXT_DARK, COLOR_TEXT_GRAY,
    FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL
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
        self.detection_history = []
        self.max_history = 10

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
        left_panel = tk.Frame(paned, bg='#1a1a1a')
        paned.add(left_panel, minsize=500, stretch="always")

        # Camera display
        self.camera_label = tk.Label(
            left_panel,
            text="Camera Feed\n\nClick 'Start Camera' to begin",
            font=(FONT_FAMILY, 14),
            bg='#1a1a1a',
            fg='white'
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
        current_frame = tk.Frame(right_panel, bg='#e8f5e9', relief=tk.SUNKEN, borderwidth=2)
        current_frame.pack(fill=tk.X, padx=15, pady=8)

        tk.Label(
            current_frame,
            text="Current Detection",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg='#e8f5e9',
            fg=COLOR_TEXT_DARK
        ).pack(pady=(8, 3))

        self.current_detection_label = tk.Label(
            current_frame,
            text="No detection",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, 'bold'),
            bg='#e8f5e9',
            fg=COLOR_TEXT_GRAY
        )
        self.current_detection_label.pack(pady=(3, 5))

        self.confidence_label = tk.Label(
            current_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg='#e8f5e9',
            fg=COLOR_TEXT_GRAY
        )
        self.confidence_label.pack(pady=(0, 8))

        # Model Status (NEW - shows if YOLO is loaded)
        model_status_frame = tk.Frame(right_panel, bg='#e3f2fd', relief=tk.SUNKEN, borderwidth=1)
        model_status_frame.pack(fill=tk.X, padx=15, pady=8)

        model_loaded = self.controller.vision.is_model_loaded
        model_text = "YOLO Model: LOADED" if model_loaded else "YOLO Model: NOT LOADED"
        model_color = COLOR_SUCCESS if model_loaded else COLOR_DANGER

        self.model_status_label = tk.Label(
            model_status_frame,
            text=model_text,
            font=(FONT_FAMILY, 9, 'bold'),
            bg='#e3f2fd',
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
            bg='#fafafa',
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
            text="Position ingredients in view\nof the camera for detection",
            font=(FONT_FAMILY, 9),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY,
            justify=tk.CENTER
        )
        info_text.pack(pady=(5, 10))

    def start_camera(self):
        """Start camera and detection"""
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
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.configure(text="● Camera Active", fg=COLOR_SUCCESS)

            # Start camera loop
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")

    def stop_camera(self):
        """Stop camera display (but keep shared camera running for other screens)"""
        self.camera_active = False

        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)

        # NOTE: We do NOT stop the shared VisionSystem camera here
        # because other screens (like RobotScreen) might need it.
        # The camera is managed at the application level in main.py

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.configure(text="● Camera Off", fg=COLOR_TEXT_GRAY)

        # Reset display
        self.camera_label.configure(
            image="",
            text="Camera Off\n\nClick 'Start Camera' to begin"
        )
        self.current_detection_label.configure(text="No detection", fg=COLOR_TEXT_GRAY)
        self.confidence_label.configure(text="")
        self.fps_label.configure(text="FPS: 0")

    def camera_loop(self):
        """Main camera loop"""
        while self.camera_active:
            try:
                # Get frame with detection
                frame, detection = self.controller.get_camera_frame()

                if frame is not None:
                    # Convert to PhotoImage
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)

                    # Update camera label
                    self.camera_label.configure(image=photo, text="")
                    self.camera_label.image = photo

                    # Update detection info
                    if detection:
                        self.update_detection(detection)

                    # Update FPS
                    fps = self.controller.get_camera_fps()
                    self.fps_label.configure(text=f"FPS: {fps:.1f}")

                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                print(f"Camera loop error: {e}")
                time.sleep(0.1)

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
            conf_color = "#F39C12"
        else:
            conf_text = f"Confidence: {confidence:.1%} (Low)"
            conf_color = "#E74C3C"

        self.confidence_label.configure(text=conf_text, fg=conf_color)

        # Add to history
        timestamp = time.strftime("%H:%M:%S")
        history_entry = f"[{timestamp}] {class_name} - {confidence:.1%}"

        self.detection_history.append(history_entry)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        # Update history listbox
        self.history_listbox.delete(0, tk.END)
        for entry in reversed(self.detection_history):
            self.history_listbox.insert(tk.END, entry)

    def on_show(self):
        """Called when screen is shown"""
        pass

    def on_hide(self):
        """Called when screen is hidden"""
        # Stop the display loop but DON'T stop the shared camera
        # The camera might be needed by other screens (e.g., RobotScreen)
        self.camera_active = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
