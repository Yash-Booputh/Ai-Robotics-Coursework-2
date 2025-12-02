"""
ChefMate Robot Assistant - Live Camera Screen
Real-time ingredient detection (no robot control)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time

from config.settings import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER,
    COLOR_BG_DARK, COLOR_BG_MEDIUM, COLOR_TEXT_LIGHT,
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
        """Create all UI widgets"""
        # Main container
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK, height=80)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        header_frame.pack_propagate(False)

        # Back button
        back_btn = tk.Button(
            header_frame,
            text="← Back",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT,
            command=self.go_back,
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        back_btn.pack(side=tk.LEFT)

        # Title
        title_label = tk.Label(
            header_frame,
            text="📷 Live Camera Detection",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack(side=tk.LEFT, padx=20)

        # Start/Stop buttons
        self.start_btn = tk.Button(
            header_frame,
            text="▶ Start Camera",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            command=self.start_camera,
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.start_btn.pack(side=tk.RIGHT, padx=5)

        self.stop_btn = tk.Button(
            header_frame,
            text="⏹ Stop Camera",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_DANGER,
            fg=COLOR_TEXT_LIGHT,
            command=self.stop_camera,
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=5)

        # Content area
        content_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Left: Camera feed
        camera_container = tk.Frame(content_frame, bg=COLOR_BG_MEDIUM, relief=tk.RAISED, borderwidth=2)
        camera_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        camera_title = tk.Label(
            camera_container,
            text="Camera Feed",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        camera_title.pack(pady=10)

        # Camera display
        self.camera_label = tk.Label(
            camera_container,
            bg=COLOR_BG_DARK,
            text="Camera Off\n\nClick 'Start Camera' to begin",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            fg="#7F8C8D",
            width=80,
            height=30
        )
        self.camera_label.pack(padx=10, pady=(0, 10), expand=True)

        # FPS and status
        status_frame = tk.Frame(camera_container, bg=COLOR_BG_MEDIUM)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.fps_label = tk.Label(
            status_frame,
            text="FPS: 0",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg="#7F8C8D"
        )
        self.fps_label.pack(side=tk.LEFT)

        self.status_label = tk.Label(
            status_frame,
            text="● Camera Off",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg="#7F8C8D"
        )
        self.status_label.pack(side=tk.RIGHT)

        # Right: Detection info
        info_container = tk.Frame(content_frame, bg=COLOR_BG_MEDIUM, width=350, relief=tk.RAISED, borderwidth=2)
        info_container.pack(side=tk.RIGHT, fill=tk.Y)
        info_container.pack_propagate(False)

        info_title = tk.Label(
            info_container,
            text="Detection Info",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        info_title.pack(pady=10)

        # Current detection box
        current_frame = tk.Frame(info_container, bg=COLOR_BG_DARK, relief=tk.SUNKEN, borderwidth=2)
        current_frame.pack(fill=tk.X, padx=20, pady=10)

        current_title = tk.Label(
            current_frame,
            text="Current Detection",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT_LIGHT
        )
        current_title.pack(pady=(10, 5))

        self.current_detection_label = tk.Label(
            current_frame,
            text="No detection",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_DARK,
            fg="#7F8C8D"
        )
        self.current_detection_label.pack(pady=(5, 10))

        self.confidence_label = tk.Label(
            current_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_DARK,
            fg="#7F8C8D"
        )
        self.confidence_label.pack(pady=(0, 10))

        # Detection history
        history_title = tk.Label(
            info_container,
            text="Recent Detections",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        history_title.pack(pady=(10, 5))

        # History list with scrollbar
        history_frame = tk.Frame(info_container, bg=COLOR_BG_MEDIUM)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        history_scrollbar = ttk.Scrollbar(history_frame)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_listbox = tk.Listbox(
            history_frame,
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT_LIGHT,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            yscrollcommand=history_scrollbar.set,
            selectmode=tk.SINGLE
        )
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.config(command=self.history_listbox.yview)

        # Info text
        info_text = tk.Label(
            info_container,
            text="ℹ️ Position cubes in view\nof the camera to test detection",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg="#7F8C8D",
            wraplength=300,
            justify=tk.CENTER
        )
        info_text.pack(pady=10)

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
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.status_label.configure(text="● Camera Active", fg=COLOR_SUCCESS)

            # Start camera loop
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")

    def stop_camera(self):
        """Stop camera"""
        self.camera_active = False

        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)

        self.controller.stop_vision_camera()

        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_label.configure(text="● Camera Off", fg="#7F8C8D")

        # Reset display
        self.camera_label.configure(
            image="",
            text="Camera Off\n\nClick 'Start Camera' to begin"
        )
        self.current_detection_label.configure(text="No detection", fg="#7F8C8D")
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

    def go_back(self):
        """Go back to home screen"""
        self.stop_camera()
        self.controller.show_frame("HomeScreen")

    def on_show(self):
        """Called when screen is shown"""
        pass

    def on_hide(self):
        """Called when screen is hidden"""
        self.stop_camera()