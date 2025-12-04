"""
ChefMate Robot Assistant - Home Screen
Modern landing page with mode selection (IRIS-style)
"""

import tkinter as tk
from tkinter import ttk
from .widgets import ModernButton, ScrollableFrame
from config.settings import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_INFO, COLOR_PURPLE,
    COLOR_BG_DARK, COLOR_BG_LIGHT, COLOR_TEXT_DARK, COLOR_TEXT_GRAY,
    FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL
)


class HomeScreen(ttk.Frame):
    """
    Home screen with three mode options:
    1. Order Pizza (main functionality)
    2. Live Camera (detection only)
    3. File Upload (testing/demo)
    """

    def __init__(self, parent, controller):
        """
        Initialize home screen

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""
        # Use scrollable frame for responsive home
        scroll_frame = ScrollableFrame(self, bg=COLOR_BG_DARK)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Content container
        content = tk.Frame(scroll_frame.scrollable_frame, bg=COLOR_BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Welcome section
        welcome_frame = tk.Frame(content, bg=COLOR_BG_LIGHT, relief=tk.RAISED, borderwidth=2)
        welcome_frame.pack(fill=tk.X, pady=(0, 15))

        welcome_title = tk.Label(
            welcome_frame,
            text="Welcome to ChefMate Robot Assistant",
            font=(FONT_FAMILY, 22, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_PRIMARY
        )
        welcome_title.pack(pady=(20, 8))

        welcome_subtitle = tk.Label(
            welcome_frame,
            text="Automated Pizza Ingredient Picker with Object Detection",
            font=(FONT_FAMILY, 11),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        )
        welcome_subtitle.pack(pady=(0, 20))

        # Mode selection section
        modes_container = tk.Frame(content, bg=COLOR_BG_LIGHT, relief=tk.RAISED, borderwidth=2)
        modes_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        modes_title = tk.Label(
            modes_container,
            text="Select Mode",
            font=(FONT_FAMILY, 13, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        modes_title.pack(pady=(15, 12))

        # Buttons container with responsive grid
        buttons_container = tk.Frame(modes_container, bg=COLOR_BG_LIGHT)
        buttons_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Configure grid to be responsive (3 columns)
        buttons_container.grid_columnconfigure(0, weight=1, minsize=250)
        buttons_container.grid_columnconfigure(1, weight=1, minsize=250)
        buttons_container.grid_columnconfigure(2, weight=1, minsize=250)
        buttons_container.grid_rowconfigure(0, weight=1)

        # Order Pizza Button (Main mode)
        btn_order = ModernButton(
            buttons_container,
            text="ORDER PIZZA\n\nPlace order and watch\nrobot pick ingredients",
            command=self.open_pizza_menu,
            bg=COLOR_SUCCESS
        )
        btn_order.grid(row=0, column=0, padx=5, pady=8, sticky='nsew', ipady=15)

        # Live Camera Button
        btn_camera = ModernButton(
            buttons_container,
            text="LIVE CAMERA\n\nReal-time ingredient\ndetection",
            command=self.open_live_camera,
            bg=COLOR_INFO
        )
        btn_camera.grid(row=0, column=1, padx=5, pady=8, sticky='nsew', ipady=15)

        # File Upload Button
        btn_upload = ModernButton(
            buttons_container,
            text="FILE UPLOAD\n\nUpload images for\ningredient detection",
            command=self.open_file_upload,
            bg=COLOR_PURPLE
        )
        btn_upload.grid(row=0, column=2, padx=5, pady=8, sticky='nsew', ipady=15)

        # Information section
        info_frame = tk.LabelFrame(
            content,
            text="System Information",
            font=(FONT_FAMILY, 11, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_PRIMARY,
            padx=20,
            pady=12
        )
        info_frame.pack(fill=tk.X, pady=(0, 15))

        info_text = (
            "• ORDER PIZZA: Main mode - Select a pizza, robot picks ingredients automatically\n"
            "• LIVE CAMERA: Test ingredient detection in real-time (no robot control)\n"
            "• FILE UPLOAD: Upload images for offline detection testing\n\n"
            "Supported ingredients: Anchovies, Basil, Cheese, Chicken, Fresh Tomato, Shrimp"
        )

        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY,
            justify=tk.LEFT
        )
        info_label.pack(pady=8)

        # Footer
        footer_frame = tk.Frame(content, bg=COLOR_BG_LIGHT, relief=tk.RAISED, borderwidth=2)
        footer_frame.pack(fill=tk.X)

        footer_label = tk.Label(
            footer_frame,
            text="AI in Robotics - Middlesex University Mauritius",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg="#999999"
        )
        footer_label.pack(pady=15)

    def open_pizza_menu(self):
        """Open pizza menu screen"""
        self.controller.show_frame("MenuScreen")

    def open_file_upload(self):
        """Open file upload screen"""
        self.controller.show_frame("FileUploadScreen")

    def open_live_camera(self):
        """Open live camera screen"""
        self.controller.show_frame("LiveCameraScreen")
