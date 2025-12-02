"""
ChefMate Robot Assistant - Home Screen
Main landing page with mode selection
"""

import tkinter as tk
from tkinter import ttk
from config.settings import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS,
    COLOR_BG_DARK, COLOR_TEXT_LIGHT, COLOR_TEXT_DARK,
    FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL
)


class HomeScreen(ttk.Frame):
    """
    Home screen with three mode options:
    1. Order Pizza (main functionality)
    2. File Upload (testing/demo)
    3. Live Camera (detection only)
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
        # Main container
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title Section
        title_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        title_frame.pack(pady=(0, 40))

        # App Title
        title_label = tk.Label(
            title_frame,
            text="🍕 ChefMate Robot Assistant",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack()

        # Subtitle
        subtitle_label = tk.Label(
            title_frame,
            text="Automated Pizza Ingredient Picker",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT_LIGHT
        )
        subtitle_label.pack(pady=(5, 0))

        # Buttons Container
        buttons_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        buttons_frame.pack(expand=True)

        # Button configurations
        button_config = [
            {
                "text": "🍕 Order Pizza",
                "description": "Place order and watch robot pick ingredients",
                "color": COLOR_PRIMARY,
                "hover_color": "#C0392B",
                "command": self.open_pizza_menu
            },
            {
                "text": "📁 File Upload",
                "description": "Upload images for ingredient detection",
                "color": COLOR_SECONDARY,
                "hover_color": "#D68910",
                "command": self.open_file_upload
            },
            {
                "text": "📷 Live Camera",
                "description": "Real-time ingredient detection (no robot)",
                "color": COLOR_SUCCESS,
                "hover_color": "#229954",
                "command": self.open_live_camera
            }
        ]

        # Create buttons
        self.buttons = []
        for idx, config in enumerate(button_config):
            btn = self.create_mode_button(
                buttons_frame,
                config["text"],
                config["description"],
                config["color"],
                config["hover_color"],
                config["command"]
            )
            btn.pack(pady=15)
            self.buttons.append(btn)

        # Footer
        footer_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        footer_frame.pack(side=tk.BOTTOM, pady=(40, 0))

        footer_label = tk.Label(
            footer_frame,
            text="AI in Robotics - Assessment 2 | Middlesex University",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_DARK,
            fg="#7F8C8D"
        )
        footer_label.pack()

    def create_mode_button(self, parent, text, description, color, hover_color, command):
        """
        Create a styled mode selection button

        Args:
            parent: Parent widget
            text: Button text
            description: Button description
            color: Button background color
            hover_color: Button hover color
            command: Button command

        Returns:
            Button frame
        """
        # Button container
        button_frame = tk.Frame(
            parent,
            bg=color,
            width=500,
            height=100,
            relief=tk.RAISED,
            borderwidth=2
        )
        button_frame.pack_propagate(False)

        # Make entire frame clickable
        button_frame.bind("<Button-1>", lambda e: command())
        button_frame.bind("<Enter>", lambda e: button_frame.configure(bg=hover_color))
        button_frame.bind("<Leave>", lambda e: button_frame.configure(bg=color))

        # Button content
        content_frame = tk.Frame(button_frame, bg=color)
        content_frame.pack(expand=True)
        content_frame.bind("<Button-1>", lambda e: command())
        content_frame.bind("<Enter>", lambda e: self.on_enter_button(button_frame, content_frame, hover_color))
        content_frame.bind("<Leave>", lambda e: self.on_leave_button(button_frame, content_frame, color))

        # Button text
        text_label = tk.Label(
            content_frame,
            text=text,
            font=(FONT_FAMILY, 18, "bold"),
            bg=color,
            fg=COLOR_TEXT_LIGHT,
            cursor="hand2"
        )
        text_label.pack()
        text_label.bind("<Button-1>", lambda e: command())
        text_label.bind("<Enter>", lambda e: self.on_enter_button(button_frame, content_frame, hover_color))
        text_label.bind("<Leave>", lambda e: self.on_leave_button(button_frame, content_frame, color))

        # Description
        desc_label = tk.Label(
            content_frame,
            text=description,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=color,
            fg=COLOR_TEXT_LIGHT,
            cursor="hand2"
        )
        desc_label.pack(pady=(5, 0))
        desc_label.bind("<Button-1>", lambda e: command())
        desc_label.bind("<Enter>", lambda e: self.on_enter_button(button_frame, content_frame, hover_color))
        desc_label.bind("<Leave>", lambda e: self.on_leave_button(button_frame, content_frame, color))

        return button_frame

    def on_enter_button(self, frame, content_frame, color):
        """Handle mouse enter event"""
        frame.configure(bg=color)
        content_frame.configure(bg=color)
        for widget in content_frame.winfo_children():
            widget.configure(bg=color)

    def on_leave_button(self, frame, content_frame, color):
        """Handle mouse leave event"""
        frame.configure(bg=color)
        content_frame.configure(bg=color)
        for widget in content_frame.winfo_children():
            widget.configure(bg=color)

    def open_pizza_menu(self):
        """Open pizza menu screen"""
        self.controller.show_frame("MenuScreen")

    def open_file_upload(self):
        """Open file upload screen"""
        self.controller.show_frame("FileUploadScreen")

    def open_live_camera(self):
        """Open live camera screen"""
        self.controller.show_frame("LiveCameraScreen")