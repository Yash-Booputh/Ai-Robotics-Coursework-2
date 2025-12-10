"""
ChefMate Robot Assistant - Loading Screen
Simple loading screen with logo, slogan, and animated loading bar
"""

import tkinter as tk
from PIL import Image, ImageTk
import os


class LoadingScreen(tk.Toplevel):
    """Simple loading screen that displays for 3.5 seconds"""

    def __init__(self, parent):
        """Initialize loading screen"""
        super().__init__(parent)

        # Colors
        self.bg_color = "#DACFBD"
        self.progress_color = "#02332D"
        self.text_color = "#02332D"

        # Window setup
        self.title("Loading ChefMate...")
        self.configure(bg=self.bg_color)
        self.overrideredirect(True)

        # Center on screen
        window_width = 600
        window_height = 500
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.attributes('-topmost', True)

        self.create_widgets()
        self.animate_loading()

    def create_widgets(self):
        """Create loading screen widgets"""
        main_frame = tk.Frame(self, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)

        # Logo
        logo_frame = tk.Frame(main_frame, bg=self.bg_color)
        logo_frame.pack(expand=True, pady=(20, 10))

        try:
            logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'logo.png')
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((180, 180), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                tk.Label(logo_frame, image=self.logo_photo, bg=self.bg_color).pack()
        except Exception as e:
            print(f"Could not load logo: {e}")

        # Slogan
        tk.Label(
            main_frame,
            text="One Slice and—Mamma Mia!",
            font=("Arial", 16, "italic"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=(10, 30))

        # App name
        tk.Label(
            main_frame,
            text="ChefMate Robot Assistant",
            font=("Arial", 20, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=(0, 40))

        # Loading bar background
        progress_bg = tk.Frame(main_frame, bg="#A89B8E", height=30)
        progress_bg.pack(fill=tk.X)

        # Loading bar
        self.progress_bar = tk.Frame(progress_bg, bg=self.progress_color, height=30)
        self.progress_bar.place(x=0, y=0, relwidth=0, relheight=1)

        # Loading text
        tk.Label(
            main_frame,
            text="Loading...",
            font=("Arial", 11),
            bg=self.bg_color,
            fg=self.text_color
        ).pack(pady=(10, 0))

    def animate_loading(self):
        """Animate loading bar over 3.5 seconds"""
        duration = 3500  # 3.5 seconds
        steps = 70  # Number of animation steps
        step_delay = duration // steps

        def update_bar(step):
            if step <= steps:
                progress = step / steps
                self.progress_bar.place(x=0, y=0, relwidth=progress, relheight=1)
                self.after(step_delay, lambda: update_bar(step + 1))
            else:
                self.destroy()

        update_bar(0)
