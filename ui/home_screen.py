"""
ChefMate Robot Assistant - Home Screen
Modern landing page with mode selection (IRIS-style)
"""

import tkinter as tk
from tkinter import ttk
from .widgets import ModernButton, ScrollableFrame
from config.settings import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS, COLOR_INFO, COLOR_PURPLE,
    COLOR_BG_DARK, COLOR_BG_LIGHT, COLOR_TEXT_DARK, COLOR_TEXT_GRAY, COLOR_TEXT_MUTED,
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

        # Configure grid to be responsive (4 columns now)
        buttons_container.grid_columnconfigure(0, weight=1, minsize=200)
        buttons_container.grid_columnconfigure(1, weight=1, minsize=200)
        buttons_container.grid_columnconfigure(2, weight=1, minsize=200)
        buttons_container.grid_columnconfigure(3, weight=1, minsize=200)
        buttons_container.grid_rowconfigure(0, weight=1)

        # Order Pizza Button (Main mode)
        btn_order = ModernButton(
            buttons_container,
            text="ORDER PIZZA\n\nPlace order and watch\nrobot pick ingredients",
            command=self.open_pizza_menu,
            bg=COLOR_SUCCESS
        )
        btn_order.grid(row=0, column=0, padx=5, pady=8, sticky='nsew', ipady=15)

        # AprilTag Pizza Order Button (NEW)
        btn_apriltag = ModernButton(
            buttons_container,
            text="APRILTAG PIZZA\n\nChef Surprise with\nAprilTag & gestures",
            command=self.open_apriltag_order,
            bg=COLOR_SECONDARY  # Warm gold color
        )
        btn_apriltag.grid(row=0, column=1, padx=5, pady=8, sticky='nsew', ipady=15)

        # Live Camera Button
        btn_camera = ModernButton(
            buttons_container,
            text="LIVE CAMERA\n\nReal-time ingredient\ndetection",
            command=self.open_live_camera,
            bg=COLOR_INFO
        )
        btn_camera.grid(row=0, column=2, padx=5, pady=8, sticky='nsew', ipady=15)

        # File Upload Button
        btn_upload = ModernButton(
            buttons_container,
            text="FILE UPLOAD\n\nUpload images for\ningredient detection",
            command=self.open_file_upload,
            bg=COLOR_PURPLE
        )
        btn_upload.grid(row=0, column=3, padx=5, pady=8, sticky='nsew', ipady=15)

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
            "• APRILTAG PIZZA: Chef Surprise - Use AprilTag & hand gestures for random pizza\n"
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
            fg=COLOR_TEXT_MUTED
        )
        footer_label.pack(pady=15)

    def on_show(self):
        """Called when screen is shown"""
        # Ensure VisionSystem camera is stopped when returning to home
        # This prevents camera conflicts when launching AprilTag detector
        self.controller.stop_vision_camera()

    def open_pizza_menu(self):
        """Open pizza menu screen"""
        self.controller.show_frame("MenuScreen")

    def open_apriltag_order(self):
        """Launch standalone AprilTag detector and wait for order"""
        import subprocess
        import json
        from pathlib import Path
        from tkinter import messagebox

        # Run the standalone AprilTag detector
        try:
            # IMPORTANT: Stop the VisionSystem camera before launching AprilTag detector
            # to prevent camera access conflicts
            self.controller.stop_vision_camera()

            # Give the camera time to fully release
            import time
            time.sleep(0.5)

            script_path = Path(__file__).parent.parent / "apriltag_pizza_detector_simple.py"

            # Inform user
            messagebox.showinfo(
                "AprilTag Chef Surprise",
                "Launching AprilTag detector...\n\n"
                "1. Show AprilTag 0 to activate\n"
                "2. Use OPEN HAND to generate pizzas\n"
                "3. Use THUMBS UP to lock your order\n"
                "4. Close the window when done\n\n"
                "Your order will be added to the cart!"
            )

            # Launch the standalone detector (blocking call)
            result = subprocess.run(["python", str(script_path)], check=True)

            # After it closes, look for the most recent order file
            orders_dir = Path(__file__).parent.parent / "pizza_orders"
            if orders_dir.exists():
                order_files = sorted(orders_dir.glob("order_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

                if order_files:
                    # Check if the most recent order was created in the last 10 seconds
                    import time
                    most_recent = order_files[0]
                    time_diff = time.time() - most_recent.stat().st_mtime

                    if time_diff < 10:  # Order was just created
                        # Load the most recent order
                        with open(most_recent, 'r') as f:
                            order_data = json.load(f)

                        # Add to recipes dynamically
                        from config.recipes import add_chef_surprise_recipe

                        # Map ingredient names back to IDs
                        ingredient_mapping = {
                            "Anchovies": "anchovies",
                            "Fresh Basil": "basil",
                            "Mozzarella Cheese": "cheese",
                            "Grilled Chicken": "chicken",
                            "Fresh Tomato": "fresh_tomato",
                            "Shrimp": "shrimp"
                        }

                        ingredients = [ingredient_mapping.get(ing, ing.lower().replace(" ", "_"))
                                       for ing in order_data["ingredients"]]

                        pizza_data = {
                            "name": "Chef Surprise",
                            "description": "Random chef's selection",
                            "ingredients": ingredients,
                            "price": float(order_data["price"].replace("$", ""))
                        }
                        add_chef_surprise_recipe(pizza_data)

                        # Set the order and go to cart
                        self.controller.set_pizza_order("Chef Surprise")
                        self.controller.show_frame("CartScreen")
                    else:
                        # User quit without completing order - stay on home screen
                        messagebox.showinfo("Order Cancelled", "No order was placed. Returning to home screen.")
                else:
                    # No order files - user quit without ordering
                    messagebox.showinfo("Order Cancelled", "No order was placed. Returning to home screen.")

        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "AprilTag detector exited with an error.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch AprilTag detector:\n{str(e)}")

    def open_file_upload(self):
        """Open file upload screen"""
        self.controller.show_frame("FileUploadScreen")

    def open_live_camera(self):
        """Open live camera screen"""
        self.controller.show_frame("LiveCameraScreen")
