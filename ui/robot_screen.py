"""
ChefMate Robot Assistant - Robot Execution Screen
Monitor robot picking ingredients with live camera feed and animated pizza making
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
import math
import random

from config.settings import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING,
    COLOR_BG_DARK, COLOR_BG_MEDIUM, COLOR_BG_LIGHT, COLOR_TEXT_LIGHT, COLOR_TEXT_DARK,
    COLOR_TEXT_GRAY, FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL
)
from config.recipes import get_pizza_ingredients, get_ingredient_display_name, PIZZA_RECIPES


class RobotScreen(ttk.Frame):
    """
    Robot execution screen
    Shows live camera feed and progress of robot picking ingredients
    """

    def __init__(self, parent, controller):
        """
        Initialize robot screen

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')

        self.pizza_name = None
        self.ingredients_list = []
        self.current_index = 0
        self.is_running = False
        self.camera_active = False

        # Pizza maker animation
        self.pizza_images = {}
        self.pizza_canvas = None
        self.pizza_center_x = 190  # Will be centered in 380x380 canvas
        self.pizza_center_y = 190  # Will be centered in 380x380 canvas
        self.pizza_radius = 120
        self.has_cheese = False
        self.placed_ingredients = []

        self.create_widgets()
        self.load_pizza_images()

    def create_widgets(self):
        """Create all UI widgets"""
        # Main container
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top section - Status and controls (reduced padding for smaller screens)
        top_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        top_frame.pack(fill=tk.X, padx=20, pady=(10, 5))

        # Title
        title_label = tk.Label(
            top_frame,
            text="Robot Execution",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack(side=tk.LEFT)

        # Status label
        self.status_label = tk.Label(
            top_frame,
            text="Ready",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_DARK,
            fg=COLOR_SUCCESS
        )
        self.status_label.pack(side=tk.LEFT, padx=30)

        # Stop button
        self.stop_btn = tk.Button(
            top_frame,
            text="Stop",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_DANGER,
            fg=COLOR_TEXT_LIGHT,
            command=self.stop_execution,
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.RIGHT)

        # Middle section - 3 columns: Camera, Pizza Maker, Progress
        # Give it less weight so log section gets space
        middle_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 5))

        # Configure grid weights for responsive, proportional sizing
        # Camera and Pizza Maker expand, Progress stays fixed
        middle_frame.grid_rowconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(0, weight=1, minsize=300)
        middle_frame.grid_columnconfigure(1, weight=1, minsize=300)
        middle_frame.grid_columnconfigure(2, weight=0)

        # Left: Camera status
        camera_frame = tk.Frame(middle_frame, bg=COLOR_BG_MEDIUM, relief=tk.RAISED, borderwidth=2)
        camera_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        camera_title = tk.Label(
            camera_frame,
            text="Camera Status",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_DARK
        )
        camera_title.pack(pady=10)

        # Camera display (smaller, fixed size)
        camera_container = tk.Frame(camera_frame, bg=COLOR_TEXT_GRAY, relief=tk.SUNKEN, borderwidth=2)
        camera_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.camera_label = tk.Label(
            camera_container,
            bg=COLOR_TEXT_GRAY,
            fg=COLOR_TEXT_LIGHT,
            text="Camera feed will appear here\nwhen robot starts",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            wraplength=350
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # FPS label
        self.fps_label = tk.Label(
            camera_frame,
            text="FPS: 0",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_SUCCESS
        )
        self.fps_label.pack(pady=(0, 10))

        # Center: Pizza Maker Animation
        pizza_frame = tk.Frame(middle_frame, bg=COLOR_TEXT_GRAY, relief=tk.RAISED, borderwidth=3)
        pizza_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        pizza_title = tk.Label(
            pizza_frame,
            text="Pizza Maker",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_TEXT_GRAY,
            fg=COLOR_SECONDARY
        )
        pizza_title.pack(pady=10)

        # Pizza canvas (fixed size, centered, smaller for laptop screens)
        canvas_container = tk.Frame(pizza_frame, bg=COLOR_TEXT_GRAY)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.pizza_canvas = tk.Canvas(
            canvas_container,
            width=380,
            height=380,
            bg=COLOR_BG_DARK,
            highlightthickness=2,
            highlightbackground=COLOR_BG_MEDIUM
        )
        self.pizza_canvas.pack(padx=10, pady=10, expand=True)

        # Pizza status label
        self.pizza_status_label = tk.Label(
            pizza_frame,
            text="Waiting for order...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_TEXT_GRAY,
            fg=COLOR_SECONDARY,
            wraplength=430
        )
        self.pizza_status_label.pack(pady=(0, 10))

        # Right: Progress panel (fixed narrow width)
        progress_outer = tk.Frame(middle_frame, bg=COLOR_BG_DARK, width=250)
        progress_outer.grid(row=0, column=2, sticky="ns", padx=(5, 0))
        progress_outer.grid_propagate(False)
        progress_outer.update_idletasks()

        progress_frame = tk.Frame(progress_outer, bg=COLOR_BG_MEDIUM, relief=tk.RAISED, borderwidth=2, width=250)
        progress_frame.pack(fill=tk.BOTH, expand=True)
        progress_frame.pack_propagate(False)

        progress_title = tk.Label(
            progress_frame,
            text="Progress",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_DARK
        )
        progress_title.pack(pady=10)

        # Pizza name
        self.pizza_label = tk.Label(
            progress_frame,
            text="No order",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_DARK
        )
        self.pizza_label.pack(pady=(0, 20))

        # Progress bar (green style)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("green.Horizontal.TProgressbar",
                       foreground=COLOR_SUCCESS,
                       background=COLOR_SUCCESS,
                       troughcolor=COLOR_BG_DARK,
                       bordercolor=COLOR_BG_DARK,
                       lightcolor=COLOR_SUCCESS,
                       darkcolor=COLOR_SUCCESS)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            style="green.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=10, padx=10, fill=tk.X)

        # Progress text
        self.progress_text = tk.Label(
            progress_frame,
            text="0 / 0 ingredients",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_DARK
        )
        self.progress_text.pack(pady=(0, 20))

        # Ingredients checklist
        checklist_label = tk.Label(
            progress_frame,
            text="Ingredients:",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_DARK
        )
        checklist_label.pack(pady=(10, 5))

        # Scrollable checklist
        checklist_canvas = tk.Canvas(progress_frame, bg=COLOR_BG_MEDIUM, highlightthickness=0, height=200)
        checklist_scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=checklist_canvas.yview)
        self.checklist_frame = tk.Frame(checklist_canvas, bg=COLOR_BG_MEDIUM)

        self.checklist_frame.bind(
            "<Configure>",
            lambda e: checklist_canvas.configure(scrollregion=checklist_canvas.bbox("all"))
        )

        checklist_canvas.create_window((0, 0), window=self.checklist_frame, anchor="nw")
        checklist_canvas.configure(yscrollcommand=checklist_scrollbar.set)

        checklist_canvas.pack(side=tk.LEFT, fill=tk.BOTH, padx=10)
        checklist_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Current action label
        self.action_label = tk.Label(
            progress_frame,
            text="Waiting to start...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg="#7F8C8D",
            wraplength=200,
            justify=tk.LEFT
        )
        self.action_label.pack(pady=20, padx=10)

        # Bottom section - Log (compact for smaller screens)
        log_frame = tk.Frame(main_frame, bg=COLOR_BG_MEDIUM, relief=tk.RAISED, borderwidth=2, height=120)
        log_frame.pack(fill=tk.X, padx=20, pady=(5, 10))
        log_frame.pack_propagate(False)

        log_title = tk.Label(
            log_frame,
            text="Activity Log",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_DARK
        )
        log_title.pack(anchor="w", padx=10, pady=(3, 0))

        # Log text widget with scrollbar
        log_container = tk.Frame(log_frame, bg=COLOR_BG_MEDIUM)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(3, 5))

        log_scrollbar = ttk.Scrollbar(log_container)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_container,
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            height=4,
            yscrollcommand=log_scrollbar.set,
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.log_text.yview)

        self.checklist_items = []

    def start_execution(self, pizza_name):
        """
        Start robot execution for an order

        Args:
            pizza_name: Name of pizza to make
        """
        self.pizza_name = pizza_name
        self.ingredients_list = get_pizza_ingredients(pizza_name)
        self.current_index = 0
        self.is_running = True

        # Update UI
        self.pizza_label.configure(text=f"Making: {pizza_name}")
        self.progress_var.set(0)
        self.progress_text.configure(text=f"0 / {len(self.ingredients_list)} ingredients")
        self.status_label.configure(text="Running", fg=COLOR_SUCCESS)
        self.stop_btn.configure(state=tk.NORMAL)

        # Create checklist
        self.create_checklist()

        # Initialize pizza maker animation
        self.has_cheese = False
        self.placed_ingredients = []
        self.draw_initial_pizza()

        # Start camera
        self.start_camera()

        # Log start
        self.add_log(f"Starting order: {pizza_name}")
        self.add_log(f"Ingredients needed: {len(self.ingredients_list)}")

        # Start execution in thread
        threading.Thread(target=self.execute_order, daemon=True).start()

    def create_checklist(self):
        """Create ingredient checklist"""
        # Clear existing
        for widget in self.checklist_frame.winfo_children():
            widget.destroy()

        self.checklist_items = []

        for idx, ingredient in enumerate(self.ingredients_list):
            item_frame = tk.Frame(self.checklist_frame, bg=COLOR_BG_MEDIUM)
            item_frame.pack(fill=tk.X, pady=2)

            status_label = tk.Label(
                item_frame,
                text="[ ]",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                bg=COLOR_BG_MEDIUM,
                fg="#7F8C8D",
                width=3
            )
            status_label.pack(side=tk.LEFT)

            name_label = tk.Label(
                item_frame,
                text=get_ingredient_display_name(ingredient),
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                bg=COLOR_BG_MEDIUM,
                fg=COLOR_TEXT_DARK,
                anchor="w"
            )
            name_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.checklist_items.append((status_label, name_label))

    def execute_order(self):
        """Execute the robot picking sequence"""
        try:
            # Call controller to start pick sequence
            success = self.controller.execute_pick_sequence(
                self.pizza_name,
                status_callback=self.update_status
            )

            if success:
                self.add_log("Order completed successfully!")
                self.status_label.configure(text="Completed", fg=COLOR_SUCCESS)

                # Clear Chef Surprise from recipes if it was used
                if self.pizza_name == "Chef Surprise":
                    from config.recipes import clear_chef_surprise
                    clear_chef_surprise()
                    self.add_log("Chef Surprise cleared from menu")

                messagebox.showinfo("Success", f"{self.pizza_name} is ready!")
            else:
                self.add_log("Order failed - could not find all ingredients")
                self.status_label.configure(text="Failed", fg=COLOR_DANGER)
                messagebox.showerror("Order Failed", "Could not find all ingredients!\n\nSome ingredients may be missing or out of view.")

        except Exception as e:
            self.add_log(f"Error: {str(e)}")
            self.status_label.configure(text="Error", fg=COLOR_DANGER)
            messagebox.showerror("Error", f"Execution error: {str(e)}")

        finally:
            self.is_running = False
            self.stop_btn.configure(state=tk.DISABLED)

    def update_status(self, message, status_type="info"):
        """
        Update status from pick sequence

        Args:
            message: Status message
            status_type: Type of status (info, success, error, warning)
        """
        # Update action label
        self.action_label.configure(text=message)

        # Add to log
        self.add_log(message)

        # Play sound effects based on message content
        if hasattr(self.controller, 'audio') and self.controller.audio:
            # Play "found" sound when ingredient is detected
            if "found" in message.lower() or "detected" in message.lower():
                self.controller.audio.play_found_cube()

            # Play "delivery" sound when cube is dropped/delivered
            elif "delivered" in message.lower() or "dropped" in message.lower() or "placed" in message.lower():
                self.controller.audio.play_drop_cube()

            # Play error sound on errors
            elif status_type == "error" or "error" in message.lower() or "failed" in message.lower():
                self.controller.audio.play_error()

        # Update checklist and pizza animation if ingredient picked
        if "Picked" in message or "delivered" in message or "grabbed" in message.lower():
            # Check which ingredient was grabbed
            if self.current_index < len(self.ingredients_list):
                ingredient = self.ingredients_list[self.current_index]

                # Add ingredient to pizza animation
                self.add_pizza_ingredient(ingredient)

            # Update progress
            self.current_index += 1

            # Update checklist
            for idx, (status_label, name_label) in enumerate(self.checklist_items):
                if idx < self.current_index:
                    status_label.configure(text="[X]", fg=COLOR_SUCCESS)
            progress = (self.current_index / len(self.ingredients_list)) * 100
            self.progress_var.set(progress)
            self.progress_text.configure(text=f"{self.current_index} / {len(self.ingredients_list)} ingredients")

            # Check if pizza is complete
            if self.current_index >= len(self.ingredients_list):
                self.mark_pizza_complete()

    def start_camera(self):
        """Start camera feed display from IntegratedPatrolGrabSystem"""
        if self.camera_active:
            return

        self.camera_active = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def camera_loop(self):
        """Camera feed loop - displays feed from IntegratedPatrolGrabSystem's camera"""
        import time
        last_time = time.time()
        frame_count = 0
        fps = 0

        while self.camera_active:
            try:
                # Try to get frame from controller's pick_sequence patrol_system
                if (hasattr(self.controller, 'pick_sequence') and
                    self.controller.pick_sequence and
                    hasattr(self.controller.pick_sequence, 'patrol_system') and
                    self.controller.pick_sequence.patrol_system):

                    patrol_system = self.controller.pick_sequence.patrol_system

                    # Access IntegratedPatrolGrabSystem's camera directly
                    if hasattr(patrol_system, 'cap') and patrol_system.cap:
                        ret, frame = patrol_system.cap.read()

                        if ret and frame is not None:
                            # Convert to PhotoImage
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            img = img.resize((380, 285), Image.Resampling.LANCZOS)
                            photo = ImageTk.PhotoImage(img)

                            # Update camera label
                            self.camera_label.configure(image=photo, text="")
                            self.camera_label.image = photo

                            # Calculate FPS
                            frame_count += 1
                            current_time = time.time()
                            if current_time - last_time >= 1.0:
                                fps = frame_count / (current_time - last_time)
                                self.fps_label.configure(text=f"FPS: {fps:.1f}")
                                frame_count = 0
                                last_time = current_time

                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                # Camera not available yet - show waiting message
                self.camera_label.configure(
                    text="Waiting for camera...",
                    font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                    fg=COLOR_TEXT_DARK
                )
                time.sleep(0.5)

        self.camera_active = False

    def stop_camera(self):
        """Stop camera feed"""
        self.camera_active = False

    def stop_execution(self, ask_confirmation=True):
        """
        Stop robot execution

        Args:
            ask_confirmation: Whether to show confirmation dialog (default True)
        """
        if ask_confirmation:
            response = messagebox.askyesno(
                "Stop Execution",
                "Are you sure you want to stop?\nThe robot will return to home position.",
                icon="warning"
            )
            if not response:
                return

        self.add_log("Stopping execution...")
        self.is_running = False
        self.controller.stop_pick_sequence()
        self.status_label.configure(text="Stopped", fg=COLOR_WARNING)
        self.stop_btn.configure(state=tk.DISABLED)

    def add_log(self, message):
        """
        Add message to log

        Args:
            message: Log message
        """
        self.log_text.configure(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # =========================================================================
    # PIZZA MAKER ANIMATION METHODS
    # =========================================================================

    def load_pizza_images(self):
        """Load all pizza base and ingredient images"""
        pizza_base_dir = "assets/pizza_bases"
        ingredient_dir = "assets/ingredients"

        pizza_base_size = (240, 240)
        ingredient_size = (50, 50)

        # Pizza base images
        base_images = {
            'dough_sauce': 'dough_with_sauce',
            'dough_cheese': 'dough_with_cheese'
        }

        for key, base_name in base_images.items():
            path = self.find_image_file(pizza_base_dir, base_name)
            if path:
                try:
                    with Image.open(path) as img:
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        img_resized = img.resize(pizza_base_size, Image.Resampling.LANCZOS)
                        self.pizza_images[key] = ImageTk.PhotoImage(img_resized)
                except Exception as e:
                    print(f"Error loading {base_name}: {e}")

        # Ingredient images
        ingredient_mappings = {
            'fresh_tomato': 'tomato',
            'basil': 'basil',
            'anchovies': 'anchovies',
            'chicken': 'chicken',
            'shrimp': 'shrimp',
            'cheese': 'cheese'  # In case we need it
        }

        for key, base_name in ingredient_mappings.items():
            path = self.find_image_file(ingredient_dir, base_name)
            if path:
                try:
                    with Image.open(path) as img:
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        img_resized = img.resize(ingredient_size, Image.Resampling.LANCZOS)
                        self.pizza_images[key] = ImageTk.PhotoImage(img_resized)
                except Exception as e:
                    print(f"Error loading {base_name}: {e}")

    def find_image_file(self, directory, base_name):
        """Find image file with either .png or .jpg extension"""
        if not os.path.exists(directory):
            return None

        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            filename = base_name + ext
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                return path
        return None

    def draw_initial_pizza(self):
        """Draw the initial pizza base on canvas"""
        if not self.pizza_canvas:
            return

        # Clear canvas
        self.pizza_canvas.delete("all")

        # Oven background
        self.pizza_canvas.create_oval(
            self.pizza_center_x - 140,
            self.pizza_center_y - 140,
            self.pizza_center_x + 140,
            self.pizza_center_y + 140,
            fill=COLOR_TEXT_GRAY,
            outline=COLOR_BG_MEDIUM,
            width=2,
            tags="oven"
        )

        # Pizza base - start with sauce
        if 'dough_sauce' in self.pizza_images:
            self.pizza_canvas.create_image(
                self.pizza_center_x,
                self.pizza_center_y,
                image=self.pizza_images['dough_sauce'],
                tags="pizza_base"
            )
        else:
            # Fallback if image not found
            self.pizza_canvas.create_oval(
                self.pizza_center_x - 120,
                self.pizza_center_y - 120,
                self.pizza_center_x + 120,
                self.pizza_center_y + 120,
                fill=COLOR_PRIMARY,
                outline=COLOR_DANGER,
                width=3,
                tags="pizza_base"
            )

        # Ensure proper layering
        self.pizza_canvas.tag_lower("pizza_base")
        self.pizza_canvas.tag_lower("oven")

        self.pizza_status_label.config(text="Preparing the dough...")

    def add_pizza_ingredient(self, ingredient_name):
        """Add an ingredient to the pizza animation"""
        if not self.pizza_canvas:
            return

        display_name = get_ingredient_display_name(ingredient_name)
        self.pizza_status_label.config(text=f"Adding {display_name}...")

        if ingredient_name == "cheese":
            self.add_cheese_layer()
        else:
            self.add_ingredient_pieces(ingredient_name)

        self.update()

    def add_cheese_layer(self):
        """Add cheese by switching to dough_with_cheese image"""
        if self.has_cheese:
            return

        self.has_cheese = True

        if 'dough_cheese' in self.pizza_images:
            self.pizza_canvas.delete("pizza_base")
            self.pizza_canvas.create_image(
                self.pizza_center_x,
                self.pizza_center_y,
                image=self.pizza_images['dough_cheese'],
                tags="pizza_base"
            )
            # Ensure proper layering
            self.pizza_canvas.tag_lower("pizza_base")
            self.pizza_canvas.tag_lower("oven")

    def add_ingredient_pieces(self, ingredient_name):
        """Add multiple pieces of an ingredient in circular pattern"""
        if ingredient_name not in self.pizza_images:
            return

        # Determine number of pieces
        counts = {
            'fresh_tomato': 7,
            'basil': 6,
            'anchovies': 7,
            'chicken': 8,
            'shrimp': 7
        }

        num_pieces = counts.get(ingredient_name, 6)

        # Get circular positions
        positions = self.distribute_ingredients_circular(num_pieces)

        # Place each piece
        for x, y in positions:
            self.pizza_canvas.create_image(
                x, y,
                image=self.pizza_images[ingredient_name],
                tags=f"ingredient_{ingredient_name}"
            )
            self.placed_ingredients.append((x, y))

    def distribute_ingredients_circular(self, num_pieces):
        """Distribute ingredients in circular pattern"""
        positions = []

        if num_pieces <= 1:
            positions.append((self.pizza_center_x, self.pizza_center_y))
        elif num_pieces <= 6:
            # Single ring
            for i in range(num_pieces):
                angle = (2 * math.pi * i / num_pieces) + random.uniform(-0.1, 0.1)
                radius = self.pizza_radius * 0.6
                x = self.pizza_center_x + radius * math.cos(angle)
                y = self.pizza_center_y + radius * math.sin(angle)
                positions.append((x, y))
        else:
            # Two rings
            inner_count = num_pieces // 2
            outer_count = num_pieces - inner_count

            # Inner ring
            for i in range(inner_count):
                angle = (2 * math.pi * i / inner_count) + random.uniform(-0.1, 0.1)
                radius = self.pizza_radius * 0.4
                x = self.pizza_center_x + radius * math.cos(angle)
                y = self.pizza_center_y + radius * math.sin(angle)
                positions.append((x, y))

            # Outer ring
            for i in range(outer_count):
                angle = (2 * math.pi * i / outer_count) + random.uniform(-0.1, 0.1)
                radius = self.pizza_radius * 0.75
                x = self.pizza_center_x + radius * math.cos(angle)
                y = self.pizza_center_y + radius * math.sin(angle)
                positions.append((x, y))

        return positions

    def mark_pizza_complete(self):
        """Mark pizza as complete"""
        self.pizza_status_label.config(
            text=f"Pizza Complete! {self.pizza_name} is ready!",
            fg=COLOR_SUCCESS
        )

        # Add sparkles
        sparkle_positions = [
            (self.pizza_center_x - 150, self.pizza_center_y - 150),
            (self.pizza_center_x + 150, self.pizza_center_y - 150),
            (self.pizza_center_x - 150, self.pizza_center_y + 150),
            (self.pizza_center_x + 150, self.pizza_center_y + 150),
        ]

        for x, y in sparkle_positions:
            self.pizza_canvas.create_text(
                x, y, text="✨", font=("Arial", 25), fill=COLOR_SECONDARY
            )

    # =========================================================================
    # END PIZZA MAKER ANIMATION METHODS
    # =========================================================================

    def on_hide(self):
        """Called when screen is hidden"""
        # Stop camera feed when leaving the screen
        self.stop_camera()

    def go_home(self):
        """Go back to home screen"""
        self.stop_camera()
        self.controller.show_frame("HomeScreen")

    def reset(self):
        """Reset the screen"""
        self.pizza_name = None
        self.ingredients_list = []
        self.current_index = 0
        self.is_running = False

        self.pizza_label.configure(text="No order")
        self.progress_var.set(0)
        self.progress_text.configure(text="0 / 0 ingredients")
        self.status_label.configure(text="Ready", fg=COLOR_SUCCESS)
        self.action_label.configure(text="Waiting to start...")

        # Clear log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)

        # Clear checklist
        for widget in self.checklist_frame.winfo_children():
            widget.destroy()