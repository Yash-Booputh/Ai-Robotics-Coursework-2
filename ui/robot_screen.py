"""
ChefMate Robot Assistant - Robot Execution Screen
Monitor robot picking ingredients with live camera feed
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import time

from config.settings import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING,
    COLOR_BG_DARK, COLOR_BG_MEDIUM, COLOR_TEXT_LIGHT,
    FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL
)
from config.recipes import get_pizza_ingredients, get_ingredient_display_name


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

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""
        # Main container
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top section - Status and controls
        top_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK, height=100)
        top_frame.pack(fill=tk.X, padx=20, pady=20)
        top_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(
            top_frame,
            text="🤖 Robot Execution",
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
            text="⏹ Stop",
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

        # Middle section - Camera feed and progress
        middle_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # Left: Camera feed
        camera_frame = tk.Frame(middle_frame, bg=COLOR_BG_MEDIUM, relief=tk.RAISED, borderwidth=2)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        camera_title = tk.Label(
            camera_frame,
            text="📷 Live Camera Feed",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        camera_title.pack(pady=10)

        # Camera display
        self.camera_label = tk.Label(
            camera_frame,
            bg=COLOR_BG_DARK,
            width=640,
            height=480
        )
        self.camera_label.pack(padx=10, pady=(0, 10))

        # FPS label
        self.fps_label = tk.Label(
            camera_frame,
            text="FPS: 0",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg="#7F8C8D"
        )
        self.fps_label.pack(pady=(0, 10))

        # Right: Progress panel
        progress_frame = tk.Frame(middle_frame, bg=COLOR_BG_MEDIUM, width=350, relief=tk.RAISED, borderwidth=2)
        progress_frame.pack(side=tk.RIGHT, fill=tk.Y)
        progress_frame.pack_propagate(False)

        progress_title = tk.Label(
            progress_frame,
            text="📋 Progress",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        progress_title.pack(pady=10)

        # Pizza name
        self.pizza_label = tk.Label(
            progress_frame,
            text="No order",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        self.pizza_label.pack(pady=(0, 20))

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)

        # Progress text
        self.progress_text = tk.Label(
            progress_frame,
            text="0 / 0 ingredients",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        self.progress_text.pack(pady=(0, 20))

        # Ingredients checklist
        checklist_label = tk.Label(
            progress_frame,
            text="Ingredients:",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
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
            wraplength=320,
            justify=tk.LEFT
        )
        self.action_label.pack(pady=20, padx=10)

        # Bottom section - Log
        log_frame = tk.Frame(main_frame, bg=COLOR_BG_MEDIUM, height=150, relief=tk.RAISED, borderwidth=2)
        log_frame.pack(fill=tk.X, padx=20, pady=20)
        log_frame.pack_propagate(False)

        log_title = tk.Label(
            log_frame,
            text="📄 Activity Log",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        log_title.pack(anchor="w", padx=10, pady=(5, 0))

        # Log text widget with scrollbar
        log_container = tk.Frame(log_frame, bg=COLOR_BG_MEDIUM)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        log_scrollbar = ttk.Scrollbar(log_container)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_container,
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT_LIGHT,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            height=6,
            yscrollcommand=log_scrollbar.set,
            state=tk.DISABLED
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.log_text.yview)

        # Bottom buttons
        bottom_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        bottom_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.home_btn = tk.Button(
            bottom_frame,
            text="🏠 Home",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT,
            command=self.go_home,
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.home_btn.pack(side=tk.LEFT)

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
        self.home_btn.configure(state=tk.DISABLED)

        # Create checklist
        self.create_checklist()

        # Start camera
        self.start_camera()

        # Log start
        self.add_log(f"🚀 Starting order: {pizza_name}")
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
                text="⭕",
                font=(FONT_FAMILY, 14),
                bg=COLOR_BG_MEDIUM,
                fg="#7F8C8D",
                width=2
            )
            status_label.pack(side=tk.LEFT)

            name_label = tk.Label(
                item_frame,
                text=get_ingredient_display_name(ingredient),
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                bg=COLOR_BG_MEDIUM,
                fg=COLOR_TEXT_LIGHT,
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
                self.add_log("✅ Order completed successfully!")
                self.status_label.configure(text="Completed", fg=COLOR_SUCCESS)
                messagebox.showinfo("Success", f"{self.pizza_name} is ready! 🎉")
            else:
                self.add_log("❌ Order failed")
                self.status_label.configure(text="Failed", fg=COLOR_DANGER)
                messagebox.showerror("Error", "Order execution failed")

        except Exception as e:
            self.add_log(f"❌ Error: {str(e)}")
            self.status_label.configure(text="Error", fg=COLOR_DANGER)
            messagebox.showerror("Error", f"Execution error: {str(e)}")

        finally:
            self.is_running = False
            self.stop_btn.configure(state=tk.DISABLED)
            self.home_btn.configure(state=tk.NORMAL)

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

        # Update checklist if ingredient picked
        if "Picked" in message or "delivered" in message:
            for idx, (status_label, name_label) in enumerate(self.checklist_items):
                if idx < self.current_index:
                    status_label.configure(text="✅", fg=COLOR_SUCCESS)

            # Update progress
            self.current_index += 1
            progress = (self.current_index / len(self.ingredients_list)) * 100
            self.progress_var.set(progress)
            self.progress_text.configure(text=f"{self.current_index} / {len(self.ingredients_list)} ingredients")

    def start_camera(self):
        """Start camera feed"""
        if self.camera_active:
            return

        self.camera_active = True
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        """Camera feed loop"""
        while self.camera_active and self.is_running:
            try:
                # Get frame from vision system
                frame, detection = self.controller.get_camera_frame()

                if frame is not None:
                    # Convert to PhotoImage
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((640, 480), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)

                    # Update camera label
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo

                    # Update FPS
                    fps = self.controller.get_camera_fps()
                    self.fps_label.configure(text=f"FPS: {fps:.1f}")

                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)

        self.camera_active = False

    def stop_camera(self):
        """Stop camera feed"""
        self.camera_active = False

    def stop_execution(self):
        """Stop robot execution"""
        response = messagebox.askyesno(
            "Stop Execution",
            "Are you sure you want to stop?\nThe robot will return to home position.",
            icon="warning"
        )

        if response:
            self.add_log("⏹ Stopping execution...")
            self.is_running = False
            self.controller.stop_pick_sequence()
            self.status_label.configure(text="Stopped", fg=COLOR_WARNING)
            self.stop_btn.configure(state=tk.DISABLED)
            self.home_btn.configure(state=tk.NORMAL)

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