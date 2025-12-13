"""
ChefMate Robot Assistant - File Upload Screen
Upload images for ingredient detection testing (IRIS-style modern design)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from .widgets import ModernButton
from config.settings import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_PURPLE,
    COLOR_BG_DARK, COLOR_BG_LIGHT, COLOR_BG_MEDIUM, COLOR_TEXT_DARK, COLOR_TEXT_GRAY,
    COLOR_TEXT_LIGHT, FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL, SUPPORTED_IMAGE_FORMATS
)


class FileUploadScreen(ttk.Frame):
    """
    File upload screen for testing detection
    Allows uploading single images or folders with navigation
    """

    def __init__(self, parent, controller):
        """
        Initialize file upload screen

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')

        # Image management
        self.image_files = []
        self.current_index = 0
        self.processed_images = {}  # Cache processed images

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets - modern IRIS style"""
        # Main container with proper sizing
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Header - more compact
        header = tk.Frame(main_frame, bg=COLOR_BG_LIGHT)
        header.pack(fill=tk.X, padx=0, pady=(0, 5))

        title = tk.Label(
            header,
            text="File Upload - Detection Testing",
            font=(FONT_FAMILY, 16, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_PURPLE
        )
        title.pack(pady=8)

        # Content area with 2-column layout
        content_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT panel - Upload controls with scrollbar 
        left_panel_container = tk.Frame(content_frame, bg=COLOR_BG_LIGHT, width=300, relief=tk.RAISED, borderwidth=1)
        left_panel_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel_container.pack_propagate(False)

        # Add canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_panel_container, bg=COLOR_BG_LIGHT, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_panel_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_panel = tk.Frame(left_canvas, bg=COLOR_BG_LIGHT, width=280)

        # Configure scroll region update
        def update_scroll_region(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        left_panel.bind("<Configure>", update_scroll_region)

        left_canvas.create_window((0, 0), window=left_panel, anchor="nw", width=280)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_panel.bind_all("<MouseWheel>", _on_mousewheel)

        upload_title = tk.Label(
            left_panel,
            text="Upload Options",
            font=(FONT_FAMILY, 13, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_PURPLE
        )
        upload_title.pack(pady=10)

        # Upload buttons
        self.upload_frame = tk.Frame(left_panel, bg=COLOR_BG_LIGHT)
        self.upload_frame.pack(pady=3, padx=10, fill=tk.X)

        ModernButton(
            self.upload_frame,
            text="SELECT IMAGE",
            command=self.upload_single_image,
            bg=COLOR_PRIMARY
        ).pack(pady=3, fill=tk.X)

        tk.Label(
            self.upload_frame,
            text="Upload single image for detection",
            font=(FONT_FAMILY, 8),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        ).pack(pady=3)

        ModernButton(
            self.upload_frame,
            text="SELECT FOLDER",
            command=self.upload_folder,
            bg=COLOR_SUCCESS
        ).pack(pady=3, fill=tk.X)

        tk.Label(
            self.upload_frame,
            text="Upload folder with multiple images",
            font=(FONT_FAMILY, 8),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        ).pack(pady=3)

        # Separator
        tk.Frame(left_panel, bg=COLOR_BG_MEDIUM, height=1).pack(fill=tk.X, padx=15, pady=8)

        # Navigation controls placeholder
        self.nav_container = tk.Frame(left_panel, bg=COLOR_BG_LIGHT)
        self.nav_container.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Separator
        tk.Frame(left_panel, bg=COLOR_BG_MEDIUM, height=1).pack(fill=tk.X, padx=15, pady=8)

        # Detection result 
        result_title = tk.Label(
            left_panel,
            text="Detection Result",
            font=(FONT_FAMILY, 11, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        result_title.pack(pady=(0, 3))

        self.result_frame = tk.Frame(left_panel, bg=COLOR_BG_MEDIUM, relief=tk.SUNKEN, borderwidth=2)
        self.result_frame.pack(fill=tk.X, padx=15, pady=(0, 5))

        self.result_label = tk.Label(
            self.result_frame,
            text="No detection yet",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_GRAY,
            wraplength=270
        )
        self.result_label.pack(pady=8)

        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_GRAY,
            wraplength=270
        )
        self.confidence_label.pack(pady=(0, 8))

        # Separator
        tk.Frame(left_panel, bg=COLOR_BG_MEDIUM, height=1).pack(fill=tk.X, padx=15, pady=8)

        # Info section 
        info_frame = tk.Frame(left_panel, bg=COLOR_BG_LIGHT)
        info_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        info_title = tk.Label(
            info_frame,
            text="Information",
            font=(FONT_FAMILY, 10, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        info_title.pack(anchor="w", pady=(0, 5))

        info_text = (
            "• Upload images to test detection\n"
            "• Formats: JPG, PNG, BMP, TIFF, WEBP\n"
            "• Bounding boxes shown on images\n"
            "• Use navigation for multiple images"
        )

        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=(FONT_FAMILY, 8),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY,
            justify=tk.LEFT,
            wraplength=270
        )
        info_label.pack(anchor="w")

        # RIGHT panel - Image display (EXPANDS)
        right_panel = tk.Frame(content_frame, bg=COLOR_BG_LIGHT, relief=tk.RAISED, borderwidth=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        display_title = tk.Label(
            right_panel,
            text="Image Preview with Detection",
            font=(FONT_FAMILY, 14, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        display_title.pack(pady=8)

        # Filename label
        self.filename_label = tk.Label(
            right_panel,
            text="",
            font=(FONT_FAMILY, 9),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        )
        self.filename_label.pack(pady=(0, 3))

        # Image display container
        display_container = tk.Frame(right_panel, bg=COLOR_TEXT_GRAY, relief=tk.SUNKEN, borderwidth=2)
        display_container.pack(padx=10, pady=(5, 10), fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            display_container,
            bg=COLOR_TEXT_GRAY,
            text="No image loaded\n\nUpload an image or folder to see detection results",
            font=(FONT_FAMILY, 12),
            fg=COLOR_TEXT_LIGHT
        )
        self.image_label.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

    def create_navigation_controls(self):
        """Create or update navigation controls"""
        # Clear existing navigation controls
        for widget in self.nav_container.winfo_children():
            widget.destroy()

        tk.Label(
            self.nav_container,
            text="Navigate Images",
            font=(FONT_FAMILY, 11, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        ).pack(pady=(0, 5))

        nav_buttons_frame = tk.Frame(self.nav_container, bg=COLOR_BG_LIGHT)
        nav_buttons_frame.pack(fill=tk.X, pady=5)

        self.prev_btn = ModernButton(
            nav_buttons_frame,
            text="← PREVIOUS",
            command=self.previous_image,
            bg=COLOR_SECONDARY
        )
        self.prev_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.next_btn = ModernButton(
            nav_buttons_frame,
            text="NEXT →",
            command=self.next_image,
            bg=COLOR_SECONDARY
        )
        self.next_btn.pack(side=tk.RIGHT, padx=2, fill=tk.X, expand=True)

        self.image_counter_label = tk.Label(
            self.nav_container,
            text="0 / 0",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        self.image_counter_label.pack(pady=5)

    def upload_single_image(self):
        """Upload and process single image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.image_files = [file_path]
            self.current_index = 0
            self.processed_images = {}
            # Clear navigation for single image
            for widget in self.nav_container.winfo_children():
                widget.destroy()
            self.process_current_image()

    def upload_folder(self):
        """Upload and process folder of images"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")

        if folder_path:
            # Find all images in folder (only this folder, not subdirectories)
            image_files = []
            try:
                files = os.listdir(folder_path)
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    # Only process files (not directories)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in SUPPORTED_IMAGE_FORMATS:
                            image_files.append(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read folder:\n{str(e)}")
                return

            if not image_files:
                messagebox.showinfo("No Images", "No supported images found in folder")
                return

            # Store images and process first one
            self.image_files = sorted(image_files)  # Sort for consistent order
            self.current_index = 0
            self.processed_images = {}

            messagebox.showinfo(
                "Folder Upload",
                f"Found {len(image_files)} images\n\nUse navigation buttons to browse through them."
            )

            # Show navigation controls
            if len(self.image_files) > 1:
                self.create_navigation_controls()

            self.process_current_image()

    def process_current_image(self):
        """Process the current image"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        file_path = self.image_files[self.current_index]

        # Check if already processed
        if file_path in self.processed_images:
            annotated_image, detections = self.processed_images[file_path]
            self.display_image(annotated_image, detections, os.path.basename(file_path))
            self.update_navigation()
            return

        try:
            # Read image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", f"Failed to load image:\n{os.path.basename(file_path)}")
                return

            # Run detection for ALL ingredients
            annotated_image, detections = self.controller.detect_all_in_image(image)

            # Cache the result
            self.processed_images[file_path] = (annotated_image, detections)

            # Display result
            self.display_image(annotated_image, detections, os.path.basename(file_path))
            self.update_navigation()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

    def previous_image(self):
        """Navigate to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.process_current_image()

    def next_image(self):
        """Navigate to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.process_current_image()

    def update_navigation(self):
        """Update navigation button states and counter"""
        if len(self.image_files) <= 1 or not hasattr(self, 'image_counter_label'):
            return

        # Update counter
        self.image_counter_label.configure(
            text=f"{self.current_index + 1} / {len(self.image_files)}"
        )

        # Enable/disable buttons
        if hasattr(self, 'prev_btn') and hasattr(self, 'next_btn'):
            if self.current_index == 0:
                self.prev_btn.config(state=tk.DISABLED)
            else:
                self.prev_btn.config(state=tk.NORMAL)

            if self.current_index == len(self.image_files) - 1:
                self.next_btn.config(state=tk.DISABLED)
            else:
                self.next_btn.config(state=tk.NORMAL)

    def display_image(self, image, detections, filename):
        """
        Display processed image with detections

        Args:
            image: Processed image with bounding boxes (BGR)
            detections: List of detection dictionaries
            filename: Image filename
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to fit display while maintaining aspect ratio (max 1000x650)
            h, w = image_rgb.shape[:2]
            max_w, max_h = 1000, 650

            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to PhotoImage
            img_pil = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(img_pil)

            # Update image label
            self.image_label.configure(image=photo, text="", bg=COLOR_TEXT_GRAY)
            self.image_label.image = photo

            # Update filename
            self.filename_label.configure(text=f"File: {filename}")

            # Update result label - show summary of all detections
            if detections and len(detections) > 0:
                # Create summary text
                if len(detections) == 1:
                    detection = detections[0]
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    result_text = f"{class_name.replace('_', ' ').title()}"
                    conf_text = f"Confidence: {confidence:.1%}"

                    if confidence >= 0.8:
                        color = COLOR_SUCCESS
                        conf_text += " (Excellent)"
                    elif confidence >= 0.6:
                        color = COLOR_SECONDARY
                        conf_text += " (Good)"
                    else:
                        color = COLOR_DANGER
                        conf_text += " (Low)"
                else:
                    # Multiple detections - show count and list
                    result_text = f"Found {len(detections)} ingredients"
                    color = COLOR_SUCCESS

                    # List all detected ingredients with confidence
                    ingredients_list = []
                    for det in detections:
                        name = det['class_name'].replace('_', ' ').title()
                        conf = det['confidence']
                        ingredients_list.append(f"• {name} ({conf:.0%})")

                    conf_text = "\n".join(ingredients_list)

                self.result_label.configure(text=result_text, fg=color)
                self.confidence_label.configure(text=conf_text, fg=color)
            else:
                self.result_label.configure(text="No ingredient detected", fg=COLOR_TEXT_GRAY)
                self.confidence_label.configure(text="", fg=COLOR_TEXT_GRAY)

        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image:\n{str(e)}")
