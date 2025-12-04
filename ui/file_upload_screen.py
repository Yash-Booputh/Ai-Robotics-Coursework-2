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
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_PURPLE, COLOR_BG_DARK, COLOR_BG_LIGHT,
    COLOR_BG_MEDIUM, COLOR_TEXT_DARK, COLOR_TEXT_GRAY, FONT_FAMILY,
    FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL, SUPPORTED_IMAGE_FORMATS
)


class FileUploadScreen(ttk.Frame):
    """
    File upload screen for testing detection
    Allows uploading single images or folders
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

        self.current_image = None
        self.current_detection = None

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
            text="File Upload - Detection Testing",
            font=(FONT_FAMILY, 18, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_PURPLE
        )
        title.pack(pady=12)

        # Content area with 2-column layout
        content_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT panel - Upload controls (FIXED WIDTH - 320px)
        left_panel = tk.Frame(content_frame, bg=COLOR_BG_LIGHT, width=320, relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)

        upload_title = tk.Label(
            left_panel,
            text="Upload Options",
            font=(FONT_FAMILY, 14, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_PURPLE
        )
        upload_title.pack(pady=20)

        # Upload buttons
        upload_frame = tk.Frame(left_panel, bg=COLOR_BG_LIGHT)
        upload_frame.pack(pady=6, padx=10, fill=tk.X)

        ModernButton(
            upload_frame,
            text="SELECT IMAGE",
            command=self.upload_single_image,
            bg=COLOR_PRIMARY
        ).pack(pady=3, fill=tk.X)

        tk.Label(
            upload_frame,
            text="Upload single image for detection",
            font=(FONT_FAMILY, 8),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        ).pack(pady=3)

        ModernButton(
            upload_frame,
            text="SELECT FOLDER",
            command=self.upload_folder,
            bg=COLOR_SUCCESS
        ).pack(pady=3, fill=tk.X)

        tk.Label(
            upload_frame,
            text="Upload folder with multiple images",
            font=(FONT_FAMILY, 8),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        ).pack(pady=3)

        # Separator
        tk.Frame(left_panel, bg=COLOR_BG_MEDIUM, height=1).pack(fill=tk.X, padx=15, pady=15)

        # Info section
        info_frame = tk.Frame(left_panel, bg=COLOR_BG_LIGHT)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        info_title = tk.Label(
            info_frame,
            text="Information",
            font=(FONT_FAMILY, 11, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        info_title.pack(anchor="w", pady=(0, 10))

        info_text = (
            "• Upload images to test ingredient detection\n\n"
            "• Supported formats:\n"
            "  JPG, PNG, BMP, TIFF, WEBP\n\n"
            "• Upload folder to process multiple images\n\n"
            "• Results show detected ingredient with confidence"
        )

        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=(FONT_FAMILY, 9),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY,
            justify=tk.LEFT,
            wraplength=280
        )
        info_label.pack(anchor="w")

        # RIGHT panel - Image display (EXPANDS)
        right_panel = tk.Frame(content_frame, bg=COLOR_BG_LIGHT, relief=tk.RAISED, borderwidth=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        display_title = tk.Label(
            right_panel,
            text="Image Preview",
            font=(FONT_FAMILY, 16, 'bold'),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        display_title.pack(pady=10)

        # Image display container
        display_container = tk.Frame(right_panel, bg=COLOR_BG_MEDIUM, relief=tk.SUNKEN, borderwidth=2)
        display_container.pack(padx=15, pady=(8, 15), fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            display_container,
            bg='#1a1a1a',
            text="No image loaded\n\nUpload an image to see detection",
            font=(FONT_FAMILY, 12),
            fg='#999999'
        )
        self.image_label.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # Detection result frame
        self.result_frame = tk.Frame(right_panel, bg='#e8f5e9', relief=tk.SUNKEN, borderwidth=2)
        self.result_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        tk.Label(
            self.result_frame,
            text="Detection Result",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg='#e8f5e9',
            fg=COLOR_TEXT_DARK
        ).pack(pady=(8, 3))

        self.result_label = tk.Label(
            self.result_frame,
            text="No detection yet",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg='#e8f5e9',
            fg=COLOR_TEXT_GRAY
        )
        self.result_label.pack(pady=(3, 8))

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
            self.process_image(file_path)

    def upload_folder(self):
        """Upload and process folder of images"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")

        if folder_path:
            # Find all images in folder
            image_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in SUPPORTED_IMAGE_FORMATS:
                        image_files.append(os.path.join(root, file))

            if not image_files:
                messagebox.showinfo("No Images", "No supported images found in folder")
                return

            # Process first image (or could create a slideshow)
            messagebox.showinfo(
                "Folder Upload",
                f"Found {len(image_files)} images\n\nShowing first image..."
            )
            self.process_image(image_files[0])

    def process_image(self, file_path):
        """
        Process uploaded image with detection

        Args:
            file_path: Path to image file
        """
        try:
            # Read image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Failed to load image")
                return

            # Run detection
            annotated_image, detection = self.controller.detect_in_image(image)

            # Display result
            self.display_image(annotated_image, detection, os.path.basename(file_path))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

    def display_image(self, image, detection, filename):
        """
        Display processed image with detection

        Args:
            image: Processed image (BGR)
            detection: Detection dictionary
            filename: Image filename
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to fit display (max 800x600)
            h, w = image_rgb.shape[:2]
            max_w, max_h = 800, 600

            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to PhotoImage
            img_pil = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(img_pil)

            # Update image label
            self.image_label.configure(image=photo, text="", bg='#1a1a1a')
            self.image_label.image = photo

            # Update result label
            if detection:
                class_name = detection['class_name']
                confidence = detection['confidence']

                result_text = f"Detected: {class_name.replace('_', ' ').title()}\nConfidence: {confidence:.1%}"

                if confidence >= 0.8:
                    color = COLOR_SUCCESS
                    status = " (Excellent)"
                elif confidence >= 0.6:
                    color = "#F39C12"
                    status = " (Good)"
                else:
                    color = "#E74C3C"
                    status = " (Low)"

                result_text += status
            else:
                result_text = "No ingredient detected"
                color = COLOR_TEXT_GRAY

            self.result_label.configure(text=result_text, fg=color)

            # Store current
            self.current_image = image
            self.current_detection = detection

        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image:\n{str(e)}")
