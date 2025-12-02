"""
ChefMate Robot Assistant - File Upload Screen
Upload images for ingredient detection testing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from config.settings import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_BG_DARK, COLOR_BG_MEDIUM,
    COLOR_TEXT_LIGHT, FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL, SUPPORTED_IMAGE_FORMATS
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
            text="📁 File Upload - Test Detection",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack(side=tk.LEFT, padx=20)

        # Content area
        content_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Left: Upload buttons
        left_frame = tk.Frame(content_frame, bg=COLOR_BG_MEDIUM, width=300, relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        upload_title = tk.Label(
            left_frame,
            text="Upload Options",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        upload_title.pack(pady=20)

        # Upload single image button
        upload_img_btn = tk.Button(
            left_frame,
            text="📷 Upload Single Image",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_PRIMARY,
            fg=COLOR_TEXT_LIGHT,
            command=self.upload_single_image,
            relief=tk.FLAT,
            padx=20,
            pady=15,
            cursor="hand2"
        )
        upload_img_btn.pack(fill=tk.X, padx=20, pady=10)

        # Upload folder button
        upload_folder_btn = tk.Button(
            left_frame,
            text="📁 Upload Folder",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            command=self.upload_folder,
            relief=tk.FLAT,
            padx=20,
            pady=15,
            cursor="hand2"
        )
        upload_folder_btn.pack(fill=tk.X, padx=20, pady=10)

        # Info section
        info_frame = tk.Frame(left_frame, bg=COLOR_BG_MEDIUM)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        info_title = tk.Label(
            info_frame,
            text="ℹ️ Information",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
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
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_MEDIUM,
            fg="#BDC3C7",
            justify=tk.LEFT,
            wraplength=250
        )
        info_label.pack(anchor="w")

        # Right: Image display
        right_frame = tk.Frame(content_frame, bg=COLOR_BG_MEDIUM, relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        display_title = tk.Label(
            right_frame,
            text="Image Preview",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        display_title.pack(pady=10)

        # Image display area
        self.image_label = tk.Label(
            right_frame,
            bg=COLOR_BG_DARK,
            text="No image loaded\n\nUpload an image to see detection",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            fg="#7F8C8D"
        )
        self.image_label.pack(expand=True, padx=20, pady=20)

        # Detection result
        self.result_frame = tk.Frame(right_frame, bg=COLOR_BG_MEDIUM)
        self.result_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        self.result_label.pack()

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
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

            # Update result label
            if detection:
                class_name = detection['class_name']
                confidence = detection['confidence']

                result_text = f"✅ Detected: {class_name}\nConfidence: {confidence:.1%}"
                color = COLOR_SUCCESS if confidence >= 0.7 else "#F39C12"
            else:
                result_text = "❌ No ingredient detected"
                color = "#7F8C8D"

            self.result_label.configure(text=result_text, fg=color)

            # Store current
            self.current_image = image
            self.current_detection = detection

        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image:\n{str(e)}")

    def go_back(self):
        """Go back to home screen"""
        self.controller.show_frame("HomeScreen")