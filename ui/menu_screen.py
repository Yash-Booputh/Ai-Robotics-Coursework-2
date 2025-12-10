"""
ChefMate Robot Assistant - Menu Screen
Pizza menu with cards showing available pizzas
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os

from config.settings import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS, COLOR_BG_DARK,
    COLOR_BG_MEDIUM, COLOR_BG_LIGHT, COLOR_TEXT_LIGHT, COLOR_TEXT_DARK,
    COLOR_TEXT_GRAY, COLOR_TEXT_MUTED, FONT_FAMILY, FONT_SIZE_HEADER,
    FONT_SIZE_LARGE, FONT_SIZE_NORMAL, PIZZA_IMAGES_DIR
)
from config.recipes import PIZZA_RECIPES, get_pizza_ingredients, get_ingredient_display_name


class MenuScreen(ttk.Frame):
    """
    Pizza menu screen displaying available pizzas as cards
    User can select ONE pizza and proceed to cart
    """

    def __init__(self, parent, controller):
        """
        Initialize menu screen

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')

        self.selected_pizza = None
        self.pizza_cards = []

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""
        # Main container with scrollbar
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        header_frame.pack(fill=tk.X, padx=20, pady=20)

        # Back button
        back_btn = tk.Button(
            header_frame,
            text="← Back",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_TEXT_GRAY,
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
            text="🍕 Select Your Pizza",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack(side=tk.LEFT, padx=20)

        # Info label
        info_label = tk.Label(
            header_frame,
            text="(Select ONE pizza - Limited ingredients available)",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_DARK,
            fg=COLOR_TEXT_GRAY
        )
        info_label.pack(side=tk.LEFT)

        # Canvas with scrollbar for pizza cards
        canvas_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        canvas = tk.Canvas(canvas_frame, bg=COLOR_BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLOR_BG_DARK)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)

        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create pizza cards (3 columns, 2 rows)
        pizzas = list(PIZZA_RECIPES.items())
        row_frame = None

        for idx, (pizza_name, pizza_data) in enumerate(pizzas):
            if idx % 3 == 0:  # Changed from 2 to 3 columns
                row_frame = tk.Frame(scrollable_frame, bg=COLOR_BG_DARK)
                row_frame.pack(fill=tk.X, pady=10)

            card = self.create_pizza_card(row_frame, pizza_name, pizza_data)
            card.pack(side=tk.LEFT, padx=10, expand=True)
            self.pizza_cards.append((pizza_name, card))

        # Bottom bar with cart button
        bottom_frame = tk.Frame(main_frame, bg=COLOR_BG_LIGHT, height=80, relief=tk.SOLID, borderwidth=1)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)

        # Selected pizza label
        self.selected_label = tk.Label(
            bottom_frame,
            text="No pizza selected",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        )
        self.selected_label.pack(side=tk.LEFT, padx=20)

        # Add to cart button
        self.cart_btn = tk.Button(
            bottom_frame,
            text="Add to Cart →",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            command=self.add_to_cart,
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.cart_btn.pack(side=tk.RIGHT, padx=20)

    def create_pizza_card(self, parent, pizza_name, pizza_data):
        """
        Create a pizza card widget

        Args:
            parent: Parent widget
            pizza_name: Name of pizza
            pizza_data: Pizza data dictionary

        Returns:
            Card frame
        """
        # Card container with subtle shadow effect
        card_frame = tk.Frame(
            parent,
            bg=COLOR_BG_LIGHT,
            width=400,
            height=500,
            relief=tk.SOLID,
            borderwidth=1,
            highlightbackground=COLOR_BG_MEDIUM,
            highlightthickness=1
        )
        card_frame.pack_propagate(False)

        # Image section
        image_frame = tk.Frame(card_frame, bg=COLOR_BG_LIGHT, height=250)
        image_frame.pack(fill=tk.X)
        image_frame.pack_propagate(False)

        # Try to load pizza image
        image_path = os.path.join(PIZZA_IMAGES_DIR, pizza_data["image"])
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                img = img.resize((380, 240), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                img_label = tk.Label(image_frame, image=photo, bg=COLOR_BG_LIGHT)
                img_label.image = photo  # Keep reference
                img_label.pack(pady=5)
            except Exception as e:
                self.create_placeholder_image(image_frame, "🍕")
        else:
            self.create_placeholder_image(image_frame, "🍕")

        # Info section
        info_frame = tk.Frame(card_frame, bg=COLOR_BG_LIGHT)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Pizza name
        name_label = tk.Label(
            info_frame,
            text=pizza_data["name"],
            font=(FONT_FAMILY, 16, "bold"),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        name_label.pack(anchor="w")

        # Description
        desc_label = tk.Label(
            info_frame,
            text=pizza_data["description"],
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY,
            wraplength=370,
            justify=tk.LEFT
        )
        desc_label.pack(anchor="w", pady=(5, 10))

        # Ingredients
        ingredients_text = "Ingredients: " + ", ".join([
            get_ingredient_display_name(ing) for ing in pizza_data["ingredients"]
        ])
        ing_label = tk.Label(
            info_frame,
            text=ingredients_text,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_MUTED,
            wraplength=370,
            justify=tk.LEFT
        )
        ing_label.pack(anchor="w", pady=(0, 10))

        # Price
        price_label = tk.Label(
            info_frame,
            text=pizza_data["price"],
            font=(FONT_FAMILY, 18, "bold"),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_SUCCESS
        )
        price_label.pack(anchor="w", pady=(5, 10))

        # Select button
        select_btn = tk.Button(
            info_frame,
            text="Select This Pizza",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_PRIMARY,
            fg=COLOR_TEXT_LIGHT,
            command=lambda: self.select_pizza(pizza_name, card_frame),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        select_btn.pack(fill=tk.X)

        # Store button reference
        card_frame.select_btn = select_btn

        return card_frame

    def create_placeholder_image(self, parent, emoji):
        """Create placeholder image with emoji"""
        placeholder = tk.Label(
            parent,
            text=emoji,
            font=("Arial", 80),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_MUTED
        )
        placeholder.pack(expand=True)

    def select_pizza(self, pizza_name, card_frame):
        """
        Select a pizza

        Args:
            pizza_name: Name of selected pizza
            card_frame: Card frame widget
        """
        # Deselect previous
        if self.selected_pizza:
            for name, card in self.pizza_cards:
                if name == self.selected_pizza:
                    card.configure(borderwidth=1, relief=tk.SOLID, highlightbackground=COLOR_BG_MEDIUM)
                    card.select_btn.configure(text="Select This Pizza", bg=COLOR_PRIMARY)

        # Select new
        self.selected_pizza = pizza_name
        card_frame.configure(borderwidth=3, relief=tk.SOLID, highlightbackground=COLOR_PRIMARY)
        card_frame.select_btn.configure(text="✓ Selected", bg=COLOR_SUCCESS)

        # Update bottom bar
        self.selected_label.configure(text=f"Selected: {pizza_name}", fg=COLOR_TEXT_DARK)
        self.cart_btn.configure(state=tk.NORMAL)

    def add_to_cart(self):
        """Add selected pizza to cart"""
        if not self.selected_pizza:
            messagebox.showwarning("No Selection", "Please select a pizza first")
            return

        # Pass selected pizza to cart
        self.controller.set_pizza_order(self.selected_pizza)
        self.controller.show_frame("CartScreen")

    def go_back(self):
        """Go back to home screen"""
        self.controller.show_frame("HomeScreen")

    def reset_selection(self):
        """Reset pizza selection"""
        self.selected_pizza = None
        for name, card in self.pizza_cards:
            card.configure(borderwidth=1, relief=tk.SOLID, highlightbackground=COLOR_BG_MEDIUM)
            card.select_btn.configure(text="Select This Pizza", bg=COLOR_PRIMARY)
        self.selected_label.configure(text="No pizza selected", fg=COLOR_TEXT_GRAY)
        self.cart_btn.configure(state=tk.DISABLED)