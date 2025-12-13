"""
ChefMate Robot Assistant - Cart Screen
Order confirmation and checkout
"""

import tkinter as tk
from tkinter import ttk, messagebox

from config.settings import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_BG_DARK, COLOR_BG_MEDIUM, COLOR_BG_LIGHT,
    COLOR_TEXT_LIGHT, COLOR_TEXT_DARK, COLOR_TEXT_GRAY, COLOR_TEXT_MUTED, COLOR_INFO,
    BUTTON_SUCCESS_HOVER, FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL
)
from config.recipes import PIZZA_RECIPES, get_ingredient_display_name


class CartScreen(ttk.Frame):
    """
    Cart/Order confirmation screen
    Shows selected pizza and allows user to confirm order
    """

    def __init__(self, parent, controller):
        """
        Initialize cart screen

        Args:
            parent: Parent widget
            controller: Main application controller
        """
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Dark.TFrame')

        self.pizza_order = None
        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""
        # Main container 
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Header 
        header_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = tk.Label(
            header_frame,
            text="🛒 Your Order",
            font=(FONT_FAMILY, 22, "bold"), 
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack()

        # Order details container
        details_frame = tk.Frame(
            main_frame,
            bg=COLOR_BG_LIGHT,
            relief=tk.SOLID,
            borderwidth=1,
            highlightbackground=COLOR_BG_MEDIUM,
            highlightthickness=1
        )
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Pizza info section
        self.pizza_info_frame = tk.Frame(details_frame, bg=COLOR_BG_LIGHT)
        self.pizza_info_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)

        # Create initial 
        self.no_order_label = tk.Label(
            self.pizza_info_frame,
            text="No pizza in cart",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_MUTED
        )
        self.no_order_label.pack(expand=True)

        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg=COLOR_BG_LIGHT, relief=tk.SOLID, borderwidth=1)
        buttons_frame.pack(fill=tk.X, pady=0, ipady=10)

        # Back button - compact
        self.back_btn = tk.Button(
            buttons_frame,
            text="← Back to Menu",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_TEXT_GRAY,
            fg=COLOR_TEXT_LIGHT,
            activebackground=COLOR_INFO,
            activeforeground=COLOR_TEXT_LIGHT,
            command=self.go_back,
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.back_btn.pack(side=tk.LEFT, padx=15, pady=5)

        # Place order button
        self.order_btn = tk.Button(
            buttons_frame,
            text="Place Order & Start Robot 🚀",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            activebackground=BUTTON_SUCCESS_HOVER,
            activeforeground=COLOR_TEXT_LIGHT,
            command=self.place_order,
            relief=tk.RAISED,
            borderwidth=2,
            padx=25,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.order_btn.pack(side=tk.RIGHT, padx=15, pady=5)

    def set_order(self, pizza_name):
        """
        Set the pizza order

        Args:
            pizza_name: Name of ordered pizza
        """
        self.pizza_order = pizza_name
        self.display_order()

    def display_order(self):
        """Display the order details"""
        # Clear existing widgets
        for widget in self.pizza_info_frame.winfo_children():
            widget.destroy()

        if not self.pizza_order or self.pizza_order not in PIZZA_RECIPES:
            self.no_order_label = tk.Label(
                self.pizza_info_frame,
                text="No pizza in cart",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                bg=COLOR_BG_LIGHT,
                fg=COLOR_TEXT_MUTED
            )
            self.no_order_label.pack(expand=True)
            self.order_btn.configure(state=tk.DISABLED)
            self.back_btn.configure(text="← Back to Menu")
            return

        pizza_data = PIZZA_RECIPES[self.pizza_order]

        # Update back button text based on pizza type
        if self.pizza_order == "Chef Surprise":
            self.back_btn.configure(text="← Back to Home")
        else:
            self.back_btn.configure(text="← Back to Menu")

        # Pizza name 
        name_label = tk.Label(
            self.pizza_info_frame,
            text=pizza_data["name"],
            font=(FONT_FAMILY, 20, "bold"),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        name_label.pack(pady=(0, 8))

        # Description 
        desc_label = tk.Label(
            self.pizza_info_frame,
            text=pizza_data["description"],
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY,
            wraplength=600
        )
        desc_label.pack(pady=(0, 12))

        # Divider 
        tk.Frame(
            self.pizza_info_frame,
            bg=COLOR_BG_MEDIUM,
            height=1
        ).pack(fill=tk.X, pady=12)

        # Ingredients section 
        ing_title = tk.Label(
            self.pizza_info_frame,
            text="Robot will pick these ingredients:",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_DARK
        )
        ing_title.pack(pady=(0, 10))

        # Ingredients list
        ingredients_frame = tk.Frame(self.pizza_info_frame, bg=COLOR_BG_LIGHT)
        ingredients_frame.pack()

        for idx, ingredient in enumerate(pizza_data["ingredients"], 1):
            ing_row = tk.Frame(ingredients_frame, bg=COLOR_BG_LIGHT)
            ing_row.pack(fill=tk.X, pady=3)

            # Number
            num_label = tk.Label(
                ing_row,
                text=f"{idx}.",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
                bg=COLOR_BG_LIGHT,
                fg=COLOR_PRIMARY,
                width=3
            )
            num_label.pack(side=tk.LEFT)

            # Ingredient name
            ing_label = tk.Label(
                ing_row,
                text=get_ingredient_display_name(ingredient),
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                bg=COLOR_BG_LIGHT,
                fg=COLOR_TEXT_GRAY
            )
            ing_label.pack(side=tk.LEFT, padx=8)

        # Divider
        tk.Frame(
            self.pizza_info_frame,
            bg=COLOR_BG_MEDIUM,
            height=1
        ).pack(fill=tk.X, pady=12)

        # Price 
        price_frame = tk.Frame(self.pizza_info_frame, bg=COLOR_BG_LIGHT)
        price_frame.pack()

        price_label_text = tk.Label(
            price_frame,
            text="Total:",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_TEXT_GRAY
        )
        price_label_text.pack(side=tk.LEFT, padx=(0, 15))

        price_label = tk.Label(
            price_frame,
            text=pizza_data["price"],
            font=(FONT_FAMILY, 22, "bold"),
            bg=COLOR_BG_LIGHT,
            fg=COLOR_SUCCESS
        )
        price_label.pack(side=tk.LEFT)

        # Enable order button
        self.order_btn.configure(state=tk.NORMAL)

    def place_order(self):
        """Confirm and place the order"""
        if not self.pizza_order:
            messagebox.showwarning("No Order", "No pizza in cart")
            return

        # Confirmation dialog
        response = messagebox.askyesno(
            "Confirm Order",
            f"Place order for {self.pizza_order}?\n\n"
            f"The robot will start picking ingredients.",
            icon="question"
        )

        if response:
            # Play menu sound for the ordered pizza
            if hasattr(self.controller, 'audio') and self.controller.audio:
                self.controller.audio.play_pizza_sound(self.pizza_order)

            # Go to robot execution screen
            self.controller.start_robot_execution(self.pizza_order)

    def go_back(self):
        """Go back to menu or home screen depending on pizza type"""
        # If Chef Surprise, go back to home (standalone detector handles ordering)
        if self.pizza_order == "Chef Surprise":
            self.controller.show_frame("HomeScreen")
        else:
            self.controller.show_frame("MenuScreen")

    def reset_cart(self):
        """Reset the cart"""
        self.pizza_order = None
        self.display_order()