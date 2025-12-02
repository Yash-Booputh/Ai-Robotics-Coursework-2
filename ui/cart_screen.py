"""
ChefMate Robot Assistant - Cart Screen
Order confirmation and checkout
"""

import tkinter as tk
from tkinter import ttk, messagebox

from config.settings import (
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_BG_DARK, COLOR_BG_MEDIUM,
    COLOR_TEXT_LIGHT, FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE,
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
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)

        # Header
        header_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        header_frame.pack(fill=tk.X, pady=(0, 30))

        title_label = tk.Label(
            header_frame,
            text="🛒 Your Order",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack()

        # Order details container
        details_frame = tk.Frame(
            main_frame,
            bg=COLOR_BG_MEDIUM,
            relief=tk.RAISED,
            borderwidth=2
        )
        details_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        # Pizza info section
        self.pizza_info_frame = tk.Frame(details_frame, bg=COLOR_BG_MEDIUM)
        self.pizza_info_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        # Create initial "no order" message
        self.no_order_label = tk.Label(
            self.pizza_info_frame,
            text="No pizza in cart",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_MEDIUM,
            fg="#7F8C8D"
        )
        self.no_order_label.pack(expand=True)

        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        buttons_frame.pack(fill=tk.X, pady=(20, 0))

        # Back button
        back_btn = tk.Button(
            buttons_frame,
            text="← Back to Menu",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT,
            command=self.go_back,
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2"
        )
        back_btn.pack(side=tk.LEFT)

        # Place order button
        self.order_btn = tk.Button(
            buttons_frame,
            text="Place Order & Start Robot 🚀",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_SUCCESS,
            fg=COLOR_TEXT_LIGHT,
            command=self.place_order,
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.order_btn.pack(side=tk.RIGHT)

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
                font=(FONT_FAMILY, FONT_SIZE_LARGE),
                bg=COLOR_BG_MEDIUM,
                fg="#7F8C8D"
            )
            self.no_order_label.pack(expand=True)
            self.order_btn.configure(state=tk.DISABLED)
            return

        pizza_data = PIZZA_RECIPES[self.pizza_order]

        # Pizza name
        name_label = tk.Label(
            self.pizza_info_frame,
            text=pizza_data["name"],
            font=(FONT_FAMILY, 24, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        name_label.pack(pady=(0, 10))

        # Description
        desc_label = tk.Label(
            self.pizza_info_frame,
            text=pizza_data["description"],
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_MEDIUM,
            fg="#BDC3C7",
            wraplength=600
        )
        desc_label.pack(pady=(0, 20))

        # Divider
        tk.Frame(
            self.pizza_info_frame,
            bg="#7F8C8D",
            height=2
        ).pack(fill=tk.X, pady=20)

        # Ingredients section
        ing_title = tk.Label(
            self.pizza_info_frame,
            text="Robot will pick these ingredients:",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        ing_title.pack(pady=(0, 15))

        # Ingredients list
        ingredients_frame = tk.Frame(self.pizza_info_frame, bg=COLOR_BG_MEDIUM)
        ingredients_frame.pack()

        for idx, ingredient in enumerate(pizza_data["ingredients"], 1):
            ing_row = tk.Frame(ingredients_frame, bg=COLOR_BG_MEDIUM)
            ing_row.pack(fill=tk.X, pady=5)

            # Number
            num_label = tk.Label(
                ing_row,
                text=f"{idx}.",
                font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
                bg=COLOR_BG_MEDIUM,
                fg=COLOR_PRIMARY,
                width=3
            )
            num_label.pack(side=tk.LEFT)

            # Ingredient name
            ing_label = tk.Label(
                ing_row,
                text=get_ingredient_display_name(ingredient),
                font=(FONT_FAMILY, FONT_SIZE_LARGE),
                bg=COLOR_BG_MEDIUM,
                fg=COLOR_TEXT_LIGHT
            )
            ing_label.pack(side=tk.LEFT, padx=10)

        # Divider
        tk.Frame(
            self.pizza_info_frame,
            bg="#7F8C8D",
            height=2
        ).pack(fill=tk.X, pady=20)

        # Price
        price_frame = tk.Frame(self.pizza_info_frame, bg=COLOR_BG_MEDIUM)
        price_frame.pack()

        price_label_text = tk.Label(
            price_frame,
            text="Total:",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            bg=COLOR_BG_MEDIUM,
            fg=COLOR_TEXT_LIGHT
        )
        price_label_text.pack(side=tk.LEFT, padx=(0, 20))

        price_label = tk.Label(
            price_frame,
            text=pizza_data["price"],
            font=(FONT_FAMILY, 28, "bold"),
            bg=COLOR_BG_MEDIUM,
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
            # Go to robot execution screen
            self.controller.start_robot_execution(self.pizza_order)

    def go_back(self):
        """Go back to menu"""
        self.controller.show_frame("MenuScreen")

    def reset_cart(self):
        """Reset the cart"""
        self.pizza_order = None
        self.display_order()