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
        # Main container - reduced padding
        main_frame = tk.Frame(self, bg=COLOR_BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Header - more compact
        header_frame = tk.Frame(main_frame, bg=COLOR_BG_DARK)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = tk.Label(
            header_frame,
            text="🛒 Your Order",
            font=(FONT_FAMILY, 22, "bold"),  # Reduced font size
            bg=COLOR_BG_DARK,
            fg=COLOR_PRIMARY
        )
        title_label.pack()

        # Order details container - white card with proportional sizing
        details_frame = tk.Frame(
            main_frame,
            bg="#FFFFFF",
            relief=tk.SOLID,
            borderwidth=1,
            highlightbackground="#e0e0e0",
            highlightthickness=1
        )
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Pizza info section - reduced padding
        self.pizza_info_frame = tk.Frame(details_frame, bg="#FFFFFF")
        self.pizza_info_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)

        # Create initial "no order" message
        self.no_order_label = tk.Label(
            self.pizza_info_frame,
            text="No pizza in cart",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg="#FFFFFF",
            fg="#9E9E9E"  # Light gray
        )
        self.no_order_label.pack(expand=True)

        # Buttons frame - compact and clean
        buttons_frame = tk.Frame(main_frame, bg="#FFFFFF", relief=tk.SOLID, borderwidth=1)
        buttons_frame.pack(fill=tk.X, pady=0, ipady=10)

        # Back button - compact
        back_btn = tk.Button(
            buttons_frame,
            text="← Back to Menu",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg="#616161",  # Darker gray for better visibility
            fg="#FFFFFF",  # White text
            activebackground="#424242",
            activeforeground="#FFFFFF",
            command=self.go_back,
            relief=tk.RAISED,
            borderwidth=2,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        back_btn.pack(side=tk.LEFT, padx=15, pady=5)

        # Place order button - compact
        self.order_btn = tk.Button(
            buttons_frame,
            text="Place Order & Start Robot 🚀",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            bg=COLOR_SUCCESS,
            fg="#FFFFFF",
            activebackground="#388E3C",
            activeforeground="#FFFFFF",
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
                bg="#FFFFFF",
                fg="#9E9E9E"  # Light gray
            )
            self.no_order_label.pack(expand=True)
            self.order_btn.configure(state=tk.DISABLED)
            return

        pizza_data = PIZZA_RECIPES[self.pizza_order]

        # Pizza name - more compact
        name_label = tk.Label(
            self.pizza_info_frame,
            text=pizza_data["name"],
            font=(FONT_FAMILY, 20, "bold"),  # Reduced size
            bg="#FFFFFF",
            fg="#212121"  # Dark gray
        )
        name_label.pack(pady=(0, 8))

        # Description - more compact
        desc_label = tk.Label(
            self.pizza_info_frame,
            text=pizza_data["description"],
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),  # Reduced size
            bg="#FFFFFF",
            fg="#616161",  # Medium gray
            wraplength=600
        )
        desc_label.pack(pady=(0, 12))

        # Divider - thinner
        tk.Frame(
            self.pizza_info_frame,
            bg="#E0E0E0",  # Light gray divider
            height=1
        ).pack(fill=tk.X, pady=12)

        # Ingredients section - more compact
        ing_title = tk.Label(
            self.pizza_info_frame,
            text="Robot will pick these ingredients:",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),  # Reduced size
            bg="#FFFFFF",
            fg="#212121"  # Dark gray
        )
        ing_title.pack(pady=(0, 10))

        # Ingredients list - more compact
        ingredients_frame = tk.Frame(self.pizza_info_frame, bg="#FFFFFF")
        ingredients_frame.pack()

        for idx, ingredient in enumerate(pizza_data["ingredients"], 1):
            ing_row = tk.Frame(ingredients_frame, bg="#FFFFFF")
            ing_row.pack(fill=tk.X, pady=3)  # Reduced spacing

            # Number
            num_label = tk.Label(
                ing_row,
                text=f"{idx}.",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),  # Reduced size
                bg="#FFFFFF",
                fg=COLOR_PRIMARY,
                width=3
            )
            num_label.pack(side=tk.LEFT)

            # Ingredient name
            ing_label = tk.Label(
                ing_row,
                text=get_ingredient_display_name(ingredient),
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),  # Reduced size
                bg="#FFFFFF",
                fg="#424242"  # Dark gray
            )
            ing_label.pack(side=tk.LEFT, padx=8)

        # Divider
        tk.Frame(
            self.pizza_info_frame,
            bg="#E0E0E0",  # Light gray divider
            height=1
        ).pack(fill=tk.X, pady=12)

        # Price - more compact
        price_frame = tk.Frame(self.pizza_info_frame, bg="#FFFFFF")
        price_frame.pack()

        price_label_text = tk.Label(
            price_frame,
            text="Total:",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),  # Reduced size
            bg="#FFFFFF",
            fg="#424242"  # Dark gray
        )
        price_label_text.pack(side=tk.LEFT, padx=(0, 15))

        price_label = tk.Label(
            price_frame,
            text=pizza_data["price"],
            font=(FONT_FAMILY, 22, "bold"),  # Reduced size
            bg="#FFFFFF",
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