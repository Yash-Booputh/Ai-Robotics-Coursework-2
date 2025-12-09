"""
ChefMate Robot Assistant - Animated Pizza Making Visualization
Interactive, game-like storytelling pizza assembly with narration
"""

import tkinter as tk
from tkinter import ttk
import math
import random
import os
import time
from PIL import Image, ImageTk

try:
    from config.settings import (
        COLOR_PRIMARY, COLOR_SUCCESS, COLOR_BG_DARK,
        FONT_FAMILY, FONT_SIZE_HEADER, FONT_SIZE_LARGE, FONT_SIZE_NORMAL
    )
    from config.recipes import PIZZA_RECIPES, get_ingredient_display_name
except ImportError:
    # Fallback values if config not available
    COLOR_PRIMARY = "#2196F3"
    COLOR_SUCCESS = "#4CAF50"
    COLOR_BG_DARK = "#263238"
    FONT_FAMILY = "Arial"
    FONT_SIZE_HEADER = 24
    FONT_SIZE_LARGE = 16
    FONT_SIZE_NORMAL = 12
   
    PIZZA_RECIPES = {
        "Margherita": {
            "name": "Margherita",
            "ingredients": ["fresh_tomato", "cheese", "basil"]
        }
    }
   
    def get_ingredient_display_name(name):
        return name.replace("_", " ").title()


class AnimatedPizzaMaker(tk.Toplevel):
    """
    Animated pizza making visualization window
    Game-like storytelling with narration and step-by-step ingredient adding
    """

    def __init__(self, parent, pizza_name):
        """
        Initialize the animation window

        Args:
            parent: Parent window
            pizza_name: Name of pizza being made
        """
        super().__init__(parent)

        self.pizza_name = pizza_name
        self.pizza_data = PIZZA_RECIPES.get(pizza_name, {})
        self.ingredients = self.pizza_data.get("ingredients", [])

        # Window setup
        self.title(f"🍕 Making {pizza_name}...")
        self.geometry("1200x850")
        self.configure(bg="#2C1810")  # Dark brown background
        self.resizable(False, False)

        # Image directories
        self.pizza_base_dir = "assets/pizza_bases"
        self.ingredient_dir = "assets/ingredients"

        # Animation state
        self.current_ingredient_index = -1
        self.has_cheese = False
        self.placed_ingredients = []  # Track placed ingredient positions
        self.animation_complete = False
        self.is_closed = False

        # Canvas parameters (pizza in center-left, story on right)
        self.canvas_width = 700
        self.canvas_height = 700
        self.pizza_center_x = 350
        self.pizza_center_y = 350
        self.pizza_radius = 160  # Actual pizza radius (toppings must stay inside)

        # Load images
        self.images = {}
        self.load_images()

        self.create_widgets()
        self.draw_initial_pizza()
       
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def find_image_file(self, directory, base_name):
        """Find image file with either .png or .jpg extension"""
        if not os.path.exists(directory):
            return None
       
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            filename = base_name + ext
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                try:
                    with Image.open(path) as test_img:
                        test_img.verify()
                    return path
                except:
                    continue
        return None

    def load_images(self):
        """Load all pizza base and ingredient images"""
        # Pizza base images
        base_images = {
            'dough_sauce': 'dough_with_sauce',
            'dough_cheese': 'dough_with_cheese'
        }
       
        pizza_base_size = (340, 340)  # Slightly larger than radius*2
       
        for key, base_name in base_images.items():
            path = self.find_image_file(self.pizza_base_dir, base_name)
            if path:
                try:
                    with Image.open(path) as img:
                        if img.mode == 'RGBA':
                            img_resized = img.resize(pizza_base_size, Image.Resampling.LANCZOS)
                        elif img.mode == 'P':
                            img_converted = img.convert('RGBA')
                            img_resized = img_converted.resize(pizza_base_size, Image.Resampling.LANCZOS)
                        else:
                            img_converted = img.convert('RGB')
                            img_resized = img_converted.resize(pizza_base_size, Image.Resampling.LANCZOS)
                       
                        self.images[key] = ImageTk.PhotoImage(img_resized)
                except Exception as e:
                    print(f"Error loading {base_name}: {e}")
       
        # Ingredient images
        ingredient_mappings = {
            'fresh_tomato': 'tomato',
            'basil': 'basil',
            'anchovies': 'anchovies',
            'chicken': 'chicken',
            'shrimp': 'shrimp'
        }
       
        ingredient_size = (70, 70)  # Smaller for more pieces
       
        for key, base_name in ingredient_mappings.items():
            path = self.find_image_file(self.ingredient_dir, base_name)
            if path:
                try:
                    with Image.open(path) as img:
                        if img.mode == 'RGBA':
                            img_resized = img.resize(ingredient_size, Image.Resampling.LANCZOS)
                        elif img.mode == 'P':
                            img_converted = img.convert('RGBA')
                            img_resized = img_converted.resize(ingredient_size, Image.Resampling.LANCZOS)
                        else:
                            if img.mode != 'RGB':
                                img_converted = img.convert('RGB')
                            else:
                                img_converted = img
                            rgba_img = Image.new('RGBA', img_converted.size, (255, 255, 255, 255))
                            rgba_img.paste(img_converted, (0, 0))
                            img_resized = rgba_img.resize(ingredient_size, Image.Resampling.LANCZOS)
                       
                        self.images[key] = ImageTk.PhotoImage(img_resized)
                except Exception as e:
                    print(f"Error loading {base_name}: {e}")

    def create_widgets(self):
        """Create all UI widgets"""
        # Main container
        main_container = tk.Frame(self, bg="#2C1810")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Title bar
        title_frame = tk.Frame(main_container, bg="#8B0000", height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text=f"🍕 Pizzaiolo's Kitchen: {self.pizza_name} 🍕",
            font=(FONT_FAMILY, 24, "bold"),
            bg="#8B0000",
            fg="white"
        )
        title_label.pack(pady=15)

        # Content area (pizza + story)
        content_frame = tk.Frame(main_container, bg="#2C1810")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side - Pizza canvas
        pizza_frame = tk.Frame(content_frame, bg="#2C1810")
        pizza_frame.pack(side=tk.LEFT, padx=(0, 20))

        canvas_label = tk.Label(
            pizza_frame,
            text="🔥 The Pizza Oven 🔥",
            font=(FONT_FAMILY, 16, "bold"),
            bg="#2C1810",
            fg="#FFD700"
        )
        canvas_label.pack(pady=(0, 10))

        self.canvas = tk.Canvas(
            pizza_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#1A0F08",
            highlightthickness=3,
            highlightbackground="#8B4513"
        )
        self.canvas.pack()

        # Right side - Recipe panel (Simplified)
        recipe_frame = tk.Frame(content_frame, bg="#3D2817", width=450,
                               relief=tk.RAISED, borderwidth=3)
        recipe_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        recipe_frame.pack_propagate(False)

        # Recipe title
        recipe_title = tk.Label(
            recipe_frame,
            text="📋 Recipe Progress",
            font=(FONT_FAMILY, 18, "bold"),
            bg="#3D2817",
            fg="#FFD700"
        )
        recipe_title.pack(pady=15)

        # Ingredient checklist
        self.ingredient_labels = {}
        ingredients_list_frame = tk.Frame(recipe_frame, bg="#3D2817")
        ingredients_list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
       
        # Follow exact ingredient array order from recipe
        for i, ingredient in enumerate(self.ingredients, 1):
            display_name = get_ingredient_display_name(ingredient)
            ing_frame = tk.Frame(ingredients_list_frame, bg="#3D2817")
            ing_frame.pack(fill=tk.X, pady=8)
           
            checkbox = tk.Label(
                ing_frame,
                text="☐",
                font=(FONT_FAMILY, 16),
                bg="#3D2817",
                fg="#999",
                width=2
            )
            checkbox.pack(side=tk.LEFT)
           
            ing_label = tk.Label(
                ing_frame,
                text=f"{i}. {display_name}",
                font=(FONT_FAMILY, 14),
                bg="#3D2817",
                fg="#CCC",
                anchor=tk.W
            )
            ing_label.pack(side=tk.LEFT, padx=10)
           
            self.ingredient_labels[ingredient] = (checkbox, ing_label)

        # Separator
        tk.Frame(recipe_frame, bg="#8B4513", height=2).pack(fill=tk.X, padx=20, pady=15)

        # Status text area
        status_frame = tk.Frame(recipe_frame, bg="#3D2817")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        status_title = tk.Label(
            status_frame,
            text="🔔 Status Updates",
            font=(FONT_FAMILY, 14, "bold"),
            bg="#3D2817",
            fg="#FFA500",
            anchor=tk.W
        )
        status_title.pack(fill=tk.X, pady=(0, 10))

        self.status_text = tk.Text(
            status_frame,
            font=(FONT_FAMILY, 11),
            bg="#2C1810",
            fg="#FFE4B5",
            wrap=tk.WORD,
            height=8,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.status_text.config(state=tk.DISABLED)

        # Bottom bar
        bottom_frame = tk.Frame(main_container, bg="#1A0F08", height=70, relief=tk.RAISED, borderwidth=2)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)

        # Status label
        self.status_label = tk.Label(
            bottom_frame,
            text="🧑‍🍳 Preparing the dough...",
            font=(FONT_FAMILY, 14, "bold"),
            bg="#1A0F08",
            fg="#FFD700"
        )
        self.status_label.pack(side=tk.LEFT, padx=30, pady=20)

        # Close button
        self.close_btn = tk.Button(
            bottom_frame,
            text="Close",
            font=(FONT_FAMILY, 12, "bold"),
            bg="#666",
            fg="white",
            command=self.close_window,
            relief=tk.FLAT,
            padx=30,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.close_btn.pack(side=tk.RIGHT, padx=30, pady=15)

    def draw_initial_pizza(self):
        """Draw the initial pizza base (dough with sauce)"""
        # Oven background
        self.canvas.create_oval(
            self.pizza_center_x - 200,
            self.pizza_center_y - 200,
            self.pizza_center_x + 200,
            self.pizza_center_y + 200,
            fill="#3D2817",
            outline="#8B4513",
            width=2,
            tags="oven"
        )

        # Pizza stone/board
        self.canvas.create_oval(
            self.pizza_center_x - 180,
            self.pizza_center_y - 180,
            self.pizza_center_x + 180,
            self.pizza_center_y + 180,
            fill="#D2B48C",
            outline="#8B7355",
            width=3,
            tags="stone"
        )

        # Pizza base - start with sauce only
        if 'dough_sauce' in self.images:
            self.pizza_base_image = self.canvas.create_image(
                self.pizza_center_x,
                self.pizza_center_y,
                image=self.images['dough_sauce'],
                tags="pizza_base"
            )
        else:
            self.pizza_base_image = self.canvas.create_oval(
                self.pizza_center_x - 170,
                self.pizza_center_y - 170,
                self.pizza_center_x + 170,
                self.pizza_center_y + 170,
                fill="#DC143C",
                outline="#8B0000",
                width=4,
                tags="pizza_base"
            )

        # Ensure proper layering
        self.canvas.tag_lower("pizza_base")
        self.canvas.tag_lower("stone")
        self.canvas.tag_lower("oven")

    def add_status_update(self, message):
        """Add a status update to the text area"""
        self.status_text.config(state=tk.NORMAL)
       
        if self.status_text.get("1.0", "end-1c") != "":
            self.status_text.insert(tk.END, "\n\n")
       
        self.status_text.insert(tk.END, f"• {message}")
        self.status_text.config(state=tk.DISABLED)
        self.status_text.see(tk.END)

    def get_circular_position(self, ring_number, position_in_ring, total_in_ring):
        """
        Get position in a circular ring pattern
       
        Args:
            ring_number: Which ring (0=center, 1=inner, 2=middle, 3=outer)
            position_in_ring: Position in this ring (0 to total_in_ring-1)
            total_in_ring: Total positions in this ring
           
        Returns:
            Tuple (x, y) for placement
        """
        if ring_number == 0:
            # Center position
            return self.pizza_center_x, self.pizza_center_y
       
        # Calculate radius for this ring (distribute evenly within pizza)
        max_radius = self.pizza_radius - 25  # Leave margin from edge
        ring_radius = (max_radius / 3) * ring_number
       
        # Calculate angle for this position
        angle = (2 * math.pi * position_in_ring / total_in_ring)
        # Add random offset for natural look
        angle += random.uniform(-0.1, 0.1)
       
        # Calculate position
        x = self.pizza_center_x + ring_radius * math.cos(angle)
        y = self.pizza_center_y + ring_radius * math.sin(angle)
       
        return x, y

    def distribute_ingredients_circular(self, num_pieces):
        """
        Distribute ingredients in circular rings pattern
       
        Args:
            num_pieces: Number of pieces to place
           
        Returns:
            List of (x, y) positions
        """
        positions = []
       
        if num_pieces <= 1:
            positions.append((self.pizza_center_x, self.pizza_center_y))
        elif num_pieces <= 6:
            # Single ring
            for i in range(num_pieces):
                x, y = self.get_circular_position(1, i, num_pieces)
                positions.append((x, y))
        elif num_pieces <= 12:
            # Two rings: center + inner ring
            positions.append((self.pizza_center_x, self.pizza_center_y))
            for i in range(num_pieces - 1):
                x, y = self.get_circular_position(2, i, num_pieces - 1)
                positions.append((x, y))
        else:
            # Three rings
            inner_count = min(6, num_pieces // 3)
            middle_count = min(8, (num_pieces - inner_count) // 2)
            outer_count = num_pieces - inner_count - middle_count
           
            # Inner ring
            for i in range(inner_count):
                x, y = self.get_circular_position(1, i, inner_count)
                positions.append((x, y))
           
            # Middle ring
            for i in range(middle_count):
                x, y = self.get_circular_position(2, i, middle_count)
                positions.append((x, y))
           
            # Outer ring
            for i in range(outer_count):
                x, y = self.get_circular_position(3, i, outer_count)
                positions.append((x, y))
       
        return positions

    def add_ingredient(self, ingredient_name):
        """
        Add an ingredient to the pizza
        """
        if self.is_closed:
            return

        self.current_ingredient_index += 1
        display_name = get_ingredient_display_name(ingredient_name)

        # Update status
        self.status_label.config(text=f"🤖 Adding: {display_name}...")
        self.add_status_update(f"Adding {display_name}...")

        # Check off in recipe list
        if ingredient_name in self.ingredient_labels:
            checkbox, label = self.ingredient_labels[ingredient_name]
            checkbox.config(text="☑", fg="#4CAF50")
            label.config(fg="#4CAF50", font=(FONT_FAMILY, 14, "bold"))

        # Add the ingredient
        if ingredient_name == "cheese":
            self.add_cheese()
        else:
            self.add_ingredient_pieces(ingredient_name)

        self.update()

    def add_cheese(self):
        """Add cheese by switching to dough_with_cheese image"""
        if self.has_cheese:
            return
       
        self.has_cheese = True
       
        if 'dough_cheese' in self.images:
            self.canvas.delete("pizza_base")
            self.pizza_base_image = self.canvas.create_image(
                self.pizza_center_x,
                self.pizza_center_y,
                image=self.images['dough_cheese'],
                tags="pizza_base"
            )
            # Ensure proper layering
            self.canvas.tag_lower("pizza_base")
            self.canvas.tag_lower("stone")
            self.canvas.tag_lower("oven")
            self.add_status_update("Cheese layer added - pizza base updated!")
        else:
            self.canvas.create_oval(
                self.pizza_center_x - 165,
                self.pizza_center_y - 165,
                self.pizza_center_x + 165,
                self.pizza_center_y + 165,
                fill="#FFF8DC",
                outline="#FFD700",
                width=2,
                tags="cheese_layer"
            )
            self.canvas.tag_lower("cheese_layer")

    def add_ingredient_pieces(self, ingredient_name):
        """Add multiple pieces of an ingredient in circular pattern"""
        if ingredient_name not in self.images:
            return
       
        # Determine number of pieces
        counts = {
            'fresh_tomato': 9,
            'basil': 7,
            'anchovies': 8,
            'chicken': 10,
            'shrimp': 9
        }
       
        num_pieces = counts.get(ingredient_name, 8)
       
        # Get circular positions
        positions = self.distribute_ingredients_circular(num_pieces)
       
        # Place each piece
        for i, (x, y) in enumerate(positions):
            self.canvas.create_image(
                x, y,
                image=self.images[ingredient_name],
                tags=f"ingredient_{ingredient_name}"
            )
            self.placed_ingredients.append((x, y))
           
            # Brief pause for animation effect (only for first few)
            if i < 3:
                self.update()
                time.sleep(0.1)

    def mark_complete(self):
        """Mark pizza as complete with celebration"""
        if self.is_closed:
            return

        self.animation_complete = True
       
        self.status_label.config(
            text="✅ Pizza Complete! Bellissimo! 🇮🇹",
            fg="#4CAF50"
        )
       
        # Final status
        self.add_status_update(
            f"Your {self.pizza_name} is ready! "
            f"All {len(self.ingredients)} ingredients added successfully. "
            f"Buon appetito!"
        )
       
        # Enable close button
        self.close_btn.config(state=tk.NORMAL, bg="#4CAF50")
       
        # Add sparkles
        sparkle_positions = [
            (self.pizza_center_x - 220, self.pizza_center_y - 220),
            (self.pizza_center_x + 220, self.pizza_center_y - 220),
            (self.pizza_center_x - 220, self.pizza_center_y + 220),
            (self.pizza_center_x + 220, self.pizza_center_y + 220),
        ]
       
        for x, y in sparkle_positions:
            self.canvas.create_text(
                x, y, text="✨", font=("Arial", 40), fill="#FFD700"
            )

    def close_window(self):
        """Close the animation window"""
        self.is_closed = True
        self.destroy()

    def get_current_index(self):
        """Get current ingredient index"""
        return self.current_ingredient_index

    def is_complete(self):
        """Check if animation is complete"""
        return self.animation_complete


# ============================================================================
# DEMO / TESTING
# ============================================================================

def demo_animation():
    """Demo the animation system standalone"""
    root = tk.Tk()
    root.withdraw()
   
    test_pizza = "Margherita"
   
    if test_pizza not in PIZZA_RECIPES:
        print(f"\nError: {test_pizza} not found in recipes")
        return
   
    anim = AnimatedPizzaMaker(root, test_pizza)
   
    ingredients = PIZZA_RECIPES[test_pizza]["ingredients"]
   
    def add_next_ingredient(index=0):
        if index < len(ingredients):
            print(f"\nAdding ingredient {index + 1}/{len(ingredients)}: {ingredients[index]}")
            anim.add_ingredient(ingredients[index])
            root.after(3000, lambda: add_next_ingredient(index + 1))
        else:
            print("\nAll ingredients added!")
            anim.mark_complete()
   
    root.after(1000, add_next_ingredient)
   
    root.mainloop()


if __name__ == "__main__":
    print("\n🍕 ANIMATED PIZZA MAKER - Game-Like Storytelling Version")
    print("="*70)
    demo_animation()