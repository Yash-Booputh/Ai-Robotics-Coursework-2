"""
ChefMate Robot Assistant - Pizza Recipes
Pizza recipes and ingredient definitions
"""

# ============================================================================
# PIZZA RECIPES
# ============================================================================

PIZZA_RECIPES = {
    "Margherita": {
        "name": "Margherita",
        "description": "Classic Italian pizza with tomato, cheese and fresh basil",
        "ingredients": ["fresh_tomato", "cheese", "basil"],
        "price": "$12.99",
        "image": "margherita.jpg"  # Place this in assets/pizza_images/
    },

    "Chicken Supreme": {
        "name": "Chicken Supreme",
        "description": "Delicious chicken with tomato and melted cheese",
        "ingredients": ["fresh_tomato", "cheese", "chicken"],
        "price": "$14.99",
        "image": "chicken_supreme.jpg"
    },

    "Seafood Delight": {
        "name": "Seafood Delight",
        "description": "Ocean fresh shrimp with tomato and cheese",
        "ingredients": ["fresh_tomato", "cheese", "shrimp"],
        "price": "$16.99",
        "image": "seafood_delight.jpg"
    },

    "Anchovy Special": {
        "name": "Anchovy Special",
        "description": "Mediterranean style with anchovies, tomato and cheese",
        "ingredients": ["fresh_tomato", "cheese", "anchovies"],
        "price": "$13.99",
        "image": "anchovy_special.jpg"
    },

    "Pesto Chicken": {
        "name": "Pesto Chicken",
        "description": "Grilled chicken with fresh basil and cheese",
        "ingredients": ["chicken", "cheese", "basil"],
        "price": "$15.99",
        "image": "pesto_chicken.jpg"
    }
}

# ============================================================================
# INGREDIENT INFORMATION
# ============================================================================

INGREDIENT_INFO = {
    "anchovies": {
        "display_name": "Anchovies",
        "description": "Salted anchovies",
        "image": "anchovies.jpg",  # Place in assets/ingredient_images/
        "color": "#8B4513"  # Brown
    },

    "basil": {
        "display_name": "Fresh Basil",
        "description": "Italian basil leaves",
        "image": "basil.jpg",
        "color": "#228B22"  # Green
    },

    "cheese": {
        "display_name": "Mozzarella Cheese",
        "description": "Fresh mozzarella",
        "image": "cheese.jpg",
        "color": "#FFD700"  # Gold/Yellow
    },

    "chicken": {
        "display_name": "Grilled Chicken",
        "description": "Tender chicken pieces",
        "image": "chicken.jpg",
        "color": "#DEB887"  # Burlywood
    },

    "fresh_tomato": {
        "display_name": "Fresh Tomato",
        "description": "Ripe tomatoes",
        "image": "tomato.jpg",
        "color": "#FF6347"  # Tomato red
    },

    "shrimp": {
        "display_name": "Shrimp",
        "description": "Fresh shrimp",
        "image": "shrimp.jpg",
        "color": "#FFA07A"  # Light salmon
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pizza_list():
    """Get list of all available pizzas"""
    return list(PIZZA_RECIPES.keys())


def get_pizza_info(pizza_name):
    """Get information about a specific pizza"""
    return PIZZA_RECIPES.get(pizza_name, None)


def get_pizza_ingredients(pizza_name):
    """Get list of ingredients for a specific pizza"""
    pizza = PIZZA_RECIPES.get(pizza_name, None)
    if pizza:
        return pizza["ingredients"]
    return []


def get_ingredient_display_name(ingredient_id):
    """Get display name for an ingredient"""
    ingredient = INGREDIENT_INFO.get(ingredient_id, None)
    if ingredient:
        return ingredient["display_name"]
    return ingredient_id.replace("_", " ").title()


def validate_pizza_name(pizza_name):
    """Check if pizza name exists"""
    return pizza_name in PIZZA_RECIPES


def get_total_ingredients_needed(pizza_name):
    """Get count of ingredients needed for a pizza"""
    ingredients = get_pizza_ingredients(pizza_name)
    return len(ingredients)