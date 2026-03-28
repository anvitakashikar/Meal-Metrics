import json
import re
import sys
from pathlib import Path

# Add ml/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import SAVED_MODELS_DIR

# Paths
MODEL_DIR = SAVED_MODELS_DIR / "ingredient_extraction"
KEYWORD_LOOKUP_PATH = MODEL_DIR / "keyword_lookup.json"
COMMON_INGREDIENTS_PATH = MODEL_DIR / "common_ingredients.json"
CATEGORY_INGREDIENTS_PATH = MODEL_DIR / "category_ingredients.json"


def load_model():
    """Load lookup data"""
    with open(KEYWORD_LOOKUP_PATH, "r") as f:
        keyword_lookup = json.load(f)
    with open(COMMON_INGREDIENTS_PATH, "r") as f:
        common_ingredients = json.load(f)
    with open(CATEGORY_INGREDIENTS_PATH, "r") as f:
        category_ingredients = json.load(f)
    return keyword_lookup, common_ingredients, category_ingredients


def extract_ingredients(meal_text, category=None, top_n=10):
    """
    Extract ingredients from meal description

    Args:
        meal_text: string describing the meal
        category: optional meal category
        top_n: number of ingredients to return

    Returns:
        list of extracted ingredients
    """
    keyword_lookup, common_ingredients, category_ingredients = load_model()

    meal_text = meal_text.lower().strip()

    # Extract keywords from meal text
    words = re.findall(r'\b[a-z]{3,}\b', meal_text)

    # Collect matching ingredients
    matched = []
    for word in words:
        if word in keyword_lookup:
            matched.extend(keyword_lookup[word])

    # Add category ingredients if available
    if category:
        cat = category.lower().strip()
        if cat in category_ingredients:
            matched.extend(category_ingredients[cat])

    # If no matches found use common ingredients
    if not matched:
        return common_ingredients[:top_n]

    # Count and return most common matches
    from collections import Counter
    counter = Counter(matched)
    ingredients = [
        ing for ing, _ in counter.most_common(top_n)
    ]

    return ingredients


def main():
    """Test ingredient extraction"""
    print("=" * 50)
    print("🥗 Ingredient Extraction Predictor")
    print("=" * 50)

    test_meals = [
        ("Grilled chicken with rice and vegetables", None),
        ("Spaghetti pasta with tomato sauce", "pasta"),
        ("Caesar salad with croutons", "salads"),
        ("Chocolate cake with vanilla frosting", "desserts"),
        ("Vegetable stir fry with tofu", "asian")
    ]

    for meal, category in test_meals:
        print(f"\n🍽️  Input: {meal}")
        if category:
            print(f"   Category: {category}")

        ingredients = extract_ingredients(meal, category)
        print(f"   Extracted ingredients:")
        for ing in ingredients:
            print(f"   - {ing}")


if __name__ == "__main__":
    main()