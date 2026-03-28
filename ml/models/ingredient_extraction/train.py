import pandas as pd
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# Add ml/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_DATA_DIR, SAVED_MODELS_DIR

# Paths
DATA_PATH = RAW_DATA_DIR / "recipe1m" / "recipes.csv"
MODEL_DIR = SAVED_MODELS_DIR / "ingredient_extraction"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load recipes dataset"""
    print("📂 Loading Food.com recipes dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} recipes")

    df = df[[
        "Name",
        "RecipeCategory",
        "RecipeIngredientParts",
    ]].dropna(subset=["Name", "RecipeIngredientParts"])

    print(f"✅ After cleaning: {len(df)} recipes")
    return df


def parse_ingredients(ingredient_str):
    """Parse ingredient string into clean list"""
    if not isinstance(ingredient_str, str):
        return []

    # Remove c() wrapper
    ingredient_str = re.sub(r'^c\(', '', ingredient_str)
    ingredient_str = re.sub(r'\)$', '', ingredient_str)

    ingredients = []
    for item in ingredient_str.split('",'):
        item = item.strip().strip('"').strip("'").lower()

        # Remove quantities and measurements
        item = re.sub(
            r'^\d+[\d./]*\s*'
            r'(cup|cups|tbsp|tsp|oz|lb|g|kg|ml|l|pound|ounce|'
            r'tablespoon|teaspoon|clove|cloves|slice|slices|'
            r'piece|pieces|can|cans|package|pkg|bunch|head|'
            r'large|medium|small|fresh|dried|chopped|minced|'
            r'diced|sliced|whole|ground|grated|shredded)*\s*',
            '', item
        )

        # Remove special characters
        item = re.sub(r'[^a-z\s]', '', item).strip()

        if len(item) > 2:
            ingredients.append(item)

    return ingredients


def build_dish_ingredient_lookup(df):
    """
    Build a lookup dictionary:
    keyword → list of common ingredients
    """
    print("\n🔤 Building dish-ingredient lookup...")

    # Build keyword → ingredients mapping
    keyword_ingredients = defaultdict(list)

    for _, row in df.iterrows():
        dish_name = str(row["Name"]).lower()
        ingredients = parse_ingredients(row["RecipeIngredientParts"])

        if not ingredients:
            continue

        # Extract keywords from dish name
        words = re.findall(r'\b[a-z]{3,}\b', dish_name)
        for word in words:
            keyword_ingredients[word].extend(ingredients)

    # For each keyword, keep top 10 most common ingredients
    lookup = {}
    for keyword, ings in keyword_ingredients.items():
        if len(ings) >= 5:
            from collections import Counter
            counter = Counter(ings)
            lookup[keyword] = [
                ing for ing, _ in counter.most_common(10)
            ]

    print(f"✅ Built lookup with {len(lookup)} keywords")
    return lookup


def build_global_common_ingredients(df, top_n=200):
    """Build list of globally common ingredients"""
    print("\n📊 Building global ingredient list...")

    from collections import Counter
    all_ingredients = []

    for ing_str in df["RecipeIngredientParts"]:
        ingredients = parse_ingredients(ing_str)
        all_ingredients.extend(ingredients)

    counter = Counter(all_ingredients)
    common = [ing for ing, _ in counter.most_common(top_n)]

    print(f"✅ Top {top_n} common ingredients identified")
    print(f"   Examples: {common[:10]}")
    return common


def build_category_ingredients(df):
    """Build category → common ingredients mapping"""
    print("\n📂 Building category-ingredient mapping...")

    from collections import Counter
    category_map = defaultdict(list)

    for _, row in df.iterrows():
        category = str(row.get("RecipeCategory", "")).lower().strip()
        if not category or category == "nan":
            continue

        ingredients = parse_ingredients(row["RecipeIngredientParts"])
        if ingredients:
            category_map[category].extend(ingredients)

    # Keep top 15 ingredients per category
    result = {}
    for cat, ings in category_map.items():
        counter = Counter(ings)
        result[cat] = [
            ing for ing, _ in counter.most_common(15)
        ]

    print(f"✅ Built mapping for {len(result)} categories")
    return result


def save_model(lookup, common_ingredients, category_map):
    """Save all lookup data"""
    print("\n💾 Saving model data...")

    with open(MODEL_DIR / "keyword_lookup.json", "w") as f:
        json.dump(lookup, f, indent=2)
    print("✅ Keyword lookup saved")

    with open(MODEL_DIR / "common_ingredients.json", "w") as f:
        json.dump(common_ingredients, f, indent=2)
    print("✅ Common ingredients saved")

    with open(MODEL_DIR / "category_ingredients.json", "w") as f:
        json.dump(category_map, f, indent=2)
    print("✅ Category ingredients saved")

    model_info = {
        "model_type": "keyword_lookup",
        "keywords": len(lookup),
        "common_ingredients": len(common_ingredients),
        "categories": len(category_map),
        "version": "2.0"
    }
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print("✅ Model info saved")


def main():
    print("=" * 50)
    print("🥗 Ingredient Extraction Model Training")
    print("=" * 50)

    # Load data
    df = load_data()

    # Build lookup tables
    lookup = build_dish_ingredient_lookup(df)
    common_ingredients = build_global_common_ingredients(df)
    category_map = build_category_ingredients(df)

    # Save
    save_model(lookup, common_ingredients, category_map)

    print("\n" + "=" * 50)
    print("✅ Ingredient Extraction Training Complete!")
    print(f"   Keywords: {len(lookup)}")
    print(f"   Common ingredients: {len(common_ingredients)}")
    print("=" * 50)


if __name__ == "__main__":
    main()