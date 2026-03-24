import requests
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
API_KEY = os.getenv("USDA_API_KEY")
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "usda"

# Food categories to download
FOOD_CATEGORIES = [
    "Dairy and Egg Products",
    "Spices and Herbs",
    "Fats and Oils",
    "Poultry Products",
    "Soups, Sauces, and Gravies",
    "Sausages and Luncheon Meats",
    "Breakfast Cereals",
    "Fruits and Fruit Juices",
    "Pork Products",
    "Vegetables and Vegetable Products",
    "Nut and Seed Products",
    "Beef Products",
    "Beverages",
    "Finfish and Shellfish Products",
    "Legumes and Legume Products",
    "Lamb, Veal, and Game Products",
    "Baked Products",
    "Sweets",
    "Cereal Grains and Pasta",
    "Fast Foods",
    "Meals, Entrees, and Side Dishes",
]

def create_output_dir():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")

def check_api_key():
    """Verify API key is loaded"""
    if not API_KEY:
        print("❌ USDA_API_KEY not found in .env file!")
        print("Please add USDA_API_KEY=your_key to ml/.env")
        return False
    print(f"✅ API Key loaded successfully")
    return True

def download_foods_by_category(category, page_size=50):
    """Download foods for a specific category"""
    print(f"\n📥 Downloading: {category}")
    
    url = f"{BASE_URL}/foods/search"
    params = {
    "api_key": API_KEY,
    "query": category,
    "dataType": "Foundation,SR Legacy",
    "pageSize": page_size,
    "pageNumber": 1
}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        foods = data.get("foods", [])
        print(f"   Found {len(foods)} foods")
        return foods
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error downloading {category}: {e}")
        return []

def extract_nutrition(food):
    """Extract relevant nutrition info from food item"""
    nutrients = {}
    for nutrient in food.get("foodNutrients", []):
        name = nutrient.get("nutrientName", "")
        value = nutrient.get("value", 0)
        unit = nutrient.get("unitName", "")
        
        if "Energy" in name:
            nutrients["calories"] = value
        elif "Protein" in name:
            nutrients["protein_g"] = value
        elif "Carbohydrate" in name and "by difference" in name:
            nutrients["carbohydrates_g"] = value
        elif "Total lipid" in name:
            nutrients["fat_g"] = value
        elif "Fiber" in name:
            nutrients["fiber_g"] = value
        elif "Sugars" in name and "total" in name.lower():
            nutrients["sugar_g"] = value
        elif "Sodium" in name:
            nutrients["sodium_mg"] = value
    
    return {
        "fdc_id": food.get("fdcId"),
        "description": food.get("description"),
        "category": food.get("foodCategory", ""),
        "nutrients": nutrients
    }

def save_data(all_foods):
    """Save downloaded data to JSON files"""
    # Save full dataset
    full_path = OUTPUT_DIR / "usda_foods_full.json"
    with open(full_path, "w") as f:
        json.dump(all_foods, f, indent=2)
    print(f"\n✅ Full dataset saved: {full_path}")
    print(f"   Total foods: {len(all_foods)}")

    # Save nutrition only dataset (smaller, cleaner)
    nutrition_data = [extract_nutrition(food) for food in all_foods]
    nutrition_path = OUTPUT_DIR / "usda_nutrition.json"
    with open(nutrition_path, "w") as f:
        json.dump(nutrition_data, f, indent=2)
    print(f"✅ Nutrition dataset saved: {nutrition_path}")

def main():
    print("=" * 50)
    print("🥦 USDA FoodData Central Downloader")
    print("=" * 50)

    # Checks
    if not check_api_key():
        return
    create_output_dir()

    # Download all categories
    all_foods = []
    for category in FOOD_CATEGORIES:
        foods = download_foods_by_category(category)
        all_foods.extend(foods)
        time.sleep(0.5)  # Avoid rate limiting

    # Save data
    save_data(all_foods)

    print("\n" + "=" * 50)
    print("✅ USDA Download Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()