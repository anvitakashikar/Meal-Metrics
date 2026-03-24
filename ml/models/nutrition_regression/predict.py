import json
import pickle
import numpy as np
from pathlib import Path
import sys
import pandas as pd

# Add ml/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import SAVED_MODELS_DIR

# Paths
MODEL_DIR = SAVED_MODELS_DIR / "nutrition_regression"
MODEL_PATH = MODEL_DIR / "nutrition_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

def load_model():
    """Load trained model, scaler and encoder"""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, scaler, le

def predict_nutrition(
    protein_g,
    carbohydrates_g,
    fat_g,
    category="unknown"
):
    """
    Predict nutrition values from macros
    
    Args:
        protein_g: Protein in grams
        carbohydrates_g: Carbohydrates in grams
        fat_g: Fat in grams
        category: Food category string
    
    Returns:
        dict with predicted nutrition values
    """
    # Load model
    model, scaler, le = load_model()

    # Encode category
    try:
        category_encoded = le.transform([category])[0]
    except ValueError:
        # Unknown category → use 0
        category_encoded = 0

    # Prepare features
    features = pd.DataFrame([[
    protein_g,
    carbohydrates_g,
    fat_g,
    category_encoded
]], columns=["protein_g", "carbohydrates_g", "fat_g", "category_encoded"])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    return {
        "calories": round(float(prediction[0]), 1),
        "fiber_g": round(max(0, float(prediction[1])), 1),
        "sugar_g": round(max(0, float(prediction[2])), 1),
        "sodium_mg": round(max(0, float(prediction[3])), 1)
    }

def main():
    """Test the prediction with sample inputs"""
    print("=" * 50)
    print("🔢 Nutrition Regression Predictor")
    print("=" * 50)

    # Test samples
    test_samples = [
        {
            "name": "Grilled Chicken",
            "protein_g": 30,
            "carbohydrates_g": 0,
            "fat_g": 5,
            "category": "Poultry Products"
        },
        {
            "name": "Pasta with Sauce",
            "protein_g": 8,
            "carbohydrates_g": 45,
            "fat_g": 10,
            "category": "Cereal Grains and Pasta"
        },
        {
            "name": "Mixed Salad",
            "protein_g": 3,
            "carbohydrates_g": 10,
            "fat_g": 5,
            "category": "Vegetables and Vegetable Products"
        }
    ]

    for sample in test_samples:
        print(f"\n🍽️  {sample['name']}")
        print(f"   Input → Protein: {sample['protein_g']}g | "
              f"Carbs: {sample['carbohydrates_g']}g | "
              f"Fat: {sample['fat_g']}g")

        result = predict_nutrition(
            protein_g=sample["protein_g"],
            carbohydrates_g=sample["carbohydrates_g"],
            fat_g=sample["fat_g"],
            category=sample["category"]
        )

        print(f"   Output →")
        print(f"     Calories:  {result['calories']} kcal")
        print(f"     Fiber:     {result['fiber_g']}g")
        print(f"     Sugar:     {result['sugar_g']}g")
        print(f"     Sodium:    {result['sodium_mg']}mg")

if __name__ == "__main__":
    main()