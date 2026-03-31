import pickle
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add ml/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import SAVED_MODELS_DIR

# Paths
MODEL_DIR = SAVED_MODELS_DIR / "portion_estimation"
MODEL_PATH = MODEL_DIR / "portion_model.pkl"
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


def estimate_portion(
    calories: float,
    protein_g: float,
    carbohydrates_g: float,
    fat_g: float,
    sodium_mg: float,
    weight_g: float = 250.0
) -> dict:
    """
    Estimate portion size from nutrition values

    Args:
        calories: calories in meal
        protein_g: protein in grams
        carbohydrates_g: carbohydrates in grams
        fat_g: fat in grams
        sodium_mg: sodium in mg
        weight_g: estimated weight in grams

    Returns:
        dict with portion size label and details
    """
    model, scaler, le = load_model()

    # Prepare features
    features = pd.DataFrame([[
        calories,
        protein_g,
        carbohydrates_g,
        fat_g,
        sodium_mg,
        weight_g
    ]], columns=[
        "calories",
        "protein_g",
        "carbohydrates_g",
        "fat_g",
        "sodium_mg",
        "weight_g"
    ])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # Decode label
    portion_label = le.inverse_transform([prediction])[0]

    # Get confidence
    confidence = float(max(probabilities))

    # Map to weight range
    weight_ranges = {
        "extra_small": "50-100g",
        "small": "100-200g",
        "medium": "200-350g",
        "large": "350-500g",
        "extra_large": "500g+"
    }

    return {
        "portion_label": portion_label,
        "weight_range": weight_ranges.get(portion_label, "200-350g"),
        "confidence": round(confidence, 2),
        "estimated_weight_g": weight_g
    }


def main():
    """Test portion estimation"""
    print("=" * 50)
    print("📏 Portion Estimation Predictor")
    print("=" * 50)

    test_meals = [
        {
            "name": "Small Snack",
            "calories": 150,
            "protein_g": 5,
            "carbohydrates_g": 20,
            "fat_g": 5,
            "sodium_mg": 200,
            "weight_g": 100
        },
        {
            "name": "Regular Lunch",
            "calories": 450,
            "protein_g": 25,
            "carbohydrates_g": 50,
            "fat_g": 15,
            "sodium_mg": 600,
            "weight_g": 300
        },
        {
            "name": "Large Dinner",
            "calories": 800,
            "protein_g": 40,
            "carbohydrates_g": 80,
            "fat_g": 30,
            "sodium_mg": 1200,
            "weight_g": 500
        },
        {
            "name": "Extra Large Meal",
            "calories": 1200,
            "protein_g": 60,
            "carbohydrates_g": 120,
            "fat_g": 45,
            "sodium_mg": 1800,
            "weight_g": 700
        }
    ]

    for meal in test_meals:
        print(f"\n🍽️  {meal['name']}")
        print(f"   Calories: {meal['calories']} kcal")
        print(f"   Weight: {meal['weight_g']}g")

        result = estimate_portion(
            calories=meal["calories"],
            protein_g=meal["protein_g"],
            carbohydrates_g=meal["carbohydrates_g"],
            fat_g=meal["fat_g"],
            sodium_mg=meal["sodium_mg"],
            weight_g=meal["weight_g"]
        )

        print(f"   Predicted portion: {result['portion_label']}")
        print(f"   Weight range: {result['weight_range']}")
        print(f"   Confidence: {result['confidence']:.0%}")


if __name__ == "__main__":
    main()