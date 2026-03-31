import json
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Add ml/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import RAW_DATA_DIR, SAVED_MODELS_DIR

# Paths
PORTION_DIR = RAW_DATA_DIR / "portion"
USDA_DIR = RAW_DATA_DIR / "usda"
MODEL_DIR = SAVED_MODELS_DIR / "portion_estimation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_portion_labels():
    """Load portion size reference labels"""
    print("📂 Loading portion labels...")

    labels_path = PORTION_DIR / "portion_labels.json"
    with open(labels_path, "r") as f:
        labels = json.load(f)

    print("✅ Portion labels loaded")
    return labels


def load_usda_data():
    """Load USDA nutrition data"""
    print("📂 Loading USDA nutrition data...")

    usda_path = USDA_DIR / "usda_nutrition.json"
    with open(usda_path, "r") as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} USDA food items")
    return data


def generate_training_data(usda_data, portion_labels):
    """
    Generate synthetic training data for portion estimation
    based on USDA nutrition data and portion size references
    """
    print("\n🔧 Generating training data...")

    portion_sizes = portion_labels["portion_sizes"]
    common_foods = portion_labels["common_foods"]

    records = []

    for item in usda_data:
        nutrients = item.get("nutrients", {})
        calories = nutrients.get("calories", 0)
        protein = nutrients.get("protein_g", 0)
        carbs = nutrients.get("carbohydrates_g", 0)
        fat = nutrients.get("fat_g", 0)
        sodium = nutrients.get("sodium_mg", 0)

        if calories <= 0:
            continue

        # Generate samples for each portion size
        portion_configs = [
            {
                "label": "extra_small",
                "multiplier": 0.5,
                "weight_g": 75
            },
            {
                "label": "small",
                "multiplier": 0.75,
                "weight_g": 150
            },
            {
                "label": "medium",
                "multiplier": 1.0,
                "weight_g": 250
            },
            {
                "label": "large",
                "multiplier": 1.5,
                "weight_g": 400
            },
            {
                "label": "extra_large",
                "multiplier": 2.0,
                "weight_g": 550
            }
        ]

        for portion in portion_configs:
            m = portion["multiplier"]

            # Add some noise for realism
            noise = np.random.uniform(0.9, 1.1)

            records.append({
                "calories": round(calories * m * noise, 1),
                "protein_g": round(protein * m * noise, 1),
                "carbohydrates_g": round(carbs * m * noise, 1),
                "fat_g": round(fat * m * noise, 1),
                "sodium_mg": round(sodium * m * noise, 1),
                "weight_g": round(portion["weight_g"] * noise, 1),
                "portion_label": portion["label"]
            })

    df = pd.DataFrame(records)
    print(f"✅ Generated {len(df)} training samples")
    print(f"   Portion distribution:")
    print(df["portion_label"].value_counts())
    return df


def preprocess_data(df):
    """Preprocess data for training"""
    print("\n🔧 Preprocessing data...")

    # Features
    feature_cols = [
        "calories",
        "protein_g",
        "carbohydrates_g",
        "fat_g",
        "sodium_mg",
        "weight_g"
    ]

    X = df[feature_cols]
    y = df["portion_label"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Classes: {list(le.classes_)}")

    # Save preprocessors
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Scaler and encoder saved")
    return X_scaled, y_encoded, le, scaler


def train_models(X, y):
    """Train multiple classification models"""
    print("\n🤖 Training models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }

    best_model = None
    best_score = 0
    results = {}

    for name, model in models.items():
        print(f"\n   Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {"accuracy": accuracy}
        print(f"   Accuracy: {accuracy:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = (name, model)

    print(f"\n🏆 Best Model: {best_model[0]} "
          f"(Accuracy: {best_score:.4f})")

    # Print detailed report for best model
    y_pred_best = best_model[1].predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(
        y_test,
        y_pred_best,
        target_names=["extra_large", "extra_small",
                      "large", "medium", "small"]
    ))

    return best_model, results


def save_model(best_model):
    """Save the best model"""
    name, model = best_model
    model_path = MODEL_DIR / "portion_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Best model saved: {model_path}")

    model_info = {
        "model_type": name,
        "features": [
            "calories",
            "protein_g",
            "carbohydrates_g",
            "fat_g",
            "sodium_mg",
            "weight_g"
        ],
        "classes": [
            "extra_small",
            "small",
            "medium",
            "large",
            "extra_large"
        ],
        "version": "1.0"
    }

    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print("✅ Model info saved")


def main():
    print("=" * 50)
    print("📏 Portion Estimation Model Training")
    print("=" * 50)

    # Load data
    portion_labels = load_portion_labels()
    usda_data = load_usda_data()

    # Generate training data
    df = generate_training_data(usda_data, portion_labels)

    # Preprocess
    X, y, le, scaler = preprocess_data(df)

    # Train
    best_model, results = train_models(X, y)

    # Save
    save_model(best_model)

    print("\n" + "=" * 50)
    print("✅ Portion Estimation Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()