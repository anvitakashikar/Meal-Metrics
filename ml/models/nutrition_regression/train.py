import json
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Add ml/ to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import USDA_DIR, SAVED_MODELS_DIR

# Paths
DATA_PATH = USDA_DIR / "usda_nutrition.json"
MODEL_DIR = SAVED_MODELS_DIR / "nutrition_regression"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and parse USDA nutrition data"""
    print("📂 Loading USDA nutrition data...")

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    records = []
    for item in data:
        nutrients = item.get("nutrients", {})

        if all(k in nutrients for k in [
            "calories", "protein_g", "carbohydrates_g", "fat_g"
        ]):
            records.append({
                "description": item.get("description", ""),
                "category": item.get("category", "unknown"),
                "calories": nutrients.get("calories", 0),
                "protein_g": nutrients.get("protein_g", 0),
                "carbohydrates_g": nutrients.get("carbohydrates_g", 0),
                "fat_g": nutrients.get("fat_g", 0),
                "fiber_g": nutrients.get("fiber_g", 0),
                "sugar_g": nutrients.get("sugar_g", 0),
                "sodium_mg": nutrients.get("sodium_mg", 0),
            })

    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} food items")
    print(f"   Columns: {list(df.columns)}")
    return df


def preprocess_data(df):
    """Preprocess data for training"""
    print("\n🔧 Preprocessing data...")

    # Encode category
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])

    feature_cols = ["protein_g", "carbohydrates_g", "fat_g", "category_encoded"]
    target_cols = ["calories", "fiber_g", "sugar_g", "sodium_mg"]

    X = df[feature_cols]
    y = df[target_cols]

    # Remove outliers beyond 3 std
    mask = (
        np.abs(
            df[["calories", "protein_g", "carbohydrates_g", "fat_g"]] -
            df[["calories", "protein_g", "carbohydrates_g", "fat_g"]].mean()
        ) <= 3 * df[["calories", "protein_g", "carbohydrates_g", "fat_g"]].std()
    ).all(axis=1)

    X = X[mask]
    y = y[mask]

    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Targets shape: {y.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler and encoder
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("✅ Scaler and encoder saved")
    return X_scaled, y, le, scaler


def train_models(X, y):
    """Train multiple regression models"""
    print("\n🤖 Training models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        ),
        "linear_regression": LinearRegression()
    }

    results = {}
    best_model = None
    best_score = -999

    for name, model in models.items():
        print(f"\n   Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"mae": mae, "r2": r2}
        print(f"   MAE: {mae:.2f}")
        print(f"   R2 Score: {r2:.4f}")

        # Track best model
        if r2 > best_score:
            best_score = r2
            best_model = (name, model)

    print(f"\n🏆 Best Model: {best_model[0]} (R2: {best_score:.4f})")
    return best_model, results


def save_model(best_model):
    """Save the best model"""
    name, model = best_model
    model_path = MODEL_DIR / "nutrition_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Best model saved: {model_path}")

    model_info = {
        "model_type": name,
        "features": [
            "protein_g",
            "carbohydrates_g",
            "fat_g",
            "category_encoded"
        ],
        "targets": [
            "calories",
            "fiber_g",
            "sugar_g",
            "sodium_mg"
        ],
        "version": "1.0"
    }

    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"✅ Model info saved")


def evaluate_model(best_model, X, y):
    """Print final evaluation summary"""
    name, model = best_model
    y_pred = model.predict(X)

    print("\n" + "=" * 50)
    print("📊 Final Model Evaluation")
    print("=" * 50)
    print(f"Model: {name}")

    targets = ["calories", "fiber_g", "sugar_g", "sodium_mg"]
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y.iloc[:, i], y_pred[:, i])
        print(f"\n{target}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  R2:  {r2:.4f}")


def main():
    print("=" * 50)
    print("🔢 Nutrition Regression Model Training")
    print("=" * 50)

    # Load data
    df = load_data()

    # Preprocess
    X, y, le, scaler = preprocess_data(df)

    # Train
    best_model, results = train_models(X, y)

    # Save
    save_model(best_model)

    # Evaluate
    evaluate_model(best_model, X, y)

    print("\n" + "=" * 50)
    print("✅ Nutrition Regression Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()