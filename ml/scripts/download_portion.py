import os
import subprocess
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "portion"

# We will use Nutrition5k dataset from Kaggle
# It contains food images with weight/portion annotations
DATASET = "mathchi/nutrition5k-dataset"

def create_output_dir():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")

def download_dataset():
    """Download dataset using Kaggle API"""
    print(f"\n📥 Downloading Portion Estimation dataset...")
    print(f"   Dataset: {DATASET}")
    print(f"   This may take a few minutes ⏳")

    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", DATASET,
            "-p", str(OUTPUT_DIR),
            "--unzip"
        ], check=True)
        print(f"✅ Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"   Trying alternative dataset...")
        return download_alternative()
    return True

def download_alternative():
    """Download alternative portion dataset if main fails"""
    print(f"\n📥 Trying alternative dataset...")
    ALT_DATASET = "dansbecker/food-101"

    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", ALT_DATASET,
            "-p", str(OUTPUT_DIR),
            "--unzip"
        ], check=True)
        print(f"✅ Alternative download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Alternative download also failed: {e}")
        return False

def verify_dataset():
    """Verify downloaded files"""
    print(f"\n🔍 Verifying dataset...")
    all_files = list(OUTPUT_DIR.rglob("*"))
    files = [f for f in all_files if f.is_file()]

    if not files:
        print("❌ No files found!")
        return False

    print(f"✅ Files found: {len(files)}")

    # Show folder structure
    folders = [f for f in OUTPUT_DIR.iterdir() if f.is_dir()]
    print(f"   Folders: {[f.name for f in folders]}")

    # Show total size
    total_size = sum(f.stat().st_size for f in files)
    total_mb = total_size / (1024 * 1024)
    print(f"   Total size: {total_mb:.1f} MB")

    return True

def create_portion_labels():
    """Create a sample portion labels file for reference"""
    print(f"\n📝 Creating portion size reference labels...")

    portion_labels = {
        "portion_sizes": {
            "extra_small": {
                "description": "Extra Small Portion",
                "weight_g": "50-100",
                "example": "Small snack, handful of nuts"
            },
            "small": {
                "description": "Small Portion",
                "weight_g": "100-200",
                "example": "Side dish, small bowl"
            },
            "medium": {
                "description": "Medium Portion",
                "weight_g": "200-350",
                "example": "Standard meal serving"
            },
            "large": {
                "description": "Large Portion",
                "weight_g": "350-500",
                "example": "Large meal, restaurant serving"
            },
            "extra_large": {
                "description": "Extra Large Portion",
                "weight_g": "500+",
                "example": "Very large meal, shared dish"
            }
        },
        "common_foods": {
            "rice_cooked": {"small": 100, "medium": 200, "large": 300},
            "chicken_breast": {"small": 100, "medium": 150, "large": 200},
            "salad": {"small": 50, "medium": 100, "large": 200},
            "pasta_cooked": {"small": 150, "medium": 250, "large": 350},
            "bread_slice": {"small": 25, "medium": 35, "large": 50},
            "apple": {"small": 100, "medium": 150, "large": 200},
            "banana": {"small": 80, "medium": 120, "large": 150},
            "milk": {"small": 100, "medium": 200, "large": 300},
            "egg": {"small": 50, "medium": 60, "large": 70},
            "potato": {"small": 100, "medium": 150, "large": 200}
        }
    }

    import json
    labels_path = OUTPUT_DIR / "portion_labels.json"
    with open(labels_path, "w") as f:
        json.dump(portion_labels, f, indent=2)
    print(f"✅ Portion labels saved: {labels_path}")

def main():
    print("=" * 50)
    print("📏 Portion Estimation Dataset Downloader")
    print("=" * 50)

    create_output_dir()

    # Check if already downloaded
    existing_files = list(OUTPUT_DIR.rglob("*.jpg")) + \
                     list(OUTPUT_DIR.rglob("*.png")) + \
                     list(OUTPUT_DIR.rglob("*.csv"))

    if existing_files:
        print("\n⚠️  Dataset already exists!")
        print("   Skipping download")
        verify_dataset()
    else:
        # Download
        success = download_dataset()
        if not success:
            print("\n⚠️  Download failed but continuing...")

    # Always create portion labels
    create_portion_labels()

    # Verify
    verify_dataset()

    print("\n" + "=" * 50)
    print("✅ Portion Dataset Download Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()