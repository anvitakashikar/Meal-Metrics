import os
import json
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import subprocess
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Config
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "recipe1m"
DATASET = "irkaal/foodcom-recipes-and-reviews"

def create_output_dir():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")

def download_dataset():
    """Download dataset using Kaggle API"""
    print(f"\n📥 Downloading Food.com Recipes dataset...")
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
        return False
    return True

def verify_dataset():
    """Verify downloaded files"""
    print(f"\n🔍 Verifying dataset...")
    files = list(OUTPUT_DIR.glob("*.csv"))

    if not files:
        print("❌ No CSV files found!")
        return False

    print(f"✅ Files found:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name} ({size_mb:.1f} MB)")
    return True

def preview_dataset():
    """Preview the dataset contents"""
    import pandas as pd

    print(f"\n👀 Dataset Preview:")
    csv_files = list(OUTPUT_DIR.glob("*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, nrows=5)
            print(f"\n📄 {csv_file.name}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Shape: {df.shape}")
        except Exception as e:
            print(f"   ❌ Could not read {csv_file.name}: {e}")

def main():
    print("=" * 50)
    print("🥗 Food.com Recipes Dataset Downloader")
    print("=" * 50)

    create_output_dir()

    # Check if already downloaded
    existing_files = list(OUTPUT_DIR.glob("*.csv"))
    if existing_files:
        print("\n⚠️  Dataset already exists!")
        print("   Skipping download")
        verify_dataset()
        preview_dataset()
        return

    # Download
    success = download_dataset()
    if not success:
        return

    # Verify
    verify_dataset()

    # Preview
    try:
        import pandas as pd
        preview_dataset()
    except ImportError:
        print("\n⚠️  pandas not installed, skipping preview")
        print("   Run: pip install pandas")

    print("\n" + "=" * 50)
    print("✅ Recipe Dataset Download Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()