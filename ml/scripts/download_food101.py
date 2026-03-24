import os
import tarfile
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

# Config
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw" / "food101"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
TAR_FILE = OUTPUT_DIR / "food-101.tar.gz"

def create_output_dir():
    """Create output directory if it doesn't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")

def download_dataset():
    """Download Food-101 dataset with progress bar"""
    print(f"\n📥 Downloading Food-101 dataset...")
    print(f"   URL: {FOOD101_URL}")
    print(f"   Size: ~5GB — this will take a while ⏳")

    response = requests.get(FOOD101_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1MB chunks

    with open(TAR_FILE, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            progress.update(len(chunk))

    print(f"\n✅ Download complete: {TAR_FILE}")

def extract_dataset():
    """Extract the tar.gz file"""
    print(f"\n📦 Extracting dataset...")
    print(f"   This may take a few minutes ⏳")

    with tarfile.open(TAR_FILE, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, OUTPUT_DIR)

    print(f"✅ Extraction complete!")

def verify_dataset():
    """Verify the dataset structure"""
    food101_dir = OUTPUT_DIR / "food-101"
    images_dir = food101_dir / "images"
    meta_dir = food101_dir / "meta"

    if not images_dir.exists():
        print("❌ Images directory not found!")
        return False

    if not meta_dir.exists():
        print("❌ Meta directory not found!")
        return False

    # Count categories
    categories = [d for d in images_dir.iterdir() if d.is_dir()]
    print(f"\n✅ Dataset verified!")
    print(f"   Categories found: {len(categories)}")
    print(f"   Sample categories: {[c.name for c in categories[:5]]}")

    # Count total images
    total_images = sum(
        len(list(cat.glob("*.jpg"))) for cat in categories
    )
    print(f"   Total images: {total_images}")
    return True

def cleanup_tar():
    """Remove tar file to save space"""
    if TAR_FILE.exists():
        os.remove(TAR_FILE)
        print(f"\n🗑️  Removed tar file to save space")

def main():
    print("=" * 50)
    print("🍔 Food-101 Dataset Downloader")
    print("=" * 50)

    create_output_dir()

    # Check if already downloaded
    food101_dir = OUTPUT_DIR / "food-101"
    if food101_dir.exists():
        print("\n⚠️  Food-101 folder already exists!")
        print("   Skipping download and extraction")
        verify_dataset()
        return

    # Download
    download_dataset()

    # Extract
    extract_dataset()

    # Verify
    verify_dataset()

    # Cleanup
    cleanup_tar()

    print("\n" + "=" * 50)
    print("✅ Food-101 Download Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()