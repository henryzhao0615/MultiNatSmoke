"""
curate_kaggle_forest_fire.py

Automates:
1. Ensure Kaggle Forest Fire dataset is available locally
2. Download via kagglehub (with cache)
3. Read txt files from target folders
4. Copy same-named images from data/kaggle_forest_fire/test-big
   into the same folder where each txt file is located

Run from code root:
    python curate_kaggle_forest_fire.py
"""

import os
import sys
import shutil
import subprocess


# -----------------------------
# Install kagglehub if missing
# -----------------------------
def ensure_kagglehub():
    try:
        import kagglehub  # noqa
        print("[INFO] kagglehub already installed.")
    except ImportError:
        print("[INFO] Installing kagglehub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])


# -----------------------------
# Ensure dataset exists locally
# -----------------------------
def ensure_dataset(raw_root):
    if os.path.exists(raw_root) and len(os.listdir(raw_root)) > 0:
        print("[INFO] Using existing dataset at:", raw_root)
        return raw_root

    try:
        import kagglehub

        print("[INFO] Downloading via kagglehub (or using cache)...")
        cache_path = kagglehub.dataset_download("kutaykutlu/forest-fire")
        print("[INFO] Kagglehub cache path:", cache_path)

        print("[INFO] Copying dataset to:", raw_root)
        os.makedirs(raw_root, exist_ok=True)

        for item in os.listdir(cache_path):
            src = os.path.join(cache_path, item)
            dst = os.path.join(raw_root, item)

            if os.path.exists(dst):
                continue

            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        print("[INFO] Dataset prepared at:", raw_root)
        return raw_root

    except Exception as e:
        print("[WARNING] Kaggle download failed:", e)
        print("\nPlease download manually from:")
        print("https://www.kaggle.com/datasets/kutaykutlu/forest-fire")
        print(f"Extract to: {raw_root}")
        sys.exit(1)


# -----------------------------
# Find all target txt files
# -----------------------------
def collect_target_txts(base_dir):
    """
    Find all txt files under:
    - MultiNatSmokeDataset/Train/Forest_Fire
    - MultiNatSmokeDataset/Test/Forest_Fire
    - TestLarge
    - TestMedium
    - TestSmall
    """
    target_dirs = [
        os.path.join(base_dir, "data", "MultiNatSmokeDataset", "Train", "Forest_Fire"),
        os.path.join(base_dir, "data", "MultiNatSmokeDataset", "Test", "Forest_Fire"),
        os.path.join(base_dir, "data", "MultiNatSmokeDataset", "TestLarge", "Forest_Fire"),
        os.path.join(base_dir, "data", "MultiNatSmokeDataset", "TestMedium", "Forest_Fire"),
        os.path.join(base_dir, "data", "MultiNatSmokeDataset", "TestSmall", "Forest_Fire"),
    ]

    txt_files = []

    for folder in target_dirs:
        if not os.path.exists(folder):
            print(f"[WARNING] Folder not found, skipped: {folder}")
            continue

        for name in os.listdir(folder):
            if not name.lower().endswith(".txt"):
                continue

            if name.lower() == "info.txt":
                print(f"[INFO] Ignored file: {os.path.join(folder, name)}")
                continue

            txt_files.append(os.path.join(folder, name))

    print(f"[INFO] Found {len(txt_files)} txt files.")
    return txt_files


# -----------------------------
# Read image names from one txt
# -----------------------------
def read_image_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


# -----------------------------
# Copy images for one txt file
# -----------------------------
def copy_images_from_test_big(txt_path, source_dir):
    """
    Copy images listed in txt_path from source_dir
    into an 'images' folder under the same folder as txt_path.
    """
    image_names = read_image_list(txt_path)

    parent_dir = os.path.dirname(txt_path)
    dest_dir = os.path.join(parent_dir, "images")
    os.makedirs(dest_dir, exist_ok=True)

    copied = 0
    missing = 0

    print(f"\n[INFO] Processing txt: {txt_path}")
    print(f"[INFO] Destination folder: {dest_dir}")

    for img_name in image_names:
        src = os.path.join(source_dir, img_name)
        dst = os.path.join(dest_dir, img_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"[MISSING] {img_name}")
            missing += 1

    print(f"[SUMMARY] {os.path.basename(txt_path)} -> Copied: {copied}, Missing: {missing}")


# -----------------------------
# Main
# -----------------------------
def main():
    base_dir = os.getcwd()
    print(base_dir)

    raw_root = os.path.join(base_dir, "data", "kaggle_forest_fire")
    source_dir = os.path.join(raw_root, "test_big")

    print(f"[INFO] Working directory: {base_dir}")

    # Step 1: ensure kagglehub
    ensure_kagglehub()

    # Step 2: ensure dataset
    ensure_dataset(raw_root)

    # Step 3: check source folder
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source folder not found: {source_dir}")
        sys.exit(1)

    print(f"[INFO] Source image folder: {source_dir}")

    # Step 4: collect target txt files
    txt_files = collect_target_txts(base_dir)

    if not txt_files:
        print("[ERROR] No txt files found in target folders.")
        sys.exit(1)

    # Step 5: copy images according to each txt
    for txt_path in txt_files:
        copy_images_from_test_big(txt_path, source_dir)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()