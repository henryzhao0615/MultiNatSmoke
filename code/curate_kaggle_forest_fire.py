#!/usr/bin/env python3
"""
curate_kaggle_forest_fire.py

Automates:
1. Ensure Kaggle Forest Fire dataset is available locally
2. Download via kagglehub (with cache)
3. Copy dataset into expected raw_data folder
4. Curate images for all dataset configs using image_list.txt

Run from dataset root:
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
# Build index for fast lookup
# -----------------------------
def build_image_index(raw_root):
    print("[INFO] Indexing raw dataset...")
    index = {}

    for root, _, files in os.walk(raw_root):
        for f in files:
            index[f] = os.path.join(root, f)

    print(f"[INFO] Indexed {len(index)} files.")
    return index


# -----------------------------
# Find dataset configs
# -----------------------------
def find_configs(base_dir):
    configs = []

    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)

        if not os.path.isdir(path):
            continue

        list_file = os.path.join(
            path,
            "kaggle_Forest_Fire",
            "image_list.txt"
        )

        if os.path.exists(list_file):
            configs.append(entry)

    print(f"[INFO] Found {len(configs)} dataset configs.")
    return configs


# -----------------------------
# Curate images
# -----------------------------
def curate_images(base_dir, config, image_index):
    list_path = os.path.join(
        base_dir,
        config,
        "kaggle_Forest_Fire",
        "image_list.txt"
    )

    dest_dir = os.path.join(
        base_dir,
        config,
        "kaggle_Forest_Fire",
        "images"
    )

    print(f"\n[INFO] Processing config: {config}")
    os.makedirs(dest_dir, exist_ok=True)

    with open(list_path, "r") as f:
        image_list = [line.strip() for line in f if line.strip()]

    copied = 0
    missing = 0

    for img_name in image_list:
        if img_name not in image_index:
            missing += 1
            continue

        src = image_index[img_name]
        dst = os.path.join(dest_dir, img_name)

        shutil.copy2(src, dst)
        copied += 1

    print(f"[SUMMARY] {config} → Copied: {copied}, Missing: {missing}")


# -----------------------------
# Verification: images vs masks
# -----------------------------
def verify_masks(base_dir, configs):
    print("\n[INFO] Verifying masks consistency with images...")

    for config in configs:
        kaggle_dir = os.path.join(base_dir, config, "kaggle_Forest_Fire")
        images_dir = os.path.join(kaggle_dir, "images")
        masks_dir = os.path.join(kaggle_dir, "masks")

        if not os.path.exists(masks_dir):
            print(f"[WARNING] Masks folder missing for {config}: {masks_dir}")
            continue

        image_files = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)))
        mask_files = set(os.path.splitext(f)[0] for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f)))

        # Check each mask has a corresponding image
        missing_images = mask_files - image_files
        if missing_images:
            print(f"[WARNING] {len(missing_images)} masks without corresponding images in {config}: {missing_images}")

        # Optionally, also warn if images missing masks
        # missing_masks = image_files - mask_files
        # if missing_masks:
        #     print(f"[WARNING] {len(missing_masks)} images missing masks in {config}: {missing_masks}")

        if not missing_images:
            print(f"[INFO] All masks have corresponding images for {config}")

# -----------------------------
# Main
# -----------------------------
def main():
    base_dir = os.getcwd()
    raw_root = os.path.join(base_dir, "raw_data", "kaggle_forest_fire")

    print(f"[INFO] Working directory: {base_dir}")

    # Step 1: ensure kagglehub
    ensure_kagglehub()

    # Step 2: ensure dataset
    ensure_dataset(raw_root)

    # Step 3: find configs
    configs = find_configs(base_dir)

    if not configs:
        print("[ERROR] No dataset configs found.")
        sys.exit(1)

    # Step 4: build index once
    image_index = build_image_index(raw_root)

    # Step 5: process each config
    for config in configs:
        curate_images(base_dir, config, image_index)

    # -----------------------------
    # Step 6: Verify masks
    # -----------------------------
    verify_masks(base_dir, configs)


if __name__ == "__main__":
    main()