# ============================================================
# Preprocessing BEFORE Annotation (Roboflow ke liye)
# Steps: Resize + Auto-Orient + Normalize + Quality Check
# Phir ye images Roboflow pe upload karuga aur annotate karuga
# ============================================================
# Install: pip install opencv-python numpy Pillow
# Run    : python preprocess_before_annotation.py
# ============================================================

import cv2
import os
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path

# ============================================================
# SETTINGS - Sirf ye change karo
# ============================================================

INPUT_DIR  = "raw_images"       #  original images yahan hain
OUTPUT_DIR = "ready_to_annotate"  # Processed images yahan jayengi
IMG_SIZE   = (640, 640)           # YOLOv8 standard size
MIN_BLUR_THRESHOLD = 100          # Isse kam blur score = blurry image (skip hogi)

# ============================================================
# FUNCTION 1: EXIF se Auto-Orient (Phone images fix)
# ============================================================

def fix_orientation(img_path):
    """
    Phone se li gayi images aksar rotated hoti hain.
    Ye function EXIF data padhke sahi rotate karta hai.
    """
    try:
        pil_img = Image.open(img_path)
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
        # PIL to OpenCV convert
        img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception:
        # Agar EXIF nahi hai toh normal load
        return cv2.imread(str(img_path))


# ============================================================
# FUNCTION 2: Blur Detection (Blurry images skip )
# ============================================================

def is_blurry(img, threshold=MIN_BLUR_THRESHOLD):
    """
    Laplacian variance se blur check karta hai.
    Score jitna kam = utni zyada blur.
    Return: (True/False, blur_score)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, round(score, 2)


# ============================================================
# FUNCTION 3: Resize with Aspect Ratio (Letterbox)
# ============================================================

def letterbox_resize(img, size=(640, 640)):
    """
    Image ko stretch kiye bina resize karta hai.
    Baaki jagah gray padding se bharta hai.
    Yahi YOLOv8 karta hai internally.
    """
    h, w = img.shape[:2]
    target_w, target_h = size

    # Scale calculate kar
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Gray canvas banao
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    # Center mein paste karna
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


# ============================================================
# FUNCTION 4: Brightness Check
# ============================================================

def check_brightness(img):
    """
    Image ki average brightness check karta hai.
    Return: 'too_dark', 'too_bright', 'ok'
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 40:
        return "too_dark", round(mean_brightness, 2)
    elif mean_brightness > 220:
        return "too_bright", round(mean_brightness, 2)
    else:
        return "ok", round(mean_brightness, 2)


# ============================================================
# MAIN FUNCTION: Poora Preprocessing Pipeline
# ============================================================

def preprocess_for_annotation(input_dir, output_dir, size=IMG_SIZE):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    # Output folders banega
    good_dir    = output_path / "good"        # Annotation ke liye ready
    blurry_dir  = output_path / "blurry"      # Blurry images (check karo)
    bright_dir  = output_path / "bad_light"   # Too dark/bright images

    for d in [good_dir, blurry_dir, bright_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Supported formats
    extensions  = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]
    image_files = [f for f in input_path.iterdir() if f.suffix in extensions]

    if not image_files:
        print(f"\n[ERROR] Koi image nahi mili: {input_dir}")
        print("INPUT_DIR ka path check karo!\n")
        return

    print(f"\n{'='*55}")
    print(f"  PREPROCESSING BEFORE ANNOTATION")
    print(f"{'='*55}")
    print(f"  Input Folder : {input_dir}")
    print(f"  Output Folder: {output_dir}")
    print(f"  Image Size   : {size[0]}x{size[1]}")
    print(f"  Total Images : {len(image_files)}")
    print(f"{'='*55}\n")

    good_count   = 0
    blurry_count = 0
    light_count  = 0

    for i, img_file in enumerate(image_files, 1):

        # --- Step 1: Load + Auto-Orient ---
        img = fix_orientation(img_file)
        if img is None:
            print(f"[{i}] SKIP (load failed) -> {img_file.name}")
            continue

        # --- Step 2: Blur Check ---
        blurry, blur_score = is_blurry(img)
        if blurry:
            out = blurry_dir / img_file.name
            cv2.imwrite(str(out), img)
            blurry_count += 1
            print(f"[{i}] BLURRY (score={blur_score}) -> {img_file.name}")
            continue

        # --- Step 3: Brightness Check ---
        light_status, brightness = check_brightness(img)
        if light_status != "ok":
            out = bright_dir / img_file.name
            cv2.imwrite(str(out), img)
            light_count += 1
            print(f"[{i}] {light_status.upper()} (brightness={brightness}) -> {img_file.name}")
            continue

        # --- Step 4: Letterbox Resize (640x640) ---
        img_processed = letterbox_resize(img, size)

        # --- Step 5: Save ---
        out = good_dir / img_file.name
        cv2.imwrite(str(out), img_processed)
        good_count += 1
        print(f"[{i}] OK (blur={blur_score}, brightness={brightness}) -> {img_file.name}")

    # Summary
    print(f"\n{'='*55}")
    print(f"  PREPROCESSING COMPLETE!")
    print(f"{'='*55}")
    print(f"  Ready to Annotate : {good_count}  images  -> {good_dir}")
    print(f"  Blurry (Review)   : {blurry_count}  images  -> {blurry_dir}")
    print(f"  Bad Light (Review): {light_count}  images  -> {bright_dir}")
    print(f"{'='*55}")
    print(f"\n  Ab '{good_dir}' wali images")
    print(f"  Roboflow pe UPLOAD karo aur ANNOTATE karo!\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    preprocess_for_annotation(INPUT_DIR, OUTPUT_DIR, IMG_SIZE)
