"""
build_rhs_csv.py  (v2 — fixed Lab conversion + duplicate handling)
===================================================================
Step 1 of the Abaca Color AI new model pipeline.

Fixes vs v1:
- Lab_L conversion fixed (was 0.0-1.0, now correct 0-100 range)
- Duplicates now kept as COLORNAME_RHSCODE instead of skipped
  e.g. 34A appears as both ORANGEREDN_34A and ORANGERED_34A

Usage:
    python build_rhs_csv.py

Output:
    abaca_pipeline/rhs_colors.csv
"""

import cv2
import numpy as np
from pathlib import Path
import csv
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
REAL_PHOTOS_DIR = Path("real_photos")
OUTPUT_CSV      = Path("abaca_pipeline/rhs_colors.csv")

BORDER_FRACTION  = 0.08
GLARE_THRESHOLD  = 210
MIN_VALID_PIXELS = 100

# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_filename(path: Path):
    stem = path.stem.upper()
    if "_GROUP_" not in stem:
        return None, None
    parts = stem.split("_GROUP_")
    if len(parts) != 2:
        return None, None
    color_name = parts[0].strip().replace('-', '').replace(' ', '')
    rhs_code   = parts[1].strip()
    if not color_name or not rhs_code:
        return None, None
    return color_name, rhs_code


def detect_circle_mask(gray, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=img_h // 4,
        param1=50, param2=30,
        minRadius=int(min(img_h, img_w) * 0.05),
        maxRadius=int(min(img_h, img_w) * 0.35),
    )
    if circles is not None:
        cx, cy, r = np.round(circles[0, 0]).astype(int)
        cv2.circle(mask, (cx, cy), int(r * 1.2), 1, -1)
    else:
        cx, cy = img_w // 2, img_h // 2
        r = int(min(img_h, img_w) * 0.18)
        cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask.astype(bool)


def extract_median_rgb(image_path: Path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    bh, bw = int(h * BORDER_FRACTION), int(w * BORDER_FRACTION)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[bh:h-bh, bw:w-bw] = True

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hole_mask = detect_circle_mask(gray, h, w)

    lum = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]
    glare_mask = lum > GLARE_THRESHOLD

    valid_mask = border_mask & ~hole_mask & ~glare_mask
    valid_pixels = img_rgb[valid_mask]

    if len(valid_pixels) < MIN_VALID_PIXELS:
        valid_mask2 = border_mask & ~hole_mask
        valid_pixels = img_rgb[valid_mask2]
        if len(valid_pixels) < MIN_VALID_PIXELS:
            return None

    R = int(np.median(valid_pixels[:, 0]))
    G = int(np.median(valid_pixels[:, 1]))
    B = int(np.median(valid_pixels[:, 2]))
    return R, G, B


def rgb_to_lab(R, G, B):
    """Manual sRGB -> XYZ -> Lab conversion. Avoids OpenCV overflow issues."""
    r, g, b = R / 255.0, G / 255.0, B / 255.0

    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = linearize(r), linearize(g), linearize(b)

    # sRGB to XYZ D65
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Normalize by D65 white point
    X /= 0.95047
    Y /= 1.00000
    Z /= 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)

    fx, fy, fz = f(X), f(Y), f(Z)
    L = round(116 * fy - 16, 2)
    a = round(500 * (fx - fy), 2)
    b_val = round(200 * (fy - fz), 2)
    return L, a, b_val


def rgb_to_hex(R, G, B):
    return f"#{R:02x}{G:02x}{B:02x}"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Abaca Color AI — Build RHS Colors CSV  (v2)")
    print("=" * 60)

    extensions = [".jpg", ".jpeg", ".png"]
    all_images = []
    for ext in extensions:
        all_images.extend(REAL_PHOTOS_DIR.rglob(f"*{ext}"))
        all_images.extend(REAL_PHOTOS_DIR.rglob(f"*{ext.upper()}"))
    all_images = list(set(all_images))
    all_images.sort()

    print(f"\nFound {len(all_images)} image files in {REAL_PHOTOS_DIR}/")
    print(f"Output: {OUTPUT_CSV}\n")

    if not all_images:
        print("ERROR: No images found.")
        sys.exit(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    results   = []
    skipped   = []
    failed    = []
    seen_keys = set()

    for i, img_path in enumerate(all_images, 1):
        color_name, rhs_code = parse_filename(img_path)

        if color_name is None:
            skipped.append(str(img_path))
            continue

        # Unique class label = COLORNAME_RHSCODE (preserves duplicates across fans)
        class_label = f"{color_name}_{rhs_code}"

        if class_label in seen_keys:
            continue
        seen_keys.add(class_label)

        rgb = extract_median_rgb(img_path)
        if rgb is None:
            failed.append(img_path.name)
            continue

        R, G, B = rgb
        L, a, b = rgb_to_lab(R, G, B)

        results.append({
            "rhs_code":    rhs_code,
            "class_label": class_label,
            "color_name":  color_name,
            "R": R, "G": G, "B": B,
            "Lab_L": L, "Lab_a": a, "Lab_b": b,
            "hex": rgb_to_hex(R, G, B),
            "source": "photo_extracted",
            "filename": img_path.name,
        })

        if i % 50 == 0 or i == len(all_images):
            print(f"  [{i}/{len(all_images)}] Processed {len(results)} colors...")

    results.sort(key=lambda x: x["class_label"])

    fieldnames = ["rhs_code", "class_label", "color_name", "R", "G", "B",
                  "Lab_L", "Lab_a", "Lab_b", "hex", "source", "filename"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  ✅ Successfully extracted : {len(results)} colors")
    print(f"  ⚠️  Skipped (bad filename) : {len(skipped)}")
    print(f"  ❌ Failed (extraction)    : {len(failed)}")

    if failed:
        print(f"\n  Failed files:")
        for f_ in failed:
            print(f"    {f_}")

    if results:
        L_vals = [r["Lab_L"] for r in results]
        print(f"\n  Lab_L range : {min(L_vals):.1f} → {max(L_vals):.1f}")
        print(f"  Expected    : ~5.0 (black) → ~95.0 (white)")

        groups = {}
        for r in results:
            g = r["color_name"]
            groups[g] = groups.get(g, 0) + 1
        print(f"\n  Color groups: {len(groups)}")
        for g, count in sorted(groups.items()):
            print(f"    {g:<35} {count} colors")

    print(f"\n  Total class labels: {len(results)}")
    print(f"  Output saved to   : {OUTPUT_CSV}")
    print("\n  ✅ Done! Run process_real_photos.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()