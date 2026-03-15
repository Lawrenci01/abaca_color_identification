"""
process_real_photos.py
======================
Parallel Card Color Extractor — uses multiple CPU cores for speed.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

REAL_PHOTOS_DIR  = Path("real_photos")
SWATCHES_DIR     = Path("swatches_real")
SWATCH_SIZE      = 96
BORDER_FRACTION  = 0.06
GLARE_THRESHOLD  = 210
MIN_VALID_PIXELS = 200

def detect_circle_hough(gray, img_h, img_w):
    blurred = cv2.GaussianBlur(gray, (11, 11), 2)
    min_r   = int(min(img_h, img_w) * 0.08)
    max_r   = int(min(img_h, img_w) * 0.40)
    for param2 in [25, 18, 12]:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=img_h // 3,
                                  param1=60, param2=param2, minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            cx, cy, r = np.round(circles[0, 0]).astype(int)
            return int(cx), int(cy), int(r)
    return None

def detect_circle_contour(gray, img_h, img_w):
    blurred  = cv2.GaussianBlur(gray, (11, 11), 2)
    min_r    = int(min(img_h, img_w) * 0.08)
    max_r    = int(min(img_h, img_w) * 0.40)
    min_area = np.pi * min_r ** 2
    max_area = np.pi * max_r ** 2
    best, best_score = None, -1
    threshold_attempts = []
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_attempts.extend([otsu, cv2.bitwise_not(otsu)])
    for t in [30, 60, 100, 150, 200]:
        _, th = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY)
        threshold_attempts.extend([th, cv2.bitwise_not(th)])
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold_attempts.extend([adaptive, cv2.bitwise_not(adaptive)])
    for thresh_img in threshold_attempts:
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area * 0.5 < area < max_area * 1.5): continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if not (min_r < r < max_r): continue
            circularity = area / (np.pi * r ** 2)
            if 0.5 < circularity < 1.3 and circularity > best_score:
                best_score = circularity
                best = (int(cx), int(cy), int(r))
    return best

def detect_circle(gray, img_h, img_w):
    result = detect_circle_hough(gray, img_h, img_w)
    if result is not None: return result, 'hough'
    result = detect_circle_contour(gray, img_h, img_w)
    if result is not None: return result, 'contour'
    return (img_w // 2, img_h // 2, int(min(img_h, img_w) * 0.18)), 'fallback'

def build_card_mask(img_rgb, gray):
    h, w   = img_rgb.shape[:2]
    bh, bw = int(h * BORDER_FRACTION), int(w * BORDER_FRACTION)
    mask = np.zeros((h, w), dtype=bool)
    mask[bh:h-bh, bw:w-bw] = True
    (cx, cy, r), method = detect_circle(gray, h, w)
    pad_r = int(r * 1.20)
    Y, X  = np.ogrid[:h, :w]
    mask[(X - cx) ** 2 + (Y - cy) ** 2 <= pad_r ** 2] = False
    lum = img_rgb[:, :, 0] * 0.299 + img_rgb[:, :, 1] * 0.587 + img_rgb[:, :, 2] * 0.114
    mask[lum > GLARE_THRESHOLD] = False
    return mask, (cx, cy, r), method

def fill_circle_inpaint(img_rgb, cx, cy, r):
    h, w      = img_rgb.shape[:2]
    pad_r     = int(r * 1.25)
    hole_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hole_mask, (cx, cy), pad_r, 255, -1)
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, hole_mask, inpaintRadius=int(r * 0.5), flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

def extract_and_save_swatch(image_path, out_path):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = img_rgb.shape[:2]
    bh, bw  = int(h * BORDER_FRACTION), int(w * BORDER_FRACTION)
    mask, (cx, cy, r), method = build_card_mask(img_rgb, gray)
    pixels = img_rgb[mask]
    if len(pixels) < MIN_VALID_PIXELS:
        pixels = img_rgb[bh:h-bh, bw:w-bw].reshape(-1, 3)
        if len(pixels) < MIN_VALID_PIXELS: return None
    R, G, B = [int(np.median(pixels[:, c])) for c in range(3)]
    img_rgb = fill_circle_inpaint(img_rgb, cx, cy, r)
    lum = img_rgb[:, :, 0] * 0.299 + img_rgb[:, :, 1] * 0.587 + img_rgb[:, :, 2] * 0.114
    img_rgb[lum > GLARE_THRESHOLD] = [R, G, B]
    patch = img_rgb[bh:h-bh, bw:w-bw]
    Image.fromarray(patch).resize((SWATCH_SIZE, SWATCH_SIZE), Image.LANCZOS).save(out_path)
    return R, G, B, method

def parse_filename(path):
    stem = path.stem.upper()
    if "_GROUP_" not in stem: return None, None
    parts = stem.split("_GROUP_")
    if len(parts) != 2: return None, None
    color_name = parts[0].strip().replace('-', '').replace(' ', '')
    rhs_code   = parts[1].strip()
    return color_name, rhs_code

def process_single_image(img_path):
    color_name, rhs_code = parse_filename(img_path)
    if color_name is None: return 'skipped', None
    class_label = f"{color_name}_{rhs_code}"
    out_path    = SWATCHES_DIR / f"{class_label}.png"
    try:
        result = extract_and_save_swatch(img_path, out_path)
        if result is None: return 'failed', img_path.name
        return 'saved', (class_label, result[3])
    except Exception as e: return 'error', f"{img_path.name}: {e}"

def main():
    SWATCHES_DIR.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png"}
    all_images = sorted([p for p in REAL_PHOTOS_DIR.rglob("*") if p.suffix.lower() in extensions])
    print(f"{'='*65}\n  process_real_photos.py — Parallel Texture Swatch Extractor\n{'='*65}")
    print(f"  Images  : {len(all_images)} found\n  CPUs    : {os.cpu_count()} detected\n")
    saved = skipped = failed = 0
    hough_n = contour_n = fallback_n = 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_image, p): p for p in all_images}
        for i, future in enumerate(as_completed(futures), 1):
            status, data = future.result()
            if status == 'saved':
                saved += 1
                lbl, method = data
                if method == 'hough': hough_n += 1
                elif method == 'contour': contour_n += 1
                else: fallback_n += 1
                if saved % 50 == 0: print(f"  [{i:>4}/{len(all_images)}] Saved: {lbl}.png ({method})")
            elif status == 'skipped': skipped += 1
            else:
                failed += 1
                print(f"  [{i:>4}/{len(all_images)}] FAILED: {data}")
    print(f"\n{'='*65}\n  Saved: {saved} | Skipped: {skipped} | Failed: {failed}")
    print(f"  Detection: Hough={hough_n}, Contour={contour_n}, Fallback={fallback_n}\n{'='*65}")

if __name__ == "__main__":
    main()