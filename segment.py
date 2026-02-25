"""
segment.py — CPU Fiber Segmentation via GrabCut
=================================================
Replaces Mask R-CNN (too slow on CPU) with OpenCV GrabCut.

Speed comparison:
  Mask R-CNN on CPU : 30–60 seconds  ← unusable
  GrabCut on CPU    :  0.3–0.8s      ← fast enough

Strategy (3-attempt cascade):
  1. GrabCut with tight center seed (50% of crop)
  2. If that fails → GrabCut with even tighter seed (40%)
  3. If both fail → color-based fallback using dominant color clustering
     to separate fiber pixels from background without any seed

Install: pip install opencv-python-headless
"""

import cv2
import numpy as np
from PIL import Image


def _run_grabcut(img_bgr: np.ndarray, H: int, W: int,
                 margin_frac: float, iters: int = 5) -> np.ndarray:
    """
    Run one GrabCut attempt with a given margin fraction.
    Returns binary mask (1=fiber, 0=bg) or None if it failed/degenerate.
    """
    mx = max(4, int(W * margin_frac))
    my = max(4, int(H * margin_frac))
    rw = W - 2 * mx
    rh = H - 2 * my
    if rw < 10 or rh < 10:
        return None

    gc_mask   = np.zeros((H, W), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, gc_mask, (mx, my, rw, rh),
                    bgd_model, fgd_model,
                    iterCount=iters,
                    mode=cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None

    binary   = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        1, 0
    ).astype(np.uint8)
    coverage = float(binary.sum()) / (H * W)

    # Reject degenerate results
    if coverage < 0.08 or coverage > 0.97:
        return None

    return binary


def _color_fallback(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    When GrabCut fails, use a simple color-space approach:
    - Sample the center 40% of the crop as "likely fiber"
    - Sample the border 15% as "likely background"
    - Build per-pixel color distance to each and threshold

    Not as clean as GrabCut but better than returning the raw crop
    when there's a cluttered background (rocks, leaves, card edges).
    """
    # Center region = fiber seed
    cy0 = int(H * 0.30); cy1 = int(H * 0.70)
    cx0 = int(W * 0.30); cx1 = int(W * 0.70)
    center_pixels = arr[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float32)

    # Border strip = background seed
    border_mask = np.zeros((H, W), dtype=bool)
    b = max(2, int(min(H, W) * 0.12))
    border_mask[:b, :]  = True
    border_mask[-b:, :] = True
    border_mask[:, :b]  = True
    border_mask[:, -b:] = True
    border_pixels = arr[border_mask].astype(np.float32)

    if len(center_pixels) < 10 or len(border_pixels) < 10:
        return np.ones((H, W), dtype=np.uint8)

    # Mean color of each region
    fg_color = center_pixels.mean(axis=0)
    bg_color = border_pixels.mean(axis=0)

    # Per-pixel: distance to fiber vs distance to background
    pixels  = arr.reshape(-1, 3).astype(np.float32)
    d_fg    = np.sqrt(((pixels - fg_color) ** 2).sum(axis=1))
    d_bg    = np.sqrt(((pixels - bg_color) ** 2).sum(axis=1))

    # Pixel is fiber if it's closer to fiber center than to border
    binary  = (d_fg < d_bg).astype(np.uint8).reshape(H, W)
    coverage = float(binary.sum()) / (H * W)

    if coverage < 0.05 or coverage > 0.97:
        return np.ones((H, W), dtype=np.uint8)

    return binary


def segment_fiber(img: Image.Image) -> tuple:
    """
    Segment fiber pixels from background.

    Returns:
        masked_img : PIL Image with background = (128,128,128)
        found      : bool — True if a real segmentation was applied
        coverage   : float — fraction of image identified as fiber
        mask       : np.ndarray (H,W) uint8 — 1=fiber, 0=background
    """
    img_rgb = img.convert('RGB')
    arr     = np.array(img_rgb)
    H, W    = arr.shape[:2]

    if W < 30 or H < 30:
        return img_rgb, False, 1.0, np.ones((H, W), dtype=np.uint8)

    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ── Attempt 1: tight seed at 25% margin ───────────────────────────────
    binary = _run_grabcut(img_bgr, H, W, margin_frac=0.25, iters=5)

    # ── Attempt 2: even tighter seed at 30% margin ────────────────────────
    if binary is None:
        binary = _run_grabcut(img_bgr, H, W, margin_frac=0.30, iters=8)

    # ── Attempt 3: color-distance fallback ────────────────────────────────
    seg_found = True
    if binary is None:
        binary    = _color_fallback(arr, H, W)
        seg_found = False   # mark as fallback so UI shows warning

    coverage = float(binary.sum()) / (H * W)

    # Apply mask: background → neutral gray (128,128,128)
    # This value is explicitly excluded in extract_dominant_color()
    mask_3ch   = np.stack([binary] * 3, axis=-1)
    bg_gray    = np.array([128, 128, 128], dtype=np.uint8)
    masked_arr = np.where(mask_3ch, arr, bg_gray)
    masked_img = Image.fromarray(masked_arr.astype(np.uint8))

    return masked_img, seg_found, coverage, binary


def mask_median_color(img: Image.Image, mask: np.ndarray):
    """
    Return the median RGB color of fiber pixels only.
    More robust than mean against leftover background pixels.
    Returns None if mask has fewer than 50 fiber pixels.
    """
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    fiber_pixels = arr[mask == 1]
    if len(fiber_pixels) < 50:
        return None
    r = int(np.median(fiber_pixels[:, 0]))
    g = int(np.median(fiber_pixels[:, 1]))
    b = int(np.median(fiber_pixels[:, 2]))
    return (r, g, b)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1])
        masked, found, cov, mask = segment_fiber(img)
        print(f"Found={found}  Coverage={cov:.1%}")
        masked.save('seg_result.jpg')
        print("Saved: seg_result.jpg")
    else:
        print("Usage: python segment.py image.jpg")