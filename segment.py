# segment.py  —  Abaca Color Scanner
# Fiber segmentation: isolate dark fiber pixels from white background
#
# v2 — complete rewrite of GrabCut approach
#
# Why GrabCut was removed
# ───────────────────────
# The original implementation used OpenCV GrabCut initialised with a
# centre rectangle.  Abaca scan images violate every assumption GrabCut
# relies on:
#
#   1. Spatial prior broken — GrabCut assumes the foreground is in the
#      centre rect and the background is on the border.  In abaca scans
#      the background (white paper/scanner surface) is ~70-80 % of every
#      region including the centre.
#
#   2. Wrong polarity — GrabCut labelled the WHITE BACKGROUND as
#      foreground (FG mean luminance = 228-236, vs expected <80 for fiber).
#      seg_found=True was being returned with a completely inverted mask.
#
#   3. Masked gray fill poisoned the MLP — the 128,128,128 fill for
#      background pixels made up ~80 % of the 96×96 feature image.
#      The MLP was never trained on such images → distribution shift.
#
#   4. Speed — GrabCut on 923×2000 px took 15-28 s per image on CPU.
#      This is unacceptable for a mobile scanner.
#
# New approach: Luminance Otsu + Morphological cleanup
# ─────────────────────────────────────────────────────
# Abaca scan images have a strongly BIMODAL luminance distribution:
#   • Background : lum 220-255  (~70-80 % of pixels)
#   • Fiber      : lum  15-180  (~20-30 % of pixels)
#
# This bimodality makes Otsu's threshold extremely reliable, fast (< 0.2 s),
# and physically correct (no spatial priors required).
#
# Pipeline
# ────────
# 1. Resize to 400 px wide working copy (speed + noise reduction)
# 2. Compute luminance → Otsu threshold → dark pixels = fiber
# 3. Validate Otsu result (threshold must be ≥ 100, coverage 3-85 %)
# 4. Morphological open (remove noise) + close (fill holes)
# 5. Remove tiny connected components (dust, specks)
# 6. Scale mask back to original resolution
# 7. Sanity-check: FG mean luminance must be < 180
#    If it isn't, fall back to fixed percentile threshold and set
#    seg_found=False (honest reporting)
# 8. Apply mask — background filled with RGB(128,128,128) for
#    compatibility with features.py (unchanged contract)

import cv2
import numpy as np
from PIL import Image

# ── Tuning constants ──────────────────────────────────────────────────────────
_WORK_W          = 400    # working resolution width (px)
_OTSU_MIN_THRESH = 60     # below this → no white background → use fallback (lowered for farm/outdoor photos)
_COV_MIN         = 0.03   # minimum valid fiber coverage (3 %)
_COV_MAX         = 0.85   # maximum valid fiber coverage (85 %)
_MORPH_FRAC      = 0.02   # morphology kernel size as fraction of min dimension
_MIN_COMP_FRAC   = 0.001  # minimum connected-component area (0.1 % of image)
_FG_LUM_MAX      = 180    # if FG mean lum > this → mask captured background
_FALLBACK_PCT    = 65     # luminance percentile for fixed-threshold fallback (lowered to reduce soil/shadow inclusion)
_FALLBACK_CAP    = 200    # hard cap on fallback threshold (exclude near-white)
_BG_FILL         = (128, 128, 128)  # gray fill for masked-out background


def _morph_cleanup(binary: np.ndarray, h: int, w: int,
                   coverage: float = 0.5) -> np.ndarray:
    """
    Open → remove noise speckles.  Close → fill small fiber holes.

    Kernel size is adaptive: sparse fiber (low coverage) uses a smaller
    kernel to avoid eroding thin strands; dense fiber (high coverage)
    uses a larger close kernel to fill internal shadow gaps.
    """
    min_dim = min(h, w)
    # Base kernel — scales with image size
    k_base = max(3, int(min_dim * _MORPH_FRAC))

    # For sparse fiber, shrink open kernel so we don't erode small strands
    # For dense fiber, grow close kernel to fill shadow holes inside mass
    open_k  = max(3, int(k_base * (0.5 + 0.5 * coverage)))
    close_k = max(3, int(k_base * (1.0 + 0.5 * coverage)))

    open_k  = open_k  if open_k  % 2 == 1 else open_k  + 1
    close_k = close_k if close_k % 2 == 1 else close_k + 1

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  k_open,  iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Revert if cleanup removed too much
    if cleaned.sum() < binary.sum() * 0.3:
        return binary
    return cleaned


def _remove_small_components(binary: np.ndarray) -> np.ndarray:
    """Drop connected components smaller than _MIN_COMP_FRAC of the image."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    min_area = binary.size * _MIN_COMP_FRAC
    clean = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean if clean.sum() > 0 else binary


def _lum_fallback(arr_full: np.ndarray) -> np.ndarray:
    """
    Fixed-percentile luminance threshold — last resort when Otsu fails.
    Returns a uint8 binary mask (1 = fiber) at original resolution.

    For cropped images (little white background), raises the percentile
    cutoff so mid-tone fiber pixels are not excluded.
    """
    lum = (arr_full[:, :, 0] * 0.299 +
           arr_full[:, :, 1] * 0.587 +
           arr_full[:, :, 2] * 0.114)
    # Use a higher percentile for cropped images (few near-white pixels)
    white_frac = float((lum > 220).sum()) / lum.size
    pct = 85 if white_frac < 0.15 else _FALLBACK_PCT
    thresh = min(float(np.percentile(lum, pct)), _FALLBACK_CAP)
    return (lum < thresh).astype(np.uint8)


def segment_fiber(img: Image.Image) -> tuple:
    """
    Segment the abaca fiber from the white scanner background.

    Parameters
    ----------
    img : PIL.Image
        Raw scan image (any size, any mode).

    Returns
    -------
    masked_img  : PIL.Image  — original pixels where fiber, gray fill elsewhere
    seg_found   : bool       — True = Otsu succeeded and mask is reliable
                               False = fell back to fixed-percentile threshold
    coverage    : float      — fraction of image pixels identified as fiber (0-1)
    binary      : np.ndarray — H×W uint8 mask (1 = fiber, 0 = background)

    Notes
    -----
    • seg_found=False does NOT mean failure — the fallback mask is still
      used and is usually accurate for these high-contrast images.
    • coverage is returned as a 0-1 fraction; features.py multiplies by 100.
    • The gray fill (128,128,128) is kept for API compatibility with
      features.py which ignores it when extracting the dominant colour
      (it uses _extract_foreground_color on the ORIGINAL image, not masked_img).
    """
    img_rgb  = img.convert("RGB")
    arr_orig = np.array(img_rgb, dtype=np.uint8)
    H, W     = arr_orig.shape[:2]

    if W < 30 or H < 30:
        ones = np.ones((H, W), dtype=np.uint8)
        return img_rgb, False, 1.0, ones

    # ── 1. Resize to working resolution ──────────────────────────────────────
    scale    = _WORK_W / W
    work_h   = max(30, int(H * scale))
    arr_work = cv2.resize(arr_orig, (_WORK_W, work_h),
                          interpolation=cv2.INTER_AREA)

    # ── 2. Grayscale + Otsu ───────────────────────────────────────────────────
    gray_work = cv2.cvtColor(arr_work, cv2.COLOR_RGB2GRAY)

    # Mask out glare pixels (lum > 210) before Otsu so specular reflections
    # on the fiber surface do not pull the threshold toward white.
    glare_pixels = gray_work > 210
    gray_no_glare = gray_work.copy()
    gray_no_glare[glare_pixels] = 0   # treat glare as background for Otsu

    otsu_val, binary_work = cv2.threshold(
        gray_no_glare, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Glare pixels are NOT fiber — force them to background in the mask
    binary_work[glare_pixels] = 0

    # ── 3. Validate Otsu result ───────────────────────────────────────────────
    raw_cov   = float((binary_work > 0).sum()) / binary_work.size
    otsu_ok   = (otsu_val >= _OTSU_MIN_THRESH and
                 _COV_MIN <= raw_cov <= _COV_MAX)

    if not otsu_ok:
        # Otsu failed (image has no white background or is all-white)
        # → fixed luminance threshold at working resolution
        lum_work  = (arr_work[:, :, 0] * 0.299 +
                     arr_work[:, :, 1] * 0.587 +
                     arr_work[:, :, 2] * 0.114)
        # If near-white pixels are rare, this is a cropped fiber image —
        # use a higher percentile so we don't cut away real fiber pixels.
        white_frac_w = float((lum_work > 220).sum()) / lum_work.size
        fallback_pct = 85 if white_frac_w < 0.15 else _FALLBACK_PCT
        thresh = min(float(np.percentile(lum_work, fallback_pct)), _FALLBACK_CAP)
        binary_work = ((lum_work < thresh).astype(np.uint8) * 255)

    # ── 4. Morphological cleanup ──────────────────────────────────────────────
    raw_coverage = float((binary_work > 0).sum()) / binary_work.size
    binary_work = _morph_cleanup(binary_work, work_h, _WORK_W, coverage=raw_coverage)

    # ── 5. Remove tiny components ─────────────────────────────────────────────
    binary_work = _remove_small_components(binary_work)

    # ── 6. Scale mask to original resolution ─────────────────────────────────
    binary_full = cv2.resize(binary_work, (W, H),
                             interpolation=cv2.INTER_NEAREST)
    binary_full = (binary_full > 127).astype(np.uint8)

    # ── 7. Sanity check — ensure FG is actually DARK (fiber, not background) ─
    lum_orig = (arr_orig[:, :, 0].astype(float) * 0.299 +
                arr_orig[:, :, 1].astype(float) * 0.587 +
                arr_orig[:, :, 2].astype(float) * 0.114)
    fg_bool  = binary_full.astype(bool)
    fg_lum   = lum_orig[fg_bool].mean() if fg_bool.any() else 255.0

    if fg_lum > _FG_LUM_MAX:
        # Mask captured the bright background — invert or use fallback
        inverted    = (binary_full == 0).astype(np.uint8)
        inv_fg_lum  = lum_orig[inverted.astype(bool)].mean() if inverted.any() else 255.0

        if inv_fg_lum < fg_lum and inv_fg_lum < _FG_LUM_MAX:
            # Inverted mask is better
            binary_full = inverted
            fg_lum      = inv_fg_lum
            seg_found   = True
        else:
            # Neither polarity is correct → fixed percentile fallback
            binary_full = _lum_fallback(arr_orig)
            seg_found   = False   # honest: Otsu failed for this image
    else:
        seg_found = otsu_ok   # True if Otsu succeeded and passed validation

    coverage = float(binary_full.sum()) / float(H * W)

    # ── 8. Apply mask ─────────────────────────────────────────────────────────
    mask_3ch   = np.stack([binary_full] * 3, axis=-1)
    bg_fill    = np.array(_BG_FILL, dtype=np.uint8)
    masked_arr = np.where(mask_3ch, arr_orig, bg_fill)
    masked_img = Image.fromarray(masked_arr)

    return masked_img, seg_found, coverage, binary_full


def mask_median_color(img: Image.Image, mask: np.ndarray):
    """
    Return the dominant fiber RGB using shadow-aware grid sampling.

    Problem with simple median: shadow pixels on fiber edges are darker
    and more desaturated than the true fiber surface color, dragging the
    mean Lab away from the correct RHS grade.

    Solution: divide the fiber mask into a 3×3 grid, compute the median
    RGB for each cell, then discard cells whose luminance is more than
    1 standard deviation below the grid median (shadow cells).  Average
    the remaining cells.  This converges on the true fiber surface color
    rather than a shadow-contaminated mean.

    Returns None if fewer than 50 fiber pixels are found.
    """
    arr          = np.array(img.convert("RGB"), dtype=np.float32)
    H, W         = arr.shape[:2]
    fiber_pixels = arr[mask == 1]

    if len(fiber_pixels) < 50:
        return None

    # Compute per-cell medians on a 3×3 grid
    rows = np.array_split(np.arange(H), 3)
    cols = np.array_split(np.arange(W), 3)
    cell_medians = []

    for rr in rows:
        for cc in cols:
            cell_mask = mask[np.ix_(rr, cc)]
            cell_arr  = arr[np.ix_(rr, cc)]
            fp = cell_arr[cell_mask == 1]
            if len(fp) < 10:
                continue
            med_r = float(np.median(fp[:, 0]))
            med_g = float(np.median(fp[:, 1]))
            med_b = float(np.median(fp[:, 2]))
            lum   = med_r * 0.299 + med_g * 0.587 + med_b * 0.114
            cell_medians.append((lum, med_r, med_g, med_b))

    if not cell_medians:
        # Fallback: simple whole-mask median
        r = int(np.median(fiber_pixels[:, 0]))
        g = int(np.median(fiber_pixels[:, 1]))
        b = int(np.median(fiber_pixels[:, 2]))
        return r, g, b

    lums = np.array([c[0] for c in cell_medians])
    lum_med = float(np.median(lums))
    lum_std = float(np.std(lums)) if len(lums) > 1 else 0.0

    # Keep cells within 1 std of median luminance (exclude dark shadow cells)
    threshold = lum_med - max(lum_std, 5.0)
    good_cells = [(r, g, b) for lum, r, g, b in cell_medians if lum >= threshold]

    if not good_cells:
        good_cells = [(r, g, b) for _, r, g, b in cell_medians]

    r = int(round(np.mean([c[0] for c in good_cells])))
    g = int(round(np.mean([c[1] for c in good_cells])))
    b = int(round(np.mean([c[2] for c in good_cells])))
    return r, g, b


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python segment.py <image.jpg>")
        sys.exit(0)

    img    = Image.open(sys.argv[1])
    masked, found, cov, mask = segment_fiber(img)

    print(f"seg_found={found}   coverage={cov:.1%}")
    masked.save("seg_result.jpg")
    print("Saved: seg_result.jpg")

    rgb = mask_median_color(img, mask)
    if rgb:
        print(f"Median fiber RGB: R={rgb[0]}  G={rgb[1]}  B={rgb[2]}")