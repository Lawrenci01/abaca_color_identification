"""
features.py — Shared Feature Extraction Module
================================================
Single source of truth for:
  • rgb_to_lab()      — RGB → CIE Lab color conversion
  • extract_features() — 165-feature vector extraction

Previously this code was copy-pasted identically into:
  train_model.py, evaluate.py, inference_server.py

Now all three import from here. Any change to feature extraction
only needs to be made once — no more silent drift between files.

Feature vector breakdown (165 total):
  [3]  Mean Lab (Delta-E normalized)     — perceptual color
  [3]  Std RGB                           — texture roughness
  [12] Lab per quadrant                  — spatial color layout
  [96] RGB histograms  32 bins × 3      — fine color resolution
  [32] HSV histograms  16H + 8S + 8V   — hue/saturation
  [10] LBP texture                       — fiber micro-texture
  [8]  Gabor texture  4 orientations    — fiber direction
  [1]  Delta-E std deviation             — color consistency
"""

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.filters import gabor


# ── Color conversion ───────────────────────────────────────────────────────────

def rgb_to_lab(r, g, b):
    """
    Convert sRGB (0–255) to CIE Lab (D65 illuminant).
    Returns (L, a, b) as floats.
    """
    def linearize(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl = linearize(float(r))
    gl = linearize(float(g))
    bl = linearize(float(b))

    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(img):
    """
    Extract a 165-dimensional feature vector from a PIL Image.

    All three pipeline scripts (train_model, evaluate, inference_server)
    must use this exact function to guarantee consistency between
    training and inference.

    Args:
        img: PIL.Image — any size, any mode (auto-converted to RGB 64×64)

    Returns:
        np.ndarray of shape (165,) dtype float32
    """
    img = img.convert('RGB').resize((64, 64))
    arr = np.array(img, dtype=np.float32)

    # ── 1. Mean Lab (Delta-E normalized) ──────────────────────────────────
    mr = arr[:, :, 0].mean()
    mg = arr[:, :, 1].mean()
    mb = arr[:, :, 2].mean()
    L, a, b    = rgb_to_lab(mr, mg, mb)
    mean_lab   = [L / 100.0, a / 128.0, b / 128.0]

    # ── 2. Std RGB ─────────────────────────────────────────────────────────
    std_rgb = [
        arr[:, :, 0].std() / 255.0,
        arr[:, :, 1].std() / 255.0,
        arr[:, :, 2].std() / 255.0,
    ]

    # ── 3. Lab per quadrant (Delta-E normalized) ───────────────────────────
    h, w = arr.shape[:2]
    quad_feats = []
    for rs in [slice(0, h // 2), slice(h // 2, h)]:
        for cs in [slice(0, w // 2), slice(w // 2, w)]:
            q = arr[rs, cs]
            qL, qa, qb = rgb_to_lab(
                q[:, :, 0].mean(),
                q[:, :, 1].mean(),
                q[:, :, 2].mean()
            )
            quad_feats.extend([qL / 100.0, qa / 128.0, qb / 128.0])

    # ── 4. RGB histograms (32 bins) ────────────────────────────────────────
    rgb_hist = []
    for ch in range(3):
        h32, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 255))
        rgb_hist.extend(h32 / (h32.sum() + 1e-9))

    # ── 5. HSV histograms ─────────────────────────────────────────────────
    arr_norm = arr / 255.0
    R, G, B  = arr_norm[:, :, 0], arr_norm[:, :, 1], arr_norm[:, :, 2]
    Cmax  = np.maximum(np.maximum(R, G), B)
    Cmin  = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin + 1e-9
    H     = np.zeros_like(R)
    mask_r = (Cmax == R)
    mask_g = (Cmax == G)
    mask_b = (Cmax == B)
    H[mask_r] = ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6
    H[mask_g] = (B[mask_g] - R[mask_g]) / delta[mask_g] + 2
    H[mask_b] = (R[mask_b] - G[mask_b]) / delta[mask_b] + 4
    H = (H / 6.0) * 255.0
    S = np.where(Cmax > 0, delta / (Cmax + 1e-9), 0) * 255.0
    V = Cmax * 255.0
    h_hue, _ = np.histogram(H, bins=16, range=(0, 255))
    h_sat, _ = np.histogram(S, bins=8,  range=(0, 255))
    h_val, _ = np.histogram(V, bins=8,  range=(0, 255))
    hsv_hist  = list(h_hue / (h_hue.sum() + 1e-9))
    hsv_hist += list(h_sat / (h_sat.sum() + 1e-9))
    hsv_hist += list(h_val / (h_val.sum() + 1e-9))

    # ── 6. LBP texture ────────────────────────────────────────────────────
    gray        = np.array(img.convert('L'), dtype=np.uint8)
    lbp         = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_feats   = list(lbp_hist / (lbp_hist.sum() + 1e-9))

    # ── 7. Gabor texture (4 orientations — fiber direction) ───────────────
    gray_float  = gray.astype(np.float32) / 255.0
    gabor_feats = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        filt_real, filt_imag = gabor(gray_float, frequency=0.3, theta=theta)
        gabor_mag = np.sqrt(filt_real ** 2 + filt_imag ** 2)
        gabor_feats.append(float(gabor_mag.mean()))
        gabor_feats.append(float(gabor_mag.std()))

    # ── 8. Delta-E color consistency ──────────────────────────────────────
    lab_pixels = []
    for y in range(0, 64, 8):
        for x in range(0, 64, 8):
            pL, pa, pb_val = rgb_to_lab(
                int(arr[y, x, 0]),   # explicit int cast — avoids float precision issues
                int(arr[y, x, 1]),
                int(arr[y, x, 2])
            )
            lab_pixels.append([pL, pa, pb_val])
    lab_pixels   = np.array(lab_pixels)
    lab_mean     = lab_pixels.mean(axis=0)
    delta_e_vals = np.sqrt(((lab_pixels - lab_mean) ** 2).sum(axis=1))
    delta_e_std  = [float(delta_e_vals.std()) / 50.0]

    # ── Concatenate all features ───────────────────────────────────────────
    features = (mean_lab + std_rgb + quad_feats +
                rgb_hist + hsv_hist + lbp_feats +
                gabor_feats + delta_e_std)
    return np.array(features, dtype=np.float32)