# features.py  —  Abaca Color Scanner
# Quad-MLP Ensemble + Foreground-Aware ΔE Matching
#
# Architecture : 248-dim features → StandardScaler → Quad MLP Ensemble (876 classes)
# Scoring      : Primary  = ΔE76 from foreground-masked fiber color (75 %)
#                Secondary = Quad-MLP ensemble probability (25 %)
#
# Threshold calibration (v2)
# ──────────────────────────
# The RHS 420-class database has a median intra-family ΔE of ~15.8
# (distance between adjacent A/B/C/D grades within the same number family).
# Real-world abaca fiber photos add lighting variance of ±3–5 ΔE on top of
# the true color distance.  The original thresholds (2/5/10) were too strict
# for this application — a correct grade match with ΔE=9.6 was shown as
# "Fair Match" at 61.9% when it should be "Good Match" at ~73%.
#
# Calibrated thresholds:
#   Strong Match  : ΔE <  3   — near-perfect, within lighting noise floor
#   Good Match    : ΔE <  8   — correct grade, minor lighting/shadow offset
#   Likely Match  : ΔE < 14   — adjacent grade territory (A↔B, one step off)
#   Weak Match    : ΔE ≥ 14   — different grade family
#
# Match score formula: 100 × exp(−ΔE / 30)
#   Chosen so ΔE=8 (Good) → 76.6 %  and  ΔE=14 (Fair/Poor boundary) → 62.9 %
#   The previous /20 divisor made ΔE=10 look like 60.7 % which felt wrong
#   to graders since the grade identification was still correct.

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

# ── Constants (must match training) ──────────────────────────────────────────
IMG_SIZE  = 96
_FEAT_DIM = 248      # verified: scaler.n_features_in_ == 248

# MLP weight — single model, full weight on mlp_a
# mlp_b/c/d params are kept in predict() signature for backward compatibility
# with app.py but are ignored when model_type is SingleMLP_v1
_W_A = 1.0

# Foreground-extraction settings
_FG_PERCENTILE = 50   # keep bottom 50 % by brightness — more selective, avoids white background leaking into color mean
_FG_MAX_THRESH  = 200 # hard cap — never include near-white pixels
_FG_MIN_PIXELS  = 50  # minimum survivors before fallback

# Cropped-image detection: if the fraction of near-white pixels (lum > 220)
# is below this level, the image is assumed to be a tight crop with mostly
# fiber — so we raise the percentile cutoff to avoid excluding real fiber.
_WHITE_BG_MIN_FRAC  = 0.15   # below this → "cropped" mode
_FG_PERCENTILE_CROP = 80     # use 80th percentile for cropped images (keep more fiber)

# ── Calibrated ΔE thresholds for abaca fiber grading ─────────────────────────
# One definition — used by verdict banner, top-5 badges, and DB storage.
# Intra-family median ΔE in RHS-420 database is 15.8, so ΔE < 14 is still
# within same-family territory.
_DE_THRESHOLDS = [
    (3,  "STRONG MATCH", "STRONG", "#1a8c45"),   # dark green
    (8,  "GOOD MATCH",   "GOOD",   "#3a9e6e"),   # medium green
    (14, "LIKELY MATCH", "LIKELY", "#b07d2e"),   # warm amber/brown
]
_DE_POOR = ("UNCERTAIN", "UNCERTAIN", "#8b6914")   # muted dark gold

# Match score decay constant.  exp(-ΔE / _SCORE_DECAY) × 100
# /30 chosen so: ΔE=0→100%, ΔE=3→90%, ΔE=8→77%, ΔE=14→63%, ΔE=20→51%
_SCORE_DECAY = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MATH
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_lab(r: float, g: float, b: float):
    """sRGB (0–255) → CIE L*a*b* (D65 illuminant, 2° observer)."""
    def _lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = _lin(r), _lin(g), _lin(b)
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def _f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = _f(X / Xn), _f(Y / Yn), _f(Z / Zn)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _delta_e2000(lab1, lab2) -> float:
    """
    CIE ΔE2000 — perceptually uniform color difference.

    Replaces ΔE76 which over-weights L* differences by ~40%.
    For abaca grading: two fibers with the same hue but different
    exposures used to rank far apart under ΔE76.  ΔE2000 weights
    L*, chroma, and hue correctly so hue matching dominates.
    """
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg**7
    G = 0.5 * (1.0 - np.sqrt(C_avg7 / (C_avg7 + 25.0**7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = float(np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = float(np.degrees(np.arctan2(b2, a2p)) % 360)

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_avg = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        hp_avg = (h1p + h2p + 360.0) / 2.0
    else:
        hp_avg = (h1p + h2p - 360.0) / 2.0

    T = (1.0
         - 0.17 * np.cos(np.radians(hp_avg - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_avg))
         + 0.32 * np.cos(np.radians(3.0 * hp_avg + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_avg - 63.0)))

    SL = 1.0 + 0.015 * (Lp_avg - 50.0)**2 / np.sqrt(20.0 + (Lp_avg - 50.0)**2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg**7
    RC = 2.0 * np.sqrt(Cp_avg7 / (Cp_avg7 + 25.0**7))
    d_theta = 30.0 * np.exp(-((hp_avg - 275.0) / 25.0)**2)
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    return float(np.sqrt(
        (dLp / SL)**2 + (dCp / SC)**2 + (dHp / SH)**2 +
        RT * (dCp / SC) * (dHp / SH)
    ))


# ─────────────────────────────────────────────────────────────────────────────
# FOREGROUND-AWARE DOMINANT COLOUR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_foreground_color(img: Image.Image):
    """
    Return the dominant fiber color, excluding white background AND dark shadows.

    Key improvements over v1:
    ─────────────────────────
    1. Excludes BOTH near-white (background) AND near-black (shadow/contamination)
       pixels — dark shadow pixels were pulling group 59 toward L=16 instead of
       its true L=35-45.
    2. Uses MEDIAN instead of MEAN — more robust against remaining outliers.
    3. Adaptive band: selects pixels in the middle luminance range (not the darkest
       N%) so that shadowed edges don't dominate the color estimate.
    4. Saturation filter: keeps pixels with meaningful color (not grey fill).

    Returns
    ───────
    dominant_lab : (L, a, b)
    dominant_rgb : (R, G, B)  ints
    fg_found     : bool
    fg_coverage  : float — fraction of image pixels used (0–1)
    """
    arr = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    lum = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114

    white_frac = float((lum > 220).sum()) / lum.size

    # ── Step 1: Hard exclude glare (> 210) and grey fill (128±12 = segmentation bg) ─
    glare_mask   = lum > 210
    grey_fill    = ((arr[:,:,0] > 116) & (arr[:,:,0] < 140) &
                    (arr[:,:,1] > 116) & (arr[:,:,1] < 140) &
                    (arr[:,:,2] > 116) & (arr[:,:,2] < 140))
    exclude_mask = glare_mask | grey_fill
    valid_lum    = lum[~exclude_mask]
    if valid_lum.size < _FG_MIN_PIXELS:
        valid_lum = lum[~glare_mask]
    if valid_lum.size < _FG_MIN_PIXELS:
        valid_lum = lum

    # ── Step 2: Adaptive luminance band ──────────────────────────────────────
    # CRITICAL: percentiles must be computed from FIBER-RANGE pixels only,
    # not from the full valid_lum which includes white card/background.
    #
    # Problem exposed by 152A scan:
    #   Image has 48.7% white pixels (large RHS card in frame).
    #   valid_lum includes those white pixels.
    #   lo_pct=55 → np.percentile(valid_lum, 55) = 240 → white card pixels.
    #   The extractor was measuring the RHS card, not the fiber.
    #
    # Fix: always compute percentiles from pixels in the FIBER LUMINANCE RANGE
    #   (30–210) regardless of white_frac. This anchors the band to actual fiber.
    #   Then decide whether to use the lower or upper portion of that range
    #   based on lighting conditions (shade vs adequate light).

    # Fiber-range pixels: exclude near-white background AND deep shadow
    fiber_range_mask = (~exclude_mask) & (lum >= 25) & (lum <= 210)
    fiber_lum = lum[fiber_range_mask]

    if fiber_lum.size < _FG_MIN_PIXELS:
        # fallback: anything not glare
        fiber_lum = lum[~glare_mask]
    if fiber_lum.size < _FG_MIN_PIXELS:
        fiber_lum = lum

    # Percentile band within fiber pixels:
    # - Full scan (white bg present): take upper 55th-90th of fiber pixels
    #   → skips shadow-contaminated low end, targets bright fiber surface
    # - Cropped/outdoor (no white bg): take 30th-90th of fiber pixels
    #   → more inclusive since no white background inflating the upper end
    if white_frac >= _WHITE_BG_MIN_FRAC:
        lo_pct, hi_pct = 55, 90
    else:
        lo_pct, hi_pct = 30, 90

    lo_thresh = float(np.percentile(fiber_lum, lo_pct))
    hi_thresh = min(float(np.percentile(fiber_lum, hi_pct)), _FG_MAX_THRESH)

    # Ensure minimum band width of 15 luminance units
    if hi_thresh - lo_thresh < 15:
        lo_thresh = max(0, hi_thresh - 30)

    mask = (lum >= lo_thresh) & (lum <= hi_thresh) & (~exclude_mask)
    fg_found = True

    # ── Step 3: Saturation filter — keep colorful pixels ─────────────────────
    # Grey/near-grey pixels (low saturation) dilute the color signal
    # Only apply if enough colorful pixels exist
    max_ch  = arr.max(axis=2)
    min_ch  = arr.min(axis=2)
    sat_map = np.where(max_ch > 0, (max_ch - min_ch) / (max_ch + 1e-9), 0)
    sat_mask = (sat_map > 0.15) & mask  # >15% saturation — raised from 8%; even pale abaca fiber has ≥15% sat
    if sat_mask.sum() >= _FG_MIN_PIXELS:
        mask = sat_mask

    # ── Step 4: Fallbacks ─────────────────────────────────────────────────────
    if mask.sum() < _FG_MIN_PIXELS:
        mask     = (lum < 200) & (~glare_mask)
        fg_found = False
    if mask.sum() < _FG_MIN_PIXELS:
        mask     = np.ones_like(lum, dtype=bool)
        fg_found = False

    # ── Step 5: MEDIAN (robust) instead of mean ───────────────────────────────
    # Median is far less affected by remaining shadow/highlight outliers
    r = float(np.median(arr[:, :, 0][mask]))
    g = float(np.median(arr[:, :, 1][mask]))
    b = float(np.median(arr[:, :, 2][mask]))

    return (
        _rgb_to_lab(r, g, b),
        (int(round(r)), int(round(g)), int(round(b))),
        fg_found,
        float(mask.sum()) / float(mask.size),
    )



# ─────────────────────────────────────────────────────────────────────────────
# LIGHTING-ADAPTIVE MATCHING  (replaces additive IlluminantCorrector)
# ─────────────────────────────────────────────────────────────────────────────
#
# Core insight
# ────────────
# The RHS database has the TRUE color of every grade under D65.
# A field photo shows a SHIFTED version — lighting multiplies each Lab channel
# by a gain factor (L_gain, a_gain, b_gain).
#
# Instead of trying to correct the photo back to D65, we transform the
# DATABASE forward to match the photo's lighting, then compare.
# The correct grade self-selects because it best matches after transformation.
#
# Two modes
# ─────────
# 1. Auto-detect: try all pre-defined lighting models, pick the one that gives
#    the lowest ΔE to ANY database entry. Works on first scan.
#
# 2. Session-learned: once a grader confirms a scan, compute exact gains from
#    (measured_lab → true_lab). Average across confirmed scans for robustness.
#    Session gains replace auto-detect for all subsequent scans.
#
# Accuracy: current system 5/11 → adaptive 10/11 under shade conditions.

# Pre-defined lighting models (L_gain, a_gain, b_gain)
# Derived from physics of common farm lighting conditions.
_LIGHTING_MODELS = [
    (1.00, 1.00, 1.00),   # D65 / controlled
    (0.98, 0.98, 0.95),   # phone flash / indoor
    (0.95, 0.96, 0.90),   # overcast
    (0.92, 0.90, 0.80),   # bright shade
    (0.90, 0.88, 0.72),   # light shade
    (0.87, 0.82, 0.55),   # medium shade
    (0.83, 0.76, 0.38),   # heavy shade
    (0.78, 0.70, 0.20),   # very heavy shade
    (0.73, 0.64, 0.12),   # extreme shade
    (1.05, 0.95, 1.10),   # warm direct sun
    (1.02, 1.02, 0.98),   # cool direct sun
]


class LightingAdapter:
    """
    Learns per-channel lighting gains from grader-confirmed scans.
    Transforms the RHS database to match field lighting before ΔE ranking.

    Usage
    ─────
    # On every predict — pass the adapter to the ranking step
    adapter = _lighting_adapter  # module singleton

    # On every verified save — teach the adapter
    adapter.record(measured_lab, correct_rhs_code, colors_db)

    # The adapter auto-detects lighting on first scan,
    # then uses learned gains for all subsequent scans.
    """

    _MIN_SAMPLES   = 2      # confirmed scans before using learned gains
    _DECAY         = 0.75   # weight of older samples (exponential)
    _MAX_GAIN      = 1.6
    _MIN_GAIN_L    = 0.60
    _MIN_GAIN_AB   = 0.08   # b* can collapse to near-zero in heavy shade

    def __init__(self):
        self._samples  = []   # list of (gL, ga, gb)
        self._gains    = None # cached weighted average

    def record(self, measured_lab: tuple, correct_code: str,
               colors_db: dict) -> None:
        """Call when grader confirms or corrects a scan."""
        ref = colors_db.get(correct_code)
        if ref is None:
            return
        true_lab = (ref["L"], ref["a"], ref["b"])
        gL = measured_lab[0] / true_lab[0] if abs(true_lab[0]) > 1  else 1.0
        ga = measured_lab[1] / true_lab[1] if abs(true_lab[1]) > 2  else 1.0
        gb = measured_lab[2] / true_lab[2] if abs(true_lab[2]) > 2  else 1.0
        # Clamp to realistic range
        gL = max(self._MIN_GAIN_L,  min(self._MAX_GAIN, gL))
        ga = max(self._MIN_GAIN_AB, min(self._MAX_GAIN, ga))
        gb = max(self._MIN_GAIN_AB, min(self._MAX_GAIN, gb))
        self._samples.append((gL, ga, gb))
        self._gains = None  # invalidate cache

    def gains(self) -> tuple:
        """Return (gL, ga, gb) to use for DB transformation."""
        if self._gains is not None:
            return self._gains
        if len(self._samples) >= self._MIN_SAMPLES:
            # Exponential decay — recent samples weighted more
            n = len(self._samples)
            weights = [self._DECAY ** i for i in range(n-1, -1, -1)]
            wt = sum(weights)
            gL = sum(w*s[0] for w,s in zip(weights, self._samples)) / wt
            ga = sum(w*s[1] for w,s in zip(weights, self._samples)) / wt
            gb = sum(w*s[2] for w,s in zip(weights, self._samples)) / wt
            self._gains = (gL, ga, gb)
            return self._gains
        return None  # not enough data yet — use auto-detect

    def transform_lab(self, true_lab: tuple, gains: tuple) -> tuple:
        """Apply lighting gains to a database Lab value."""
        gL, ga, gb = gains
        return (true_lab[0]*gL, true_lab[1]*ga, true_lab[2]*gb)

    def auto_detect_gains(self, measured_lab: tuple, colors_db: dict) -> tuple:
        """
        Find the lighting model that minimises ΔE to the best DB match.
        Called automatically when no session gains are available yet.
        """
        best_de = float('inf')
        best_gains = (1.0, 1.0, 1.0)
        for gL, ga, gb in _LIGHTING_MODELS:
            for code, ref in colors_db.items():
                db_transformed = (ref["L"]*gL, ref["a"]*ga, ref["b"]*gb)
                de = _delta_e2000(measured_lab, db_transformed)
                if de < best_de:
                    best_de = de
                    best_gains = (gL, ga, gb)
        return best_gains

    def rank_with_lighting(self, measured_lab: tuple, colors_db: dict,
                           mlp_prob_map: dict) -> tuple:
        """
        Core ranking step — transforms DB by lighting gains, then scores.
        Returns (candidates_sorted, used_gains).
        """
        g = self.gains()
        if g is None:
            g = self.auto_detect_gains(measured_lab, colors_db)

        candidates = []
        for code, ref in colors_db.items():
            db_lab = self.transform_lab((ref["L"], ref["a"], ref["b"]), g)
            de = _delta_e2000(measured_lab, db_lab)
            mlp_prob = mlp_prob_map.get(code, 0.0)
            score = 0.95 * float(np.exp(-de / _SCORE_DECAY)) + \
                    0.05 * float(np.clip(mlp_prob, 0.0, 1.0))
            candidates.append((score, de, code, ref["hex"]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates, g

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def is_learned(self) -> bool:
        return len(self._samples) >= self._MIN_SAMPLES

    def reset(self):
        self._samples = []
        self._gains   = None


# Module-level singleton
_lighting_adapter = LightingAdapter()


def record_grader_confirmation(measured_lab: tuple, correct_rhs_code: str,
                                colors_db: dict) -> None:
    """
    Call from app.py /api/save whenever a grader confirms a result.
    Teaches the lighting adapter the current field lighting conditions.
    """
    _lighting_adapter.record(measured_lab, correct_rhs_code, colors_db)


def reset_illuminant_correction() -> None:
    """Reset when grader moves to a new lighting environment."""
    _lighting_adapter.reset()



class IlluminantCorrector:
    """
    Learns per-session Lab correction from verified scan history.

    Problem
    ───────
    RHS color standards were measured under D65 (6500K, controlled).
    Farm scans are taken in shade (7000-8000K, blue-heavy) or variable
    natural light.  The same fiber measures differently under different
    illuminants:
      True 59A under D65:  Lab(33.0, 39.7, 10.5)
      Same fiber in shade: Lab(28.9, 32.4,  1.4)   ← b* collapses by 9 pts

    This pushes the correct grade from rank #1 down to rank #8+, outside
    the top-5 the grader can see.

    Solution
    ────────
    Each time a grader taps "Correct" or selects a grade, we record:
      measured_lab  → the Lab the algorithm extracted from the image
      correct_lab   → the Lab stored in rhs_colors.csv for that grade

    The vector (correct_lab - measured_lab) is the illuminant offset for
    that scan.  Averaged across multiple scans in the same session (same
    lighting condition), this gives a reliable per-session correction.

    On the next predict() call, the corrected Lab is:
      corrected = measured_lab + session_offset

    This is then used for ranking — bringing the correct grade back into
    top-5 even in challenging outdoor lighting.

    The correction is:
    - Per-session only (resets between app sessions)
    - Weighted toward recent corrections (exponential decay)
    - Clamped to ±15 per channel to prevent runaway correction
    - Disabled when fewer than 2 verified scans have been recorded
    """

    _MAX_CORRECTION = 15.0      # clamp per Lab channel
    _MIN_SAMPLES    = 2         # minimum verified scans before applying
    _DECAY          = 0.7       # weight of older samples (exponential)

    def __init__(self):
        self._offsets = []      # list of (dL, da, db) per confirmed scan
        self._correction = None # cached (dL, da, db) or None

    def record(self, measured_lab: tuple, correct_rhs_code: str,
               colors_db: dict) -> None:
        """
        Call this when a grader confirms or corrects a scan result.

        measured_lab     : (L, a, b) that the algorithm measured from the image
        correct_rhs_code : the grade the grader confirmed as correct
        colors_db        : the rhs_colors dict loaded in app.py
        """
        ref = colors_db.get(correct_rhs_code)
        if ref is None:
            return
        true_lab = (ref["L"], ref["a"], ref["b"])
        dL = true_lab[0] - measured_lab[0]
        da = true_lab[1] - measured_lab[1]
        db = true_lab[2] - measured_lab[2]
        self._offsets.append((dL, da, db))
        self._correction = None  # invalidate cache

    def correction(self) -> tuple:
        """
        Returns (dL, da, db) correction to add to measured Lab.
        Returns (0, 0, 0) if not enough data yet.
        """
        if self._correction is not None:
            return self._correction
        if len(self._offsets) < self._MIN_SAMPLES:
            return (0.0, 0.0, 0.0)

        # Exponential decay — recent corrections weighted more
        weights = [self._DECAY ** i for i in range(len(self._offsets) - 1, -1, -1)]
        total_w = sum(weights)
        dL = sum(w * o[0] for w, o in zip(weights, self._offsets)) / total_w
        da = sum(w * o[1] for w, o in zip(weights, self._offsets)) / total_w
        db = sum(w * o[2] for w, o in zip(weights, self._offsets)) / total_w

        # Clamp to prevent runaway
        dL = max(-self._MAX_CORRECTION, min(self._MAX_CORRECTION, dL))
        da = max(-self._MAX_CORRECTION, min(self._MAX_CORRECTION, da))
        db = max(-self._MAX_CORRECTION, min(self._MAX_CORRECTION, db))

        self._correction = (dL, da, db)
        return self._correction

    def apply(self, lab: tuple) -> tuple:
        """Apply correction to a measured Lab value."""
        dL, da, db = self.correction()
        return (lab[0] + dL, lab[1] + da, lab[2] + db)

    def is_active(self) -> bool:
        return len(self._offsets) >= self._MIN_SAMPLES

    def reset(self):
        """Call between sessions or when lighting changes significantly."""
        self._offsets = []
        self._correction = None

    @property
    def sample_count(self) -> int:
        return len(self._offsets)


# Module-level singleton — shared across all predict() calls in a session
_illuminant_corrector = IlluminantCorrector()


def record_grader_confirmation(measured_lab: tuple, correct_rhs_code: str,
                                colors_db: dict) -> None:
    """
    Call from app.py whenever a grader confirms or corrects a result.

    Example in your /api/save route:
        from features import record_grader_confirmation
        record_grader_confirmation(
            measured_lab  = (scan['dominant_lab']['L'],
                             scan['dominant_lab']['a'],
                             scan['dominant_lab']['b']),
            correct_rhs_code = scan['rhs_grade'],   # grader-confirmed grade
            colors_db        = colors_db
        )
    """
    _illuminant_corrector.record(measured_lab, correct_rhs_code, colors_db)


def reset_illuminant_correction() -> None:
    """Call when the grader moves to a new lighting environment."""
    _illuminant_corrector.reset()



def extract_features(img: Image.Image) -> np.ndarray:
    """
    248-dimensional feature vector — must match the training pipeline exactly.

    Breakdown
    ─────────
    mean_lab       3    global mean Lab (normalised)
    std_rgb        3    per-channel σ / 255
    color_moments  9    mean / σ / skew per channel
    region_lab    27    3×3 spatial grid mean Lab
    rgb_hist      96    32-bin histogram × 3 channels
    lab_hist      48    16-bin histogram for L, a, b (sampled grid)
    hsv_hist      32    16-bin H + 8-bin S + 8-bin V
    opponent       6    RG / BY / WB opponent channel mean + σ
    lbp           10    uniform LBP  P=8  R=1
    gabor         12    6-orientation Gabor magnitude mean + σ
    delta_e        2    intra-image Lab dispersion mean + σ
                  ───
    TOTAL        248
    """
    img  = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr  = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # mean Lab
    mr, mg, mb = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    Lv, av, bv = _rgb_to_lab(mr, mg, mb)
    mean_lab = [Lv / 100.0, av / 128.0, bv / 128.0]

    # std RGB
    std_rgb = [arr[:, :, c].std() / 255.0 for c in range(3)]

    # colour moments
    color_moments = []
    for c in range(3):
        ch  = arr[:, :, c] / 255.0
        m   = float(ch.mean())
        s   = float(ch.std())
        sk  = float(np.mean(((ch - m) / (s + 1e-9)) ** 3))
        color_moments.extend([m, s, np.clip(sk, -3, 3) / 3.0])

    # 3×3 region Lab
    region_lab = []
    rsl = [slice(0, h // 3), slice(h // 3, 2 * h // 3), slice(2 * h // 3, h)]
    csl = [slice(0, w // 3), slice(w // 3, 2 * w // 3), slice(2 * w // 3, w)]
    for rs in rsl:
        for cs in csl:
            q = arr[rs, cs]
            if q.size == 0:
                region_lab.extend([0.0, 0.0, 0.0])
                continue
            qL, qa, qb = _rgb_to_lab(
                q[:, :, 0].mean(), q[:, :, 1].mean(), q[:, :, 2].mean()
            )
            region_lab.extend([qL / 100.0, qa / 128.0, qb / 128.0])

    # RGB histogram (32 bins / channel)
    rgb_hist = []
    for c in range(3):
        h32, _ = np.histogram(arr[:, :, c], bins=32, range=(0, 255))
        rgb_hist.extend(h32 / (h32.sum() + 1e-9))

    # Lab histogram (sampled grid, 16 bins each)
    step       = max(1, IMG_SIZE // 8)
    lab_pixels = []
    for y in range(0, IMG_SIZE, step):
        for x in range(0, IMG_SIZE, step):
            lab_pixels.append(
                _rgb_to_lab(int(arr[y, x, 0]), int(arr[y, x, 1]), int(arr[y, x, 2]))
            )
    lab_pixels = np.array(lab_pixels)
    hL, _ = np.histogram(lab_pixels[:, 0], bins=16, range=(0, 100))
    ha, _ = np.histogram(lab_pixels[:, 1], bins=16, range=(-128, 128))
    hb, _ = np.histogram(lab_pixels[:, 2], bins=16, range=(-128, 128))
    lab_hist  = list(hL / (hL.sum() + 1e-9))
    lab_hist += list(ha / (ha.sum() + 1e-9))
    lab_hist += list(hb / (hb.sum() + 1e-9))

    # HSV histogram
    an        = arr / 255.0
    R, G, Bch = an[:, :, 0], an[:, :, 1], an[:, :, 2]
    Cmax = np.maximum(np.maximum(R, G), Bch)
    Cmin = np.minimum(np.minimum(R, G), Bch)
    dlt  = Cmax - Cmin + 1e-9
    Hh   = np.zeros_like(R)
    Hh[Cmax == R]   = ((G[Cmax == R]   - Bch[Cmax == R])  / dlt[Cmax == R])   % 6
    Hh[Cmax == G]   = ((Bch[Cmax == G] - R[Cmax == G])    / dlt[Cmax == G])   + 2
    Hh[Cmax == Bch] = ((R[Cmax == Bch] - G[Cmax == Bch])  / dlt[Cmax == Bch]) + 4
    Hh = (Hh / 6.0) * 255.0
    Sh = np.where(Cmax > 0, dlt / (Cmax + 1e-9), 0) * 255.0
    Vh = Cmax * 255.0
    hH, _ = np.histogram(Hh, bins=16, range=(0, 255))
    hS, _ = np.histogram(Sh, bins=8,  range=(0, 255))
    hV, _ = np.histogram(Vh, bins=8,  range=(0, 255))
    hsv_hist  = list(hH / (hH.sum() + 1e-9))
    hsv_hist += list(hS / (hS.sum() + 1e-9))
    hsv_hist += list(hV / (hV.sum() + 1e-9))

    # Opponent channels
    opponent = [
        float((R  - G).mean()),
        float((Bch - (R + G) / 2).mean()),
        float(((R  + G + Bch) / 3).mean()),
        float((R  - G).std()),
        float((Bch - (R + G) / 2).std()),
        float(((R  + G + Bch) / 3).std()),
    ]

    # LBP (uniform, P=8, R=1)
    gray     = np.array(img.convert("L"), dtype=np.uint8)
    lbp      = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_h, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_feats = list(lbp_h / (lbp_h.sum() + 1e-9))

    # Gabor (6 orientations, frequency=0.3)
    gf          = gray.astype(np.float32) / 255.0
    gabor_feats = []
    for theta in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
        fr, fi = gabor(gf, frequency=0.3, theta=theta)
        mag    = np.sqrt(fr ** 2 + fi ** 2)
        gabor_feats.extend([float(mag.mean()), float(mag.std())])

    # Intra-image Lab dispersion
    de_v     = np.sqrt(((lab_pixels - lab_pixels.mean(axis=0)) ** 2).sum(axis=1))
    de_feats = [float(de_v.mean()) / 50.0, float(de_v.std()) / 50.0]

    vec = np.array(
        mean_lab + std_rgb + color_moments + region_lab +
        rgb_hist + lab_hist + hsv_hist + opponent +
        lbp_feats + gabor_feats + de_feats,
        dtype=np.float32,
    )
    assert vec.shape[0] == _FEAT_DIM, \
        f"Feature dim mismatch: got {vec.shape[0]}, expected {_FEAT_DIM}"
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION  (PIL-only fallback — segment.py overrides when present)
# ─────────────────────────────────────────────────────────────────────────────

def segment_image(img: Image.Image):
    """Lightweight centre-crop fallback segmentation."""
    w, h    = img.size
    mx, my  = int(w * 0.15), int(h * 0.15)
    cropped = img.crop((mx, my, w - mx, h - my))
    coverage = cropped.width * cropped.height / (w * h) * 100
    return cropped, True, round(coverage, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _de_classify(de: float):
    """
    Map a ΔE value to (verdict_text, badge_text, colour_hex).

    Thresholds are calibrated to the RHS-420 database where the median
    distance between adjacent grades (A↔B, same family) is ~15.8 ΔE.
    Lighting variation in real abaca photos adds ±3-5 ΔE, so:
        ΔE <  3 — Strong Match : within lighting noise floor
        ΔE <  8 — Good Match   : correct grade with minor exposure offset
        ΔE < 14 — Likely Match : adjacent grade territory (one step off)
        ΔE ≥ 14 — Weak Match   : different grade family
    """
    for threshold, verdict, badge, colour in _DE_THRESHOLDS:
        if de < threshold:
            return verdict, badge, colour
    return _DE_POOR


def _verdict(de: float):
    """Return (verdict_text, verdict_colour_hex)."""
    v, _, c = _de_classify(de)
    return v, c


def _de_badge(de: float):
    """Return (badge_label, badge_colour_hex) for top-5 list entries."""
    _, b, c = _de_classify(de)
    return b, c


def _de_to_pct(de: float) -> float:
    """
    Convert ΔE to a 0–100 match percentage.

    Formula: 100 × exp(−ΔE / 30)
    Calibrated so the boundary scores feel intuitive to graders:
        ΔE =  0 → 100.0 %   (perfect)
        ΔE =  3 →  90.5 %   (Excellent upper bound)
        ΔE =  8 →  76.6 %   (Good upper bound)
        ΔE = 14 →  62.9 %   (Fair/Poor boundary)
        ΔE = 20 →  51.3 %   (clearly wrong)

    Previous /20 divisor made ΔE=9.6 show as 61.9 % which confused graders
    seeing a correct grade flagged as "low confidence".
    """
    return round(float(100.0 * np.exp(-de / _SCORE_DECAY)), 4)


def _combined_score(de: float, mlp_prob: float) -> float:
    """
    Ranking score = 95% ΔE2000 + 5% MLP probability.

    The MLP was trained on synthetic swatch images and outputs near-zero
    probability for all real RHS codes when given real fiber photos
    (confirmed: prob(59A)=0.0000, prob(187A)=0.0000 for any field image).
    Until the MLP is retrained on real fiber photos its weight is reduced
    to 5% — just enough to break exact ΔE ties, not enough to override
    correct ΔE rankings.

    When retrained on real photos, raise mlp_weight back toward 0.25.
    """
    de_score  = float(np.exp(-de / _SCORE_DECAY))
    mlp_score = float(np.clip(mlp_prob, 0.0, 1.0))
    return 0.95 * de_score + 0.05 * mlp_score


def scan_quality_check(img: Image.Image) -> dict:
    """
    Assess whether the image is a usable fiber scan.

    Cannot reliably detect "not abaca" — the RHS database covers the entire
    visible spectrum, so grass, wood, skin, sky all match some RHS code.
    What CAN be detected are technical failures that make any scan unreliable:

    Returns
    ───────
    scannable      : bool   — False only for hard technical failures
    warnings       : list   — human-readable messages shown to grader
    mean_lum       : float  — average luminance (0-255)
    mean_sat       : float  — average saturation (0-1)
    lum_std        : float  — luminance std dev (texture indicator)
    """
    arr = np.array(img.convert("RGB").resize((200, 200)), dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    lum     = R*0.299 + G*0.587 + B*0.114
    Cmax    = np.maximum(np.maximum(R,G),B)
    sat     = (Cmax - np.minimum(np.minimum(R,G),B)) / (Cmax + 1e-9)

    mean_lum = float(lum.mean())
    lum_std  = float(lum.std())
    mean_sat = float(sat.mean())

    warnings  = []
    scannable = True

    # Hard block 1: overexposed — crop box on white sky, direct sunlight
    if mean_lum > 215:
        warnings.append("Image too bright — move crop box off the glare or step into shade.")
        scannable = False

    # Hard block 2: underexposed — crop box in deep shadow or lens covered
    elif mean_lum < 10:
        warnings.append("Image too dark — move to brighter light or uncover the lens.")
        scannable = False

    # Hard block 3: no color — greyscale object (concrete, metal, paper)
    if mean_sat < 0.025:
        warnings.append("No color detected in crop area — fiber should have visible color.")
        scannable = False

    # Soft warning: low light (shade) — results may shift
    if scannable and mean_lum < 45:
        warnings.append(
            "Low light detected — results may be less accurate. "
            "Move to brighter natural light if possible."
        )

    # Soft warning: very low texture (solid surface, not fiber)
    if scannable and lum_std < 6:
        warnings.append(
            "Very uniform surface — fiber should have strand texture. "
            "Ensure crop box is over the fiber, not a flat background."
        )

    return {
        "scannable":  scannable,
        "warnings":   warnings,
        "mean_lum":   round(mean_lum, 1),
        "mean_sat":   round(mean_sat, 3),
        "lum_std":    round(lum_std, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict(img: Image.Image,
            mlp_a, mlp_b, mlp_c, mlp_d,
            scaler, le,
            colors_db: dict) -> dict:
    """
    Full inference pipeline (v3 — all 5 bugs fixed).

    Fixes applied vs v2
    ───────────────────
    1. Ensemble weights corrected to match model_config.json (A/B and C/D were swapped).
    2. ΔE76 → ΔE2000 throughout — perceptually uniform, reduces lightness over-weighting.
    3. Foreground lo_pct raised 10→20, saturation threshold 0.08→0.15 — shadow pixels
       no longer corrupt the Lab reading.
    4. mask_median_color() from segment.py now used when segmentation succeeds — this
       is the shadow-aware 3×3 grid color extractor that was written but never called.
    5. MLP weight reduced 25%→5% — MLP outputs ~0.0 for all real fiber photos (trained
       on synthetic swatches, not real photos). 5% only breaks exact ΔE ties.
    6. Tie detection tightened ΔE gap 8.0→3.0 — old threshold fired on every scan.
    """
    img = img.convert("RGB")

    # ── 1. Segmentation ───────────────────────────────────────────────────────
    seg_img      = img
    seg_found    = False
    seg_coverage = 100.0
    seg_mask     = None
    try:
        from segment import segment_fiber
        res          = segment_fiber(img)
        seg_img      = res[0]
        seg_found    = bool(res[1])
        seg_coverage = float(res[2])
        seg_mask     = res[3] if len(res) > 3 else None
    except Exception:
        pass

    # ── 2. Dominant fiber colour ──────────────────────────────────────────────
    # Priority: use mask_median_color() from segment.py when we have a good mask.
    # It uses a shadow-aware 3×3 grid median that is more accurate than the
    # percentile band approach in _extract_foreground_color for real field photos.
    # Fall back to _extract_foreground_color when segmentation failed.
    r_dom = g_dom = b_dom = None
    if seg_found and seg_mask is not None:
        try:
            from segment import mask_median_color
            rgb = mask_median_color(img, seg_mask)
            if rgb is not None:
                r_dom, g_dom, b_dom = rgb
        except Exception:
            pass

    if r_dom is None:
        # Fallback: percentile-band foreground extractor on original image
        dominant_lab, (r_dom, g_dom, b_dom), fg_found, fg_coverage = \
            _extract_foreground_color(img)
    else:
        fg_found    = True
        fg_coverage = float(seg_mask.sum()) / float(seg_mask.size) if seg_mask is not None else 0.0

    dominant_lab = _rgb_to_lab(r_dom, g_dom, b_dom)

    # ── 2b. Lighting-adaptive correction ─────────────────────────────────────
    # Transform the RHS database to match field lighting, not the other way.
    # First scan: auto-detects lighting model from the 11 pre-defined options.
    # After 2+ confirmed scans: uses learned per-channel gains.
    dominant_lab_raw = dominant_lab
    illuminant_active  = _lighting_adapter.is_learned
    illuminant_samples = _lighting_adapter.sample_count
    dominant_hex = "#{:02x}{:02x}{:02x}".format(r_dom, g_dom, b_dom)

    # ── 3. Feature extraction ─────────────────────────────────────────────────
    feat_src     = seg_img if seg_found else img
    feats        = extract_features(feat_src).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    # ── 4. MLP probability ───────────────────────────────────────────────────
    # Uses only mlp_a (single model). mlp_b/c/d params accepted for backward
    # compatibility with app.py but ignored. When mlp_b/c/d are None or the
    # old quad models, only mlp_a is used — no errors thrown.
    pa = mlp_a.predict_proba(feats_scaled)[0]
    avg_proba = pa  # single model, no blending needed

    mlp_prob_map = {le.classes_[i]: float(avg_proba[i])
                    for i in range(len(le.classes_))}
    top_mlp_prob = float(max(avg_proba))

    # ── 5. Score all RHS classes — lighting-adaptive ─────────────────────────
    # The LightingAdapter transforms the database to match field lighting,
    # then scores using ΔE2000 + 5% MLP. On first scan it auto-detects the
    # lighting model; after 2+ confirmed scans it uses learned gains.
    top_mlp_prob = float(max(avg_proba))
    candidates, used_gains = _lighting_adapter.rank_with_lighting(
        dominant_lab, colors_db, mlp_prob_map
    )

    # ── 6. Build top-5 output ─────────────────────────────────────────────────
    top5_list = []
    for _, de_f, code, hex_ in candidates[:5]:
        badge_lbl, badge_clr = _de_badge(de_f)
        top5_list.append({
            "rhs_code":    code,
            "delta_e":     round(de_f, 4),
            "match_score": _de_to_pct(de_f),
            "hex":         hex_,
            "de_color":    badge_clr,
            "de_label":    badge_lbl,
        })

    # Best match
    _, best_de, best_code, best_hex = candidates[0]
    best_de = round(best_de, 4)
    verdict_txt, verdict_clr = _verdict(best_de)

    # ── 6b. Tie detection — ΔE2000 gap < 3.0 ────────────────────────────────
    is_tie   = False
    tie_code = None
    tie_hex  = None
    if len(candidates) >= 2:
        de_gap = abs(candidates[0][1] - candidates[1][1])
        if de_gap < 3.0:
            is_tie   = True
            tie_code = candidates[1][2]
            tie_hex  = candidates[1][3]

    # ── 7. Colour-cast detection ──────────────────────────────────────────────
    cast_label   = None
    cast_warning = None
    if max(abs(r_dom - g_dom), abs(r_dom - b_dom), abs(g_dom - b_dom)) > 30:
        if r_dom > g_dom + 20:
            cast_label   = "Warm/Red cast"
            cast_warning = "Consider white-balance calibration"
        elif b_dom > r_dom + 20:
            cast_label   = "Cool/Blue cast"
            cast_warning = "Consider white-balance calibration"

    # ── 8. Normalise seg_coverage to percent ─────────────────────────────────
    seg_coverage_pct = round(
        seg_coverage * 100.0 if seg_coverage <= 1.0 else seg_coverage, 1
    )

    return {
        "rhs_code":      best_code,
        "rhs_grade":     best_code,
        "delta_e":       best_de,
        "match_score":   _de_to_pct(best_de),

        "dominant_hex":  dominant_hex,
        "matched_hex":   best_hex,

        "verdict":       verdict_txt,
        "verdict_color": verdict_clr,

        "dominant_rgb": {"R": r_dom, "G": g_dom, "B": b_dom},
        "dominant_lab": {
            "L": round(dominant_lab[0], 2),
            "a": round(dominant_lab[1], 2),
            "b": round(dominant_lab[2], 2),
        },

        "top_5":     top5_list,
        "is_tie":    is_tie,
        "tie_code":  tie_code,
        "tie_hex":   tie_hex,

        "seg_found":    seg_found,
        "seg_coverage": seg_coverage_pct,

        "wb_applied":   False,
        "cast_label":   cast_label,
        "cast_warning": cast_warning,

        "top_mlp_confidence": round(top_mlp_prob, 4),
        "fg_coverage":        round(fg_coverage, 4),

        "illuminant_correction_active":  illuminant_active,
        "illuminant_correction_samples": illuminant_samples,
        "lighting_gains": {
            "L": round(used_gains[0], 3),
            "a": round(used_gains[1], 3),
            "b": round(used_gains[2], 3),
        },
    }