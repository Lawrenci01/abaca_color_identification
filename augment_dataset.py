"""
augment_dataset.py — Parallel Dataset Generation (876 classes)

Fixes applied:
- Seed fix: each worker uses hash(class_label) for unique, reproducible seeds
             per class — prevents duplicate augmentation sequences across workers
- Fixed single-line if/return statements (were not returning correctly)
- Auto-cleans augmented/ folder before generating (prevents stale image mixing)
- Prints skipped classes at end
"""
import csv, random, io, os, shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── CONFIG ────────────────────────────────────────────────────────────────────
SWATCH_SIZE = 96
N_AUGMENTS  = 300   # was 1000 — fiber augs give better coverage per image
N_TEST      = 10

RHS_CSV      = Path("abaca_pipeline/rhs_colors.csv")
SWATCHES_DIR = Path("swatches_real")
OUT_DIR      = Path("abaca_pipeline/augmented")
MANIFEST_OUT = Path("abaca_pipeline/augmented_manifest.csv")

# ── AUGMENTATION FUNCTIONS ────────────────────────────────────────────────────

def aug_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 1.40))

def aug_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.30))

def aug_gamma(img):
    gamma = random.uniform(0.70, 1.45)
    arr   = np.array(img, dtype=np.float32) / 255.0
    return Image.fromarray((np.power(np.clip(arr, 0, 1), gamma) * 255).astype(np.uint8))

def aug_noise(img):
    arr = np.array(img, dtype=np.float32)
    return Image.fromarray(
        np.clip(arr + np.random.normal(0, random.uniform(2, 12), arr.shape), 0, 255).astype(np.uint8)
    )

def aug_rotate(img):
    # Limited to ±15° — full 360° creates black corners on real texture crops
    return img.rotate(random.uniform(-15, 15), expand=False)

def aug_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT if random.random() < 0.5 else Image.FLIP_TOP_BOTTOM)

def aug_crop(img):
    w, h = img.size
    m    = int(w * random.uniform(0.05, 0.18))
    l, t = random.randint(0, m), random.randint(0, m)
    r, b = w - random.randint(0, m), h - random.randint(0, m)
    return img if r <= l or b <= t else img.crop((l, t, r, b)).resize((w, h), Image.LANCZOS)

def aug_shadow(img):
    arr   = np.array(img, dtype=np.float32)
    s, d  = random.uniform(0.60, 0.92), random.choice(['left', 'right', 'top', 'bottom'])
    hh, w = arr.shape[:2]
    sp    = random.randint(w // 4, 3 * w // 4)
    if d == 'left':    arr[:, :sp] *= s
    elif d == 'right': arr[:, sp:] *= s
    elif d == 'top':   arr[:sp, :] *= s
    else:              arr[sp:, :] *= s
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def aug_saturation(img):
    return ImageEnhance.Color(img).enhance(random.uniform(0.80, 1.25))

def aug_sharpness(img):
    return ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 2.0))

def aug_blur(img):
    # Capped at 0.8 — radius 1.2 washes out texture on 96px images
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

def aug_jpeg(img):
    # Floor raised to 75 — quality 65 creates harsh blocks on 96px images
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=random.randint(75, 95))
    buf.seek(0)
    return Image.open(buf).copy()

def aug_exposure_gradient(img):
    arr   = np.array(img, dtype=np.float32)
    hh, w = arr.shape[:2]
    d, s  = random.choice(['horizontal', 'vertical', 'diagonal']), random.uniform(0.12, 0.30)
    if d == 'horizontal':
        arr *= np.linspace(1 - s, 1 + s, w)[np.newaxis, :, np.newaxis]
    elif d == 'vertical':
        arr *= np.linspace(1 - s, 1 + s, hh)[:, np.newaxis, np.newaxis]
    else:
        gx = np.linspace(1 - s / 2, 1 + s / 2, w)
        gy = np.linspace(1 - s / 2, 1 + s / 2, hh)
        arr *= gx[np.newaxis, :, np.newaxis] * gy[:, np.newaxis, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def aug_channel_scale(img):
    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * random.uniform(0.95, 1.05), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# ── ABACA FIBER-SPECIFIC AUGMENTATIONS ────────────────────────────────────────
# These simulate the real visual properties of photographed abaca fiber:
# parallel strands, inter-strand shadows, surface sheen, fiber waviness,
# and the circular hole vignette from the blue grading card.

def aug_fiber_strands(img):
    """
    Simulate abaca fiber strand texture — parallel lines of alternating
    brightness running in a dominant direction.

    Real abaca fiber has bundles of parallel strands. When photographed,
    these create a fine linear pattern: slightly brighter on strand tops,
    slightly darker in the grooves between strands.

    Strand width: 2-5px at 96px image size (represents 1-3mm bundles).
    Brightness variation: ±8-18% — subtle enough to look like texture,
    strong enough to affect feature extraction.
    Angle: mostly vertical (0-30°) since fibers are usually laid flat,
    but rotated ±30° to cover different orientations.
    """
    arr   = np.array(img, dtype=np.float32)
    h, w  = arr.shape[:2]
    angle = random.uniform(-30, 30)          # fiber orientation
    strand_w = random.randint(2, 5)          # pixel width of one strand
    depth    = random.uniform(0.08, 0.18)    # brightness variation depth

    # Build a stripe pattern then rotate it
    stripe = np.zeros((h, w), dtype=np.float32)
    for x in range(w):
        # sine wave along x axis gives smooth strand profile
        stripe[:, x] = np.sin(x * np.pi / strand_w) * depth

    # Rotate stripe map to match fiber angle
    import cv2 as _cv2
    M = _cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    stripe = _cv2.warpAffine(stripe, M, (w, h))

    # Apply multiplicatively to all channels (preserves hue)
    arr *= (1.0 + stripe[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def aug_fiber_shadow_lines(img):
    """
    Simulate the micro-shadows cast between fiber strands.

    Between each strand bundle, a narrow dark groove forms where strands
    don't lie perfectly flat. This creates thin dark lines (1-2px) spaced
    every 3-8px across the image. Unlike aug_fiber_strands which is smooth,
    these are sharp-edged dark lines — harder shadows.
    """
    arr    = np.array(img, dtype=np.float32)
    h, w   = arr.shape[:2]
    spacing = random.randint(3, 8)     # pixels between shadow lines
    depth   = random.uniform(0.12, 0.28)  # shadow darkness
    angle   = random.uniform(-25, 25)

    shadow = np.zeros((h, w), dtype=np.float32)
    for x in range(0, w, spacing):
        shadow[:, min(x, w-1)] = -depth
        if x + 1 < w:
            shadow[:, x + 1] = -depth * 0.4  # softer edge

    import cv2 as _cv2
    M = _cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    shadow = _cv2.warpAffine(shadow, M, (w, h))

    arr *= (1.0 + shadow[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def aug_surface_sheen(img):
    """
    Simulate the semi-gloss specular sheen on dried abaca fiber.

    Dried abaca has a slight waxy/silky surface that catches light
    along strand edges. This appears as small bright elliptical or
    linear highlights — not full glare, just a localized brightness boost
    in a directional band across the image.

    This is different from aug_exposure_gradient: gradient is smooth
    and uniform; sheen is a narrow bright band with soft falloff,
    angled at ~30-60° to match typical fiber/light geometry.
    """
    arr  = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Sheen band: center position, width, angle, intensity
    center_x = random.randint(w // 4, 3 * w // 4)
    center_y = random.randint(h // 4, 3 * h // 4)
    band_w   = random.randint(8, 24)   # narrow band
    strength = random.uniform(0.10, 0.25)
    angle    = random.uniform(20, 70)  # degrees from horizontal

    # Build distance map from the band axis
    Y, X = np.ogrid[:h, :w]
    rad  = np.radians(angle)
    # Signed distance from band center line
    dist = np.abs((X - center_x) * np.cos(rad) + (Y - center_y) * np.sin(rad))
    # Gaussian falloff from band center
    sheen_map = strength * np.exp(-(dist ** 2) / (2 * (band_w ** 2)))

    arr *= (1.0 + sheen_map[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def aug_fiber_waviness(img):
    """
    Simulate slight fiber bundle waviness — fibers are not perfectly
    parallel, they have gentle curves and crossings.

    Implemented as a subtle elastic deformation: each row is shifted
    left/right by a small sinusoidal amount, creating a wave-like
    distortion that mimics fiber curvature without breaking the
    overall color signal.

    Wave amplitude: 2-6px (very subtle at 96px)
    Wave frequency: 0.5-2 cycles across the image
    """
    import cv2 as _cv2
    arr  = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    amplitude = random.uniform(2.0, 6.0)
    frequency = random.uniform(0.5, 2.0)
    phase     = random.uniform(0, 2 * np.pi)

    # Build displacement map
    rows = np.arange(h)
    shift = amplitude * np.sin(2 * np.pi * frequency * rows / h + phase)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for r in range(h):
        map_x[r, :] = np.arange(w) + shift[r]
        map_y[r, :] = r

    warped = _cv2.remap(arr, map_x, map_y,
                        interpolation=_cv2.INTER_LINEAR,
                        borderMode=_cv2.BORDER_REFLECT)
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))


def aug_circular_vignette(img):
    """
    Simulate the circular hole vignette from the blue grading card.

    When a grader photographs abaca through the circular hole in the
    blue card, the edge of the hole creates a soft falloff — pixels
    near the edge receive slightly less direct light and may show a
    subtle darkening toward the circle boundary.

    This also adds a very faint cool (blue) color cast at the extreme
    edges, from light scatter off the blue card surface.

    Radius: 85-95% of half the image width (circle almost fills frame).
    Edge softness: 5-12px gradual falloff.
    """
    arr  = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    cx, cy = w / 2, h / 2
    radius = min(w, h) / 2 * random.uniform(0.82, 0.96)
    edge_w = random.uniform(5.0, 14.0)
    depth  = random.uniform(0.08, 0.20)

    Y, X  = np.ogrid[:h, :w]
    dist  = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Smooth vignette: 1.0 at center, falls off near edge
    vignette = 1.0 - depth * np.clip((dist - (radius - edge_w)) / edge_w, 0, 1)

    # Very subtle blue cast at edges (light scatter from blue card)
    blue_cast = np.clip((dist - (radius - edge_w * 2)) / (edge_w * 2), 0, 1) * 0.06
    arr[:, :, 0] = np.clip(arr[:, :, 0] * vignette, 0, 255)           # R
    arr[:, :, 1] = np.clip(arr[:, :, 1] * vignette, 0, 255)           # G
    arr[:, :, 2] = np.clip(arr[:, :, 2] * vignette + blue_cast * 40, 0, 255)  # B +cast

    return Image.fromarray(arr.astype(np.uint8))


def aug_uneven_fiber_density(img):
    """
    Simulate uneven fiber density — abaca fibers don't pack perfectly.
    Some areas are denser (darker, more shadowed), others are sparser
    (lighter, more surface visible).

    Implemented as a low-frequency brightness map using Perlin-like
    noise (sum of 2-3 sinusoids at different frequencies and phases),
    creating blob-like regions of slight over/under exposure across
    the image — mimicking real fiber bundle density variation.
    """
    arr  = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    density = np.zeros((h, w), dtype=np.float32)

    # Sum 2-3 low-frequency sinusoidal components
    n_waves = random.randint(2, 3)
    for _ in range(n_waves):
        fx  = random.uniform(0.5, 2.5) / w   # spatial frequency
        fy  = random.uniform(0.5, 2.5) / h
        phi = random.uniform(0, 2 * np.pi)
        amp = random.uniform(0.04, 0.10)
        density += amp * np.sin(2 * np.pi * (fx * X + fy * Y) + phi)

    arr *= (1.0 + density[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── AUGMENTATION POOL ─────────────────────────────────────────────────────────
# Two pools:
# COLOR_AUGS    — change color/brightness/exposure (preserve texture structure)
# STRUCTURE_AUGS — change texture/spatial structure (preserve color signal)
# FIBER_AUGS    — new: simulate real abaca-specific visual properties
COLOR_AUGS        = [aug_brightness, aug_contrast, aug_gamma, aug_shadow,
                     aug_saturation, aug_exposure_gradient, aug_channel_scale]
COLOR_WEIGHTS     = [0.18, 0.15, 0.13, 0.12, 0.12, 0.16, 0.14]

STRUCTURE_AUGS    = [aug_noise, aug_rotate, aug_flip, aug_crop,
                     aug_sharpness, aug_blur, aug_jpeg]
STRUCTURE_WEIGHTS = [0.15, 0.14, 0.12, 0.14, 0.15, 0.15, 0.15]

# Abaca fiber-specific — applied with 60% probability per image
FIBER_AUGS        = [aug_fiber_strands, aug_fiber_shadow_lines, aug_surface_sheen,
                     aug_fiber_waviness, aug_circular_vignette, aug_uneven_fiber_density]
FIBER_WEIGHTS     = [0.22, 0.18, 0.16, 0.14, 0.16, 0.14]


def apply_augmentations(img):
    """
    Apply augmentations per image in three passes:

    Pass 1 — Color (max 2 from COLOR group)
      Simulates lighting, exposure, white-balance variation.
      Capped at 2 to prevent compounding color drift away from
      the reference RHS Lab value.

    Pass 2 — Structure (1 from STRUCTURE group)
      Simulates camera blur, JPEG compression, rotation.
      Always at least 1 to ensure structural variety.

    Pass 3 — Fiber texture (0-2 from FIBER group, 60% probability)
      Simulates abaca-specific visual properties:
      strand lines, inter-strand shadows, surface sheen,
      fiber waviness, circular vignette, density variation.
      Applied AFTER color so the color signal is not distorted
      by the texture simulation.

    This 3-pass approach means ~60% of training images will have
    realistic abaca texture simulation, closing the domain gap
    between the flat RHS card swatches and real fiber photos.
    """
    cp = np.array(COLOR_WEIGHTS)   / sum(COLOR_WEIGHTS)
    sp = np.array(STRUCTURE_WEIGHTS) / sum(STRUCTURE_WEIGHTS)
    fp = np.array(FIBER_WEIGHTS)   / sum(FIBER_WEIGHTS)

    # Pass 1: Color
    color_n, struct_n = (2, 1) if random.random() < 0.6 else (1, 2)
    chosen_color  = np.random.choice(len(COLOR_AUGS),     size=color_n,  replace=False, p=cp)
    chosen_struct = np.random.choice(len(STRUCTURE_AUGS), size=struct_n, replace=False, p=sp)

    ops = [COLOR_AUGS[i] for i in chosen_color] + [STRUCTURE_AUGS[i] for i in chosen_struct]
    random.shuffle(ops)
    for fn in ops:
        img = fn(img)

    # Pass 2: Fiber texture (applied 60% of the time)
    # 30% chance: 1 fiber aug — subtle texture hint
    # 30% chance: 2 fiber augs — fuller texture simulation
    # 40% chance: no fiber aug — keeps some "clean" examples for robustness
    r = random.random()
    if r < 0.30:
        fiber_n = 1
    elif r < 0.60:
        fiber_n = 2
    else:
        fiber_n = 0

    if fiber_n > 0:
        chosen_fiber = np.random.choice(len(FIBER_AUGS), size=fiber_n, replace=False, p=fp)
        for i in chosen_fiber:
            img = FIBER_AUGS[i](img)

    return img


# ── WORKER FUNCTION ───────────────────────────────────────────────────────────

def process_color_group(color_data):
    # Fix: unique reproducible seed per class using class_label hash
    # Prevents workers from generating identical augmentation sequences
    seed = abs(hash(color_data['class_label'])) % (2**32)
    random.seed(seed)
    np.random.seed(seed)

    class_label = color_data['class_label']
    rhs_code    = color_data['rhs_code']
    r, g, b     = int(color_data['R']), int(color_data['G']), int(color_data['B'])

    swatch_path = SWATCHES_DIR / f"{class_label}.png"
    if not swatch_path.exists():
        return None, class_label

    color_dir = OUT_DIR / class_label
    color_dir.mkdir(exist_ok=True, parents=True)

    base_img  = Image.open(swatch_path).convert('RGB').resize((SWATCH_SIZE, SWATCH_SIZE), Image.LANCZOS)
    base_path = color_dir / "base.png"
    base_img.save(base_path)

    results = [{
        'path': str(base_path), 'rhs_code': rhs_code, 'class_label': class_label,
        'R': r, 'G': g, 'B': b,
        'Lab_L': color_data['Lab_L'], 'Lab_a': color_data['Lab_a'], 'Lab_b': color_data['Lab_b'],
        'split': 'train', 'source': 'real',
    }]

    for i in range(N_AUGMENTS):
        aug_img  = apply_augmentations(base_img.copy())
        aug_path = color_dir / f"aug_{i:04d}.png"
        aug_img.save(aug_path)
        results.append({
            'path': str(aug_path), 'rhs_code': rhs_code, 'class_label': class_label,
            'R': r, 'G': g, 'B': b,
            'Lab_L': color_data['Lab_L'], 'Lab_a': color_data['Lab_a'], 'Lab_b': color_data['Lab_b'],
            'split': 'test' if i < N_TEST else 'train',
            'source': 'augmented',
        })

    return results, None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not RHS_CSV.exists():
        print(f"ERROR: {RHS_CSV} not found")
        return
    if not SWATCHES_DIR.exists():
        print(f"ERROR: {SWATCHES_DIR}/ not found — run process_real_photos.py first")
        return

    # Auto-clean augmented/ to prevent stale image mixing on reruns
    if OUT_DIR.exists():
        print(f"  Cleaning old augmented/ folder ...")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RHS_CSV, encoding='utf-8') as f:
        colors = list(csv.DictReader(f))

    print(f"{'='*65}")
    print(f"  DATASET GENERATION — Parallel Mode ({len(colors)} classes)")
    print(f"{'='*65}")
    print(f"  Images per class : {N_AUGMENTS + 1:,}  ({N_TEST} test / {N_AUGMENTS - N_TEST} train)")
    print(f"  Total images     : {len(colors) * (1 + N_AUGMENTS):,}")
    print(f"  CPUs detected    : {os.cpu_count()}")
    print(f"  Swatch source    : {SWATCHES_DIR}/")
    print(f"  Output           : {OUT_DIR.resolve()}\n")

    manifest        = []
    missing         = []
    processed_count = 0

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_color_group, color): color for color in colors}
        for future in as_completed(futures):
            results, missed = future.result()
            if missed:
                missing.append(missed)
            if results:
                manifest.extend(results)
                processed_count += 1
                if processed_count % 20 == 0:
                    print(f"  Progress: {processed_count}/{len(colors)} classes "
                          f"(~{len(manifest):,} images)")

    # Write manifest
    fieldnames = ['path', 'rhs_code', 'class_label', 'R', 'G', 'B',
                  'Lab_L', 'Lab_a', 'Lab_b', 'split', 'source']
    with open(MANIFEST_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest)

    train = sum(1 for m in manifest if m['split'] == 'train')
    test  = sum(1 for m in manifest if m['split'] == 'test')

    print(f"\n{'='*65}\n  RESULTS\n{'='*65}")
    print(f"  Total   : {len(manifest):,}")
    print(f"  Train   : {train:,}")
    print(f"  Test    : {test:,}")
    print(f"  Classes : {processed_count} / {len(colors)}")
    if missing:
        print(f"  Skipped : {len(missing)} missing swatches:")
        for lbl in missing:
            print(f"    - {lbl}")
    print(f"  Manifest: {MANIFEST_OUT}")
    print(f"\n  Next step: python train_model.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()