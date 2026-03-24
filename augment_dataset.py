"""
augment_dataset.py — Parallel Dataset Generation (876 classes)

Fiber augmentation analysis (measured Lab drift vs adjacent grade separation=10.12):
    KEPT   aug_fiber_strands        mean ΔE=1.05  safe — brightness modulation only
    KEPT   aug_fiber_waviness       mean ΔE=0.45  safe — spatial warp, no color change
    KEPT   aug_uneven_fiber_density mean ΔE=1.36  safe — low-frequency density variation
    REMOVED aug_fiber_shadow_lines  mean ΔE=4.24  too much color drift
    REMOVED aug_surface_sheen       mean ΔE=6.14  pushes into adjacent class territory
    REMOVED aug_circular_vignette   mean ΔE=9.20  blue cast corrupts Lab values badly

N_AUGMENTS=1000 — matches original dataset size for 90%+ val accuracy.
"""
import csv, random, io, os, shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── CONFIG ────────────────────────────────────────────────────────────────────
SWATCH_SIZE = 96
N_AUGMENTS  = 1000
N_TEST      = 10

RHS_CSV      = Path("abaca_pipeline/rhs_colors.csv")
SWATCHES_DIR = Path("swatches_real")
OUT_DIR      = Path("abaca_pipeline/augmented")
MANIFEST_OUT = Path("abaca_pipeline/augmented_manifest.csv")


# ── STANDARD AUGMENTATIONS ────────────────────────────────────────────────────

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
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

def aug_jpeg(img):
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


# ── SAFE FIBER AUGMENTATIONS (mean Lab drift < 2.0 ΔE) ───────────────────────

def aug_fiber_strands(img):
    """
    Fiber strand texture via brightness modulation only.
    Mean Lab drift=1.05 — safe, median not affected by alternating stripes.
    """
    arr      = np.array(img, dtype=np.float32)
    h, w     = arr.shape[:2]
    angle    = random.uniform(-30, 30)
    strand_w = random.randint(2, 5)
    depth    = random.uniform(0.06, 0.12)

    stripe = np.zeros((h, w), dtype=np.float32)
    for x in range(w):
        stripe[:, x] = np.sin(x * np.pi / strand_w) * depth

    import cv2 as _cv2
    M      = _cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    stripe = _cv2.warpAffine(stripe, M, (w, h))
    arr   *= (1.0 + stripe[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def aug_fiber_waviness(img):
    """
    Fiber bundle waviness via spatial warp only.
    Mean Lab drift=0.45 — moves pixels, does not change their values.
    """
    import cv2 as _cv2
    arr       = np.array(img, dtype=np.float32)
    h, w      = arr.shape[:2]
    amplitude = random.uniform(1.0, 4.0)
    frequency = random.uniform(0.5, 2.0)
    phase     = random.uniform(0, 2 * np.pi)

    rows  = np.arange(h)
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


def aug_uneven_fiber_density(img):
    """
    Uneven fiber density via low-frequency brightness blobs.
    Mean Lab drift=1.36 — low-frequency variation averages out at median.
    """
    arr  = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    density = np.zeros((h, w), dtype=np.float32)

    n_waves = random.randint(2, 3)
    for _ in range(n_waves):
        fx  = random.uniform(0.5, 2.5) / w
        fy  = random.uniform(0.5, 2.5) / h
        phi = random.uniform(0, 2 * np.pi)
        amp = random.uniform(0.03, 0.07)
        density += amp * np.sin(2 * np.pi * (fx * X + fy * Y) + phi)

    arr *= (1.0 + density[:, :, np.newaxis])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── AUGMENTATION POOLS ────────────────────────────────────────────────────────
COLOR_AUGS     = [aug_brightness, aug_contrast, aug_gamma, aug_shadow,
                  aug_saturation, aug_exposure_gradient, aug_channel_scale]
COLOR_WEIGHTS  = [0.18, 0.15, 0.13, 0.12, 0.12, 0.16, 0.14]

STRUCTURE_AUGS    = [aug_noise, aug_rotate, aug_flip, aug_crop,
                     aug_sharpness, aug_blur, aug_jpeg]
STRUCTURE_WEIGHTS = [0.15, 0.14, 0.12, 0.14, 0.15, 0.15, 0.15]

# Only safe fiber augs — mean Lab drift < 2.0 ΔE each
FIBER_AUGS    = [aug_fiber_strands, aug_fiber_waviness, aug_uneven_fiber_density]
FIBER_WEIGHTS = [0.40, 0.30, 0.30]


def apply_augmentations(img):
    """
    Two-pass augmentation:

    Pass 1 — Color + Structure (always applied)
      Standard lighting/camera variation.
      Max 2 color augs to prevent compounding color drift.

    Pass 2 — Safe fiber texture (40% probability, 1 aug only)
      Only augs with mean Lab drift < 2.0 ΔE.
      Adds realistic fiber texture without corrupting the color signal.
      60% of images stay as clean color references.
    """
    cp = np.array(COLOR_WEIGHTS)     / sum(COLOR_WEIGHTS)
    sp = np.array(STRUCTURE_WEIGHTS) / sum(STRUCTURE_WEIGHTS)
    fp = np.array(FIBER_WEIGHTS)     / sum(FIBER_WEIGHTS)

    color_n, struct_n = (2, 1) if random.random() < 0.6 else (1, 2)
    chosen_color  = np.random.choice(len(COLOR_AUGS),     size=color_n,  replace=False, p=cp)
    chosen_struct = np.random.choice(len(STRUCTURE_AUGS), size=struct_n, replace=False, p=sp)

    ops = [COLOR_AUGS[i] for i in chosen_color] + [STRUCTURE_AUGS[i] for i in chosen_struct]
    random.shuffle(ops)
    for fn in ops:
        img = fn(img)

    # 40% chance of 1 safe fiber aug
    if random.random() < 0.40:
        idx = np.random.choice(len(FIBER_AUGS), p=fp)
        img = FIBER_AUGS[idx](img)

    return img


# ── WORKER FUNCTION ───────────────────────────────────────────────────────────

def process_color_group(color_data):
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