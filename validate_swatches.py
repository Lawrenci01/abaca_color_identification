"""
validate_swatches.py
====================
Validates real texture swatches in swatches_real/ against rhs_colors.csv.

Changes vs previous version:
- Points to swatches_real/ instead of swatches/ (real photo swatches)
- Removed synthetic swatch replacement — reports failures only,
  does NOT replace with flat synthetic patches (would ruin texture training)
- Added texture check — warns if swatch looks like a flat color block
  (std dev too low), which would indicate process_real_photos.py wasn't run
"""
import csv
import math
import numpy as np
from pathlib import Path
from PIL import Image

RHS_CSV      = Path("abaca_pipeline/rhs_colors.csv")
SWATCHES_DIR = Path("swatches_real")
SWATCH_SIZE  = 96
DE_THRESHOLD        = 25.0
BRIGHTNESS_THRESHOLD = 240
FLAT_STD_THRESHOLD  = 3.0   # std dev below this = likely flat color block


def rgb_to_lab(r, g, b):
    def linearize(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    rl, gl, bl = linearize(float(r)), linearize(float(g)), linearize(float(b))
    X = rl*0.4124564 + gl*0.3575761 + bl*0.1804375
    Y = rl*0.2126729 + gl*0.7151522 + bl*0.0721750
    Z = rl*0.0193339 + gl*0.1191920 + bl*0.9503041
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    def f(t): return t**(1/3) if t > 0.008856 else 7.787*t + 16/116
    fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
    return 116*fy - 16, 500*(fx - fy), 200*(fy - fz)


def delta_e(r1, g1, b1, r2, g2, b2):
    L1, a1, b1_ = rgb_to_lab(r1, g1, b1)
    L2, a2, b2_ = rgb_to_lab(r2, g2, b2)
    return math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1_-b2_)**2)


def check_swatch(path, expected_r, expected_g, expected_b):
    """
    Returns (status, sr, sg, sb, de, reason)
    status: 'ok' | 'missing' | 'corrupted' | 'flat'
    """
    if not path.exists():
        return 'missing', 0, 0, 0, 999.0, 'file not found'

    try:
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32)

        sr = float(arr[:, :, 0].mean())
        sg = float(arr[:, :, 1].mean())
        sb = float(arr[:, :, 2].mean())

        # Check for blank/white
        if sr > BRIGHTNESS_THRESHOLD and sg > BRIGHTNESS_THRESHOLD and sb > BRIGHTNESS_THRESHOLD:
            return 'corrupted', int(sr), int(sg), int(sb), 999.0, 'blank/white image'

        # Check for nearly black
        if sr < 15 and sg < 15 and sb < 15:
            return 'corrupted', int(sr), int(sg), int(sb), 999.0, 'nearly black image'

        # Check for flat color block (old pipeline remnant)
        std_dev = float(arr.std())
        if std_dev < FLAT_STD_THRESHOLD:
            return 'flat', int(sr), int(sg), int(sb), 0.0, \
                f'flat color block detected (std={std_dev:.2f}) — re-run process_real_photos.py'

        # Check color accuracy
        de = delta_e(sr, sg, sb, expected_r, expected_g, expected_b)
        if de > DE_THRESHOLD:
            return 'corrupted', int(sr), int(sg), int(sb), round(de, 1), \
                f'color too far from expected (ΔE={de:.1f})'

        return 'ok', int(sr), int(sg), int(sb), round(de, 1), 'pass'

    except Exception as e:
        return 'corrupted', 0, 0, 0, 999.0, f'read error: {e}'


def main():
    if not RHS_CSV.exists():
        print(f"ERROR: rhs_colors.csv not found at {RHS_CSV}")
        print(f"Run: python build_rhs_csv.py first")
        return

    if not SWATCHES_DIR.exists():
        print(f"ERROR: {SWATCHES_DIR}/ not found")
        print(f"Run: python process_real_photos.py first")
        return

    with open(RHS_CSV, encoding='utf-8') as f:
        colors = list(csv.DictReader(f))

    print(f"{'='*60}")
    print(f"  SWATCH VALIDATION — {len(colors)} RHS colors")
    print(f"{'='*60}")
    print(f"  Swatches folder : {SWATCHES_DIR.resolve()}")
    print(f"  ΔE threshold    : {DE_THRESHOLD}  (flag if color drift > this)")
    print(f"  Flat threshold  : std < {FLAT_STD_THRESHOLD}  (flag if old flat swatch)")
    print()

    report          = []
    ok_count        = 0
    corrupted_count = 0
    missing_count   = 0
    flat_count      = 0

    for color in colors:
        class_label = color['class_label']
        exp_r = int(color['R'])
        exp_g = int(color['G'])
        exp_b = int(color['B'])
        path  = SWATCHES_DIR / f"{class_label}.png"

        status, sr, sg, sb, de, reason = check_swatch(path, exp_r, exp_g, exp_b)

        report.append({
            'class_label': class_label,
            'status':      status,
            'expected_R':  exp_r, 'expected_G': exp_g, 'expected_B': exp_b,
            'sampled_R':   sr,    'sampled_G':  sg,    'sampled_B':  sb,
            'delta_e':     de,
            'reason':      reason,
        })

        if status == 'ok':
            ok_count += 1
        elif status == 'missing':
            missing_count += 1
            print(f"  ❌ MISSING   {class_label:<30} expected=({exp_r},{exp_g},{exp_b})")
        elif status == 'flat':
            flat_count += 1
            print(f"  ⬜ FLAT      {class_label:<30} {reason}")
        else:
            corrupted_count += 1
            print(f"  ⚠️  CORRUPT  {class_label:<30} sampled=({sr},{sg},{sb})  {reason}")

    # Save report
    report_path = SWATCHES_DIR / "validation_report.csv"
    fieldnames  = ['class_label', 'status', 'expected_R', 'expected_G', 'expected_B',
                   'sampled_R', 'sampled_G', 'sampled_B', 'delta_e', 'reason']
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(report)

    print()
    print(f"{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  ✅  OK           : {ok_count}")
    print(f"  ⚠️   Corrupted    : {corrupted_count}")
    print(f"  ❌  Missing       : {missing_count}")
    print(f"  ⬜  Flat (old)    : {flat_count}")
    print(f"  Total            : {len(colors)}")
    print(f"\n  Validation report → {report_path}")

    issues = corrupted_count + missing_count + flat_count
    if issues == 0:
        print(f"\n  🎉 All swatches are clean real texture! Ready for augmentation.")
    else:
        print(f"\n  ⚠️  {issues} swatches need attention:")
        if flat_count:
            print(f"     - {flat_count} flat swatches → re-run process_real_photos.py")
        if missing_count:
            print(f"     - {missing_count} missing → check real_photos/ for these classes")
        if corrupted_count:
            print(f"     - {corrupted_count} corrupted → check real_photos/ for these classes")

    print(f"\n  Next step: python augment_dataset.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()