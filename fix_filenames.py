"""
fix_filenames.py  (v3 — corrective)
====================================
Two operations in one pass:

1. REMOVE wrong N prefix from rhs_code (added by a previous bad script run)
   e.g. VIOLET_GROUP_N83A  ->  VIOLET_GROUP_83A   (83 never had N)

2. ADD correct N prefix to rhs_code (only specific codes from XYZN_GROUP_ originals)
   e.g. VIOLET_GROUP_87A   ->  VIOLET_GROUP_N87A   (87 came from VIOLETN_GROUP_87A)

Ground truth derived from original fan zip filenames.
"""

from pathlib import Path
import argparse

# For each color: the ONLY rhs code numbers that should have N prefix.
# Everything else for that color should NOT have N.
CORRECT_N_CODES = {
    "BLUE":         {"109"},
    "ORANGE":       {"25", "30"},
    "ORANGERED":    {"34"},
    "PURPLE":       {"77", "78", "79"},
    "PURPLEVIOLET": {"80", "81", "82"},
    "REDPURPLE":    {"57", "66", "74"},
    "VIOLETBLUE":   {"89", "92"},
    "VIOLET":       {"87", "88"},
}

REAL_PHOTOS_DIR = Path("real_photos")


def get_rhs_number(rhs_code):
    """Extract numeric part: 'N109A' or '109A' -> '109'"""
    code = rhs_code.lstrip("N")
    return ''.join(filter(str.isdigit, code))


def main(dry_run):
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    all_images = []
    for ext in extensions:
        all_images.extend(REAL_PHOTOS_DIR.rglob(f"*{ext}"))
    all_images = list(set(all_images))
    all_images.sort()

    to_rename  = []
    already_ok = []

    for img_path in all_images:
        upper = img_path.stem.upper()
        if "_GROUP_" not in upper:
            continue
        parts = upper.split("_GROUP_")
        if len(parts) != 2:
            continue

        color_name = parts[0]
        rhs_code   = parts[1]

        if color_name not in CORRECT_N_CODES:
            already_ok.append(img_path.name)
            continue

        correct_n_numbers = CORRECT_N_CODES[color_name]
        rhs_number    = get_rhs_number(rhs_code)
        has_n         = rhs_code.startswith("N")
        should_have_n = rhs_number in correct_n_numbers

        if has_n and not should_have_n:
            # N was wrongly added -- remove it
            clean_code = rhs_code[1:]
            new_name = f"{color_name}_GROUP_{clean_code}{img_path.suffix}"
            to_rename.append((img_path, img_path.parent / new_name, "REMOVE N"))

        elif not has_n and should_have_n:
            # N is missing -- add it
            new_name = f"{color_name}_GROUP_N{rhs_code}{img_path.suffix}"
            to_rename.append((img_path, img_path.parent / new_name, "ADD N"))

        else:
            already_ok.append(img_path.name)

    add_n    = [x for x in to_rename if x[2] == "ADD N"]
    remove_n = [x for x in to_rename if x[2] == "REMOVE N"]

    print(f"Files to fix    : {len(to_rename)}")
    print(f"  ADD N         : {len(add_n)}")
    print(f"  REMOVE N      : {len(remove_n)}")
    print(f"Already correct : {len(already_ok)}")

    if to_rename:
        print("\nChanges:")
        for old, new, tag in to_rename:
            print(f"  [{tag}]  {old.name}  ->  {new.name}")

    if dry_run:
        print("\n[dry-run] No files were changed.")
        return

    if to_rename:
        for old, new, _ in to_rename:
            old.rename(new)
        print(f"\nDone! Fixed {len(to_rename)} files.")
        print("Now run: python build_rhs_csv.py")
    else:
        print("\nNothing to rename.")


parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Preview changes without renaming files")
args = parser.parse_args()
main(dry_run=args.dry_run)