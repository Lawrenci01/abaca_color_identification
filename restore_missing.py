"""
restore_missing.py
==================
Restores 24 missing files from extracted fan zip folders.

Usage:
    python restore_missing.py --src-dir /tmp

Where /tmp contains fan1/cropped/ and fan2/cropped/ subfolders.
"""

import shutil
import argparse
from pathlib import Path

RESTORE_MAP = [
    ("fan1/cropped/ORANGEN_GROUP_25A.JPG",    "cropped/ORANGE_GROUP_N25A.JPG"),
    ("fan1/cropped/ORANGEN_GROUP_25B.JPG",    "cropped/ORANGE_GROUP_N25B.JPG"),
    ("fan1/cropped/ORANGEN_GROUP_25C.JPG",    "cropped/ORANGE_GROUP_N25C.JPG"),
    ("fan1/cropped/ORANGEN_GROUP_25D.JPG",    "cropped/ORANGE_GROUP_N25D.JPG"),
    ("fan1/cropped/ORANGEN_GROUP_30A.JPG",    "cropped/ORANGE_GROUP_N30A.JPG"),
    ("fan1/cropped/ORANGEN_GROUP_30B.JPG",    "cropped/ORANGE_GROUP_N30B.JPG"),
    ("fan1/cropped/ORANGEN_GROUP_30C.JPG",    "cropped/ORANGE_GROUP_N30C.JPG"),
    ("fan1/cropped/ORANGEN_GROUP_30D.JPG",    "cropped/ORANGE_GROUP_N30D.JPG"),
    ("fan1/cropped/ORANGEREDN_GROUP_34A.JPG", "cropped/ORANGERED_GROUP_N34A.JPG"),
    ("fan1/cropped/ORANGEREDN_GROUP_34B.JPG", "cropped/ORANGERED_GROUP_N34B.JPG"),
    ("fan1/cropped/ORANGEREDN_GROUP_34C.JPG", "cropped/ORANGERED_GROUP_N34C.JPG"),
    ("fan1/cropped/ORANGEREDN_GROUP_34D.JPG", "cropped/ORANGERED_GROUP_N34D.JPG"),
    ("fan2/cropped/PURPLEN_GROUP_77A.jpg",    "cropped/PURPLE_GROUP_N77A.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_77B.jpg",    "cropped/PURPLE_GROUP_N77B.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_77C.jpg",    "cropped/PURPLE_GROUP_N77C.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_77D.jpg",    "cropped/PURPLE_GROUP_N77D.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_78A.jpg",    "cropped/PURPLE_GROUP_N78A.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_78B.jpg",    "cropped/PURPLE_GROUP_N78B.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_78C.jpg",    "cropped/PURPLE_GROUP_N78C.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_78D.jpg",    "cropped/PURPLE_GROUP_N78D.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_79A.jpg",    "cropped/PURPLE_GROUP_N79A.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_79B.jpg",    "cropped/PURPLE_GROUP_N79B.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_79C.jpg",    "cropped/PURPLE_GROUP_N79C.jpg"),
    ("fan2/cropped/PURPLEN_GROUP_79D.jpg",    "cropped/PURPLE_GROUP_N79D.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_89A.jpg","cropped/VIOLETBLUE_GROUP_N89A.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_89B.jpg","cropped/VIOLETBLUE_GROUP_N89B.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_89C.jpg","cropped/VIOLETBLUE_GROUP_N89C.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_89D.jpg","cropped/VIOLETBLUE_GROUP_N89D.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_92A.jpg","cropped/VIOLETBLUE_GROUP_N92A.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_92B.jpg","cropped/VIOLETBLUE_GROUP_N92B.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_92C.jpg","cropped/VIOLETBLUE_GROUP_N92C.jpg"),
    ("fan2/cropped/VIOLETBLUEN_GROUP_92D.jpg","cropped/VIOLETBLUE_GROUP_N92D.jpg"),
]

REAL_PHOTOS = Path("real_photos")

parser = argparse.ArgumentParser()
parser.add_argument("--src-dir", required=True, help="Folder containing fan1/ and fan2/ subfolders")
args = parser.parse_args()

src_base = Path(args.src_dir)
restored = 0
errors   = 0

for src_rel, dest_rel in RESTORE_MAP:
    src_path  = src_base / src_rel
    dest_path = REAL_PHOTOS / dest_rel

    if dest_path.exists():
        print(f"  ⏭  Already exists: {dest_path.name}")
        continue

    if not src_path.exists():
        print(f"  ❌ Source not found: {src_path}")
        errors += 1
        continue

    shutil.copy2(src_path, dest_path)
    print(f"  ✅ Restored: {dest_path.name}")
    restored += 1

print(f"\nDone! Restored {restored} files, {errors} errors.")
if restored > 0:
    print("Now run: python build_rhs_csv.py")