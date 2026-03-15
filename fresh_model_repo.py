"""
fresh_model_repo.py  —  Abaca Color Scanner
=============================================
Creates lawrencio/abaca-models fresh and uploads
only the new Quad MLP files from abaca_pipeline/.

Usage:
    python fresh_model_repo.py --token hf_...

Run from your project root:
    cd /path/to/abaca_color_ai
    python fresh_model_repo.py --token hf_...
"""

import argparse
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID   = "lawrencio/abaca-models"
REPO_TYPE = "model"

# Only the new Quad MLP files — clean slate, no old junk
UPLOAD_FILES = [
    ("abaca_pipeline/model_mlp_a.joblib",   "abaca_pipeline/model_mlp_a.joblib"),
    ("abaca_pipeline/model_mlp_b.joblib",   "abaca_pipeline/model_mlp_b.joblib"),
    ("abaca_pipeline/model_mlp_c.joblib",   "abaca_pipeline/model_mlp_c.joblib"),
    ("abaca_pipeline/model_mlp_d.joblib",   "abaca_pipeline/model_mlp_d.joblib"),
    ("abaca_pipeline/scaler_knn.joblib",    "abaca_pipeline/scaler_knn.joblib"),
    ("abaca_pipeline/label_encoder.joblib", "abaca_pipeline/label_encoder.joblib"),
    ("abaca_pipeline/model_config.json",    "abaca_pipeline/model_config.json"),
    ("abaca_pipeline/rhs_colors.csv",       "abaca_pipeline/rhs_colors.csv"),
]


def human_size(path):
    size = Path(path).stat().st_size
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    return f"{size / 1_000:.0f} KB"


def check_local_files():
    missing = [local for local, _ in UPLOAD_FILES if not Path(local).exists()]
    if missing:
        print("\n❌ Missing local files:")
        for f in missing:
            print(f"   {f}")
        print("\nRun from your abaca_color_ai project root.")
        return False
    return True


def run(token: str):
    api = HfApi(token=token)

    print(f"\n🌿 Abaca Model Repo — Fresh Upload")
    print(f"   Repo : {REPO_ID}")
    print(f"   URL  : https://huggingface.co/{REPO_ID}\n")

    if not check_local_files():
        return

    # ── Step 1: Create fresh repo ─────────────────────────────────────────────
    print("  🆕 Creating fresh repo ...")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            exist_ok=True,
        )
        print(f"  ✅ Repo created: https://huggingface.co/{REPO_ID}\n")
    except Exception as e:
        print(f"  ❌ Repo creation failed: {e}\n")
        return

    time.sleep(3)

    # ── Step 2: Upload files one by one ───────────────────────────────────────
    print(f"  📦 Uploading {len(UPLOAD_FILES)} files ...\n")
    success, failed = 0, []

    for local, repo_path in UPLOAD_FILES:
        size = human_size(local)
        print(f"  ⬆️  {repo_path:<45} ({size}) ...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Upload: {repo_path}",
            )
            print("✅")
            success += 1
        except Exception as e:
            print(f"❌  {e}")
            failed.append(repo_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Uploaded : {success}/{len(UPLOAD_FILES)}")
    if failed:
        print(f"  Failed   : {len(failed)}")
        for f in failed:
            print(f"    ✗ {f}")
    print(f"{'─'*60}\n")

    if not failed:
        print("🚀 Model repo ready!")
        print(f"   https://huggingface.co/{REPO_ID}\n")
        print("   Next — deploy the Space:")
        print("   python deploy_to_hf.py --token hf_...\n")
    else:
        print("⚠️  Some files failed. Re-run to retry.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fresh upload of Quad MLP models to lawrencio/abaca-models"
    )
    parser.add_argument("--token", required=True, help="HF WRITE token (hf_...)")
    args = parser.parse_args()
    run(args.token)