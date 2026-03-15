# download_models.py — Abaca Color Scanner
# Downloads the Quad MLP model files from HF model repo at container startup.
# Run automatically before app.py via Dockerfile CMD.

from huggingface_hub import hf_hub_download
from pathlib import Path
import sys

MODEL_REPO   = "lawrencio/abaca-models"
PIPELINE_DIR = Path("abaca_pipeline")

# ── Files to download (path inside the HF model repo → local destination) ────
FILES = {
    "abaca_pipeline/model_mlp_a.joblib"   : PIPELINE_DIR / "model_mlp_a.joblib",
    "abaca_pipeline/model_mlp_b.joblib"   : PIPELINE_DIR / "model_mlp_b.joblib",
    "abaca_pipeline/model_mlp_c.joblib"   : PIPELINE_DIR / "model_mlp_c.joblib",
    "abaca_pipeline/model_mlp_d.joblib"   : PIPELINE_DIR / "model_mlp_d.joblib",
    "abaca_pipeline/scaler_knn.joblib"    : PIPELINE_DIR / "scaler_knn.joblib",
    "abaca_pipeline/label_encoder.joblib" : PIPELINE_DIR / "label_encoder.joblib",
    "abaca_pipeline/rhs_colors.csv"       : PIPELINE_DIR / "rhs_colors.csv",
    "abaca_pipeline/model_config.json"    : PIPELINE_DIR / "model_config.json",
}

def download_models(force=False):
    PIPELINE_DIR.mkdir(exist_ok=True)

    print(f"\n⬇️  Downloading models from {MODEL_REPO} ...")
    print(f"   Destination : {PIPELINE_DIR.resolve()}\n")

    success, skipped, failed = 0, 0, []

    for repo_path, local_dest in FILES.items():
        filename = local_dest.name

        # Skip if file already exists and force=False
        if local_dest.exists() and not force:
            size_mb = local_dest.stat().st_size / 1_000_000
            print(f"  ✅ {filename:<35} already exists ({size_mb:.1f} MB) — skipping")
            skipped += 1
            continue

        print(f"  ⬇️  {filename:<35} downloading ...", end=" ", flush=True)
        try:
            downloaded = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=repo_path,          # path inside the repo (abaca_pipeline/...)
                local_dir=".",               # root dir; preserves subfolder structure
                repo_type="model",
            )
            # hf_hub_download places file at ./abaca_pipeline/<name> which is local_dest
            size_mb = local_dest.stat().st_size / 1_000_000
            print(f"✅  ({size_mb:.1f} MB)")
            success += 1
        except Exception as e:
            print(f"❌  FAILED")
            print(f"      Error: {e}")
            failed.append(repo_path)

    print(f"\n{'─'*55}")
    print(f"  Downloaded : {success}")
    print(f"  Skipped    : {skipped}  (already present)")
    print(f"  Failed     : {len(failed)}")
    print(f"{'─'*55}\n")

    if failed:
        print("❌ Some model files failed to download. App cannot start.")
        for f in failed:
            print(f"   {f}")
        sys.exit(1)

    print("✅ All model files ready.\n")


if __name__ == "__main__":
    force = "--force" in sys.argv
    download_models(force=force)