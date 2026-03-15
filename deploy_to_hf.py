"""
deploy_to_hf.py — Deploy new Single MLP to Hugging Face

Usage:
    python deploy_to_hf.py           # deploy everything
    python deploy_to_hf.py --models  # models only (faster)
    python deploy_to_hf.py --app     # app files only
    python deploy_to_hf.py --check   # check what will be uploaded without uploading

What this uploads:
    MODELS (required for inference):
        model_mlp_a.joblib      — new single MLP (replaces quad)
        scaler_knn.joblib       — new scaler
        label_encoder.joblib    — new label encoder
        model_config.json       — updated config (model_type: SingleMLP_v1)
        rhs_colors.csv          — color reference (unchanged)

    APP FILES (code running on HF):
        app.py                  — updated (mlp_b/c/d optional)
        features.py             — updated (single model inference)
        db.py, segment.py       — unchanged

    PIPELINE SCRIPTS:
        augment_dataset.py      — updated (fiber augmentations, N=300)
        train_model.py          — updated (single MLP)
        evaluate.py             — updated (single MLP report)
        build_rhs_csv.py, process_real_photos.py, run_pipeline.py

    REPORT:
        report/evaluation_report.txt
        report/confusion_matrix.png
        report/delta_e_distribution.png
        report/model_comparison.png
        report/color_comparison_grid.png
        report/per_class_metrics.csv

NOTE: model_mlp_b/c/d.joblib are NOT uploaded.
      app.py on HF loads them as optional — no crash if missing.
      The live app will use single MLP automatically.
"""
import sys
from pathlib import Path

HF_REPO = "lawrencio/abaca-color-scanner"

# ── File lists ────────────────────────────────────────────────────────────────

MODEL_FILES = [
    # Core — required
    "abaca_pipeline/model_mlp_a.joblib",
    "abaca_pipeline/scaler_knn.joblib",
    "abaca_pipeline/label_encoder.joblib",
    "abaca_pipeline/model_config.json",
    "abaca_pipeline/rhs_colors.csv",
    # NOTE: b/c/d intentionally excluded — app.py handles their absence
]

APP_FILES = [
    "app.py",
    "features.py",
    "db.py",
    "segment.py",
]

PIPELINE_FILES = [
    "augment_dataset.py",
    "train_model.py",
    "evaluate.py",
    "build_rhs_csv.py",
    "process_real_photos.py",
    "run_pipeline.py",
    "deploy_to_hf.py",
]

REPORT_FILES = [
    "report/evaluation_report.txt",
    "report/confusion_matrix.png",
    "report/delta_e_distribution.png",
    "report/model_comparison.png",
    "report/color_comparison_grid.png",
    "report/per_class_metrics.csv",
]

ALL_FILES = MODEL_FILES + APP_FILES + PIPELINE_FILES + REPORT_FILES


def check_files():
    """Check which files exist and which are missing before uploading."""
    base_dir = Path(__file__).parent
    print(f"\n{'='*65}")
    print(f"  PRE-DEPLOY CHECK")
    print(f"{'='*65}")

    missing_critical = []
    missing_optional = []
    ready = []

    critical = set(MODEL_FILES + APP_FILES)

    for rel_path in ALL_FILES:
        local = base_dir / rel_path
        if local.exists():
            size_mb = local.stat().st_size / 1e6
            ready.append((rel_path, size_mb))
        elif rel_path in critical:
            missing_critical.append(rel_path)
        else:
            missing_optional.append(rel_path)

    print(f"\n  ✅ Ready to upload ({len(ready)} files):")
    for rel_path, size_mb in ready:
        print(f"     {rel_path:<55} ({size_mb:.1f} MB)")

    if missing_critical:
        print(f"\n  ❌ MISSING CRITICAL ({len(missing_critical)} files) — fix before deploying:")
        for f in missing_critical:
            print(f"     {f}")

    if missing_optional:
        print(f"\n  ⚠️  Missing optional ({len(missing_optional)} files) — will be skipped:")
        for f in missing_optional:
            print(f"     {f}")

    # Verify model_config shows SingleMLP
    config_path = base_dir / "abaca_pipeline/model_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get('model_type', 'unknown')
        val_acc    = cfg.get('mlp_a_val_accuracy', 'unknown')
        arch       = cfg.get('mlp_a_architecture', 'unknown')
        print(f"\n  Model info:")
        print(f"     Type         : {model_type}")
        print(f"     Architecture : {arch}")
        print(f"     Val accuracy : {val_acc}%")
        if model_type == 'SingleMLP_v1':
            print(f"     Status       : ✅ New single MLP — ready to deploy")
        else:
            print(f"     Status       : ⚠️  Still showing old model type — did training finish?")

    print(f"\n{'='*65}")
    return len(missing_critical) == 0


def deploy(file_list, label):
    """Upload a list of files to HuggingFace."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌  huggingface_hub not installed.")
        print("    Run: pip install huggingface_hub")
        return False

    try:
        api  = HfApi()
        user = api.whoami()
        print(f"✅  Logged in as: {user['name']}")
    except Exception:
        print("❌  Not logged in to Hugging Face.")
        print("    Run: huggingface-cli login")
        return False

    try:
        api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")
        print(f"✅  Repo: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"⚠️  Repo warning: {e}")

    base_dir = Path(__file__).parent
    uploaded, skipped, failed = 0, 0, []

    print(f"\nUploading {label} ({len(file_list)} files) ...")
    for rel_path in file_list:
        local_path = base_dir / rel_path
        if not local_path.exists():
            print(f"  ⚠️  Skip  : {rel_path}")
            skipped += 1
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=rel_path,
                repo_id=HF_REPO,
                repo_type="model",
            )
            size_mb = local_path.stat().st_size / 1e6
            print(f"  ✅  {rel_path:<55} ({size_mb:.1f} MB)")
            uploaded += 1
        except Exception as e:
            print(f"  ❌  Failed : {rel_path} — {e}")
            failed.append(rel_path)

    print(f"\n  Uploaded : {uploaded}")
    print(f"  Skipped  : {skipped}")
    print(f"  Failed   : {len(failed)}")

    if failed:
        print(f"\n  Failed files:")
        for f in failed:
            print(f"    {f}")

    return len(failed) == 0


def main():
    args = sys.argv[1:]

    print(f"\n{'='*65}")
    print(f"  DEPLOY TO HUGGING FACE")
    print(f"  Repo: {HF_REPO}")
    print(f"{'='*65}")

    # --check: just show what would be uploaded
    if '--check' in args:
        check_files()
        return

    # Always run check first
    ok = check_files()
    if not ok:
        print("\n❌  Critical files missing. Fix them before deploying.")
        print("    Make sure train_model.py finished successfully.")
        return

    print()

    if '--models' in args:
        # Models only
        ok = deploy(MODEL_FILES, "MODELS")

    elif '--app' in args:
        # App code only
        ok = deploy(APP_FILES, "APP FILES")

    else:
        # Full deploy — models first, then app, then pipeline, then report
        print("[1/4] Uploading models ...")
        ok = deploy(MODEL_FILES, "models")
        if not ok:
            print("\n❌  Model upload failed. Check errors above.")
            return

        print("\n[2/4] Uploading app files ...")
        deploy(APP_FILES, "app files")

        print("\n[3/4] Uploading pipeline scripts ...")
        deploy(PIPELINE_FILES, "pipeline scripts")

        print("\n[4/4] Uploading report ...")
        deploy(REPORT_FILES, "report")

    print(f"\n{'='*65}")
    print(f"  DEPLOY COMPLETE")
    print(f"  Live app: https://huggingface.co/spaces/{HF_REPO.replace('/', '/')}")
    print(f"{'='*65}")
    print()
    print("  The live app will now use Single MLP automatically.")
    print("  model_mlp_b/c/d are not uploaded — app.py handles this gracefully.")
    print()
    print("  To verify the live app is using the new model:")
    print("  1. Open the HF Space URL above")
    print("  2. Log in and go to Settings → it should show SingleMLP_v1")
    print()


if __name__ == "__main__":
    main()