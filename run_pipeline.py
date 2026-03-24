"""
run_pipeline.py — Abaca Color AI Pipeline (Single MLP, 876 classes)

Changes vs previous version:
- Updated step order for new real-photo pipeline
- step1: build_rhs_csv.py     (Builds RHS colors CSV from real photos)
- step2: process_real_photos.py (NEW — saves real texture swatches)
- step3: augment_dataset.py
- step4: train_model.py        (Single MLP)
- step5: evaluate.py
- step6: app.py
- Removed rgb_regression_scatter.png (no longer generated)
- Removed extract_colors.py from HF upload (replaced by build_rhs_csv.py)
- Added --validate flag to run validate_swatches.py between steps 2 and 3

Usage:
    python run_pipeline.py              # full pipeline (steps 1-5)
    python run_pipeline.py --train      # steps 2-4 only (skip build_rhs_csv)
    python run_pipeline.py --eval       # step 5 only
    python run_pipeline.py --serve      # launch app
    python run_pipeline.py --deploy     # upload to Hugging Face only
    python run_pipeline.py --all        # steps 1-5 + deploy
    python run_pipeline.py --validate   # validate swatches only
"""
import sys, importlib.util
from pathlib import Path

HF_REPO = "lawrencio/abaca-models"   # model files repo

HF_UPLOAD_FILES = [
    # Models
    "abaca_pipeline/model_mlp_a.joblib",   # Single MLP only
    "abaca_pipeline/scaler_knn.joblib",
    "abaca_pipeline/label_encoder.joblib",
    "abaca_pipeline/model_config.json",
    "abaca_pipeline/rhs_colors.csv",
    # App files
    "app.py", "db.py", "features.py", "segment.py",
    # Pipeline scripts
    "build_rhs_csv.py", "process_real_photos.py",
    "augment_dataset.py", "train_model.py",
    "evaluate.py", "run_pipeline.py",
    # Report
    "report/evaluation_report.txt",
    "report/confusion_matrix.png",
    "report/delta_e_distribution.png",
    "report/model_comparison.png",
    "report/color_comparison_grid.png",
    "report/per_class_metrics.csv",
]


def run_step(script_path, description):
    print(f"\n{'='*65}\n  {description}\n{'='*65}")
    spec   = importlib.util.spec_from_file_location("module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def deploy_to_huggingface():
    print(f"\n{'='*65}\n  DEPLOY → Hugging Face: {HF_REPO}\n{'='*65}")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌  huggingface_hub not installed.\n    Run: pip install huggingface_hub")
        return False
    try:
        api  = HfApi()
        user = api.whoami()
        print(f"✅  Logged in as: {user['name']}")
    except Exception:
        print("❌  Not logged in.\n    Run: huggingface-cli login")
        return False
    try:
        api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")
        print(f"✅  Repo ready: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"⚠️  Repo creation warning: {e}")

    base_dir                = Path(__file__).parent
    uploaded, skipped, failed = 0, 0, []
    print(f"\nUploading {len(HF_UPLOAD_FILES)} files ...")
    for rel_path in HF_UPLOAD_FILES:
        local_path = base_dir / rel_path
        if not local_path.exists():
            print(f"  ⚠️  Skip (not found): {rel_path}")
            skipped += 1
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=rel_path,
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"  ✅  {rel_path:<55} ({local_path.stat().st_size/1e6:.1f} MB)")
            uploaded += 1
        except Exception as e:
            print(f"  ❌  Failed: {rel_path} — {e}")
            failed.append(rel_path)

    print(f"\n{'='*65}")
    print(f"  Uploaded : {uploaded}")
    print(f"  Skipped  : {skipped}  (not found)")
    print(f"  Failed   : {len(failed)}")
    print(f"{'='*65}")
    return len(failed) == 0


if __name__ == "__main__":
    args     = sys.argv[1:]
    base_dir = Path(__file__).parent

    steps = {
        'step1': (base_dir / "build_rhs_csv.py",        "STEP 1: Build RHS colors CSV from real photos"),
        'step2': (base_dir / "process_real_photos.py",  "STEP 2: Extract real texture swatches"),
        'step3': (base_dir / "augment_dataset.py",      "STEP 3: Generate augmented dataset"),
        'step4': (base_dir / "train_model.py",          "STEP 4: Train Single MLP (mlp_a only)"),
        'step5': (base_dir / "evaluate.py",             "STEP 5: Evaluate model & generate report"),
        'step6': (base_dir / "app.py",                  "STEP 6: Launch scanner"),
        'val':   (base_dir / "validate_swatches.py",    "VALIDATE: Check real texture swatches"),
    }

    deploy = '--deploy' in args

    if '--serve' in args:
        # Just launch the app
        run_step(*steps['step6'])

    elif '--eval' in args:
        # Evaluate only
        run_step(*steps['step5'])
        if deploy:
            deploy_to_huggingface()

    elif '--validate' in args:
        # Validate swatches only
        run_step(*steps['val'])

    elif '--train' in args:
        # Skip step1 (build_rhs_csv) — assumes CSV already exists
        # Runs: process_real_photos → augment → train → evaluate
        for k in ['step2', 'step3', 'step4', 'step5']:
            run_step(*steps[k])
        if deploy:
            deploy_to_huggingface()

    elif '--deploy' in args and len([a for a in args if a != '--deploy']) == 0:
        # Deploy only
        deploy_to_huggingface()

    elif '--all' in args:
        # Full pipeline + deploy
        for k in ['step1', 'step2', 'step3', 'step4', 'step5']:
            run_step(*steps[k])
        if deploy:
            deploy_to_huggingface()

    else:
        # Default: full pipeline
        # Skip step1 if rhs_colors.csv already exists
        csv_path = Path("abaca_pipeline/rhs_colors.csv")
        if not csv_path.exists():
            print("rhs_colors.csv not found — running Step 1 first ...")
            run_step(*steps['step1'])
        else:
            n_colors = sum(1 for _ in open(csv_path)) - 1
            print(f"✅  Found existing rhs_colors.csv ({n_colors} colors) — skipping Step 1")

        for k in ['step2', 'step3', 'step4', 'step5']:
            run_step(*steps[k])

        print(f"\n{'='*65}")
        print(f"  ✅  Pipeline complete!")
        print(f"  Launch : python run_pipeline.py --serve")
        print(f"  Deploy : python run_pipeline.py --deploy")
        print(f"{'='*65}")

        if deploy:
            deploy_to_huggingface()