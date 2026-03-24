"""
deploy_to_hf.py — Abaca Color Scanner Deployment
==================================================

TWO separate HF repos:
    MODEL_REPO : lawrencio/abaca-models         — joblib/csv/json model files only
    SPACE_REPO : lawrencio/abaca-color-scanner  — app code (app.py, templates/, etc.)

WHAT GETS UPLOADED WHERE:
    abaca-models repo  → abaca_pipeline/ subfolder in repo
        Only inference files: *.joblib, rhs_colors.csv, model_config.json
        Excludes everything else: augmented/, backup/, swatches/, *.py, *.png, manifest

    abaca-color-scanner Space → app code only, NO model files
        app.py, features.py, db.py, segment.py, download_models.py
        Dockerfile, requirements.txt, sw.js (optional)
        templates/  — all HTML/CSS/JS

Usage:
    python deploy_to_hf.py           # deploy app code to Space only (most common)
    python deploy_to_hf.py --models  # push model files to abaca-models only
    python deploy_to_hf.py --all     # push both models + app
    python deploy_to_hf.py --check   # preview what will be uploaded, no upload

After --models or --all: restart the Space so it re-downloads the new model files.
"""
import sys, json
from pathlib import Path

MODEL_REPO   = "lawrencio/abaca-models"
SPACE_REPO   = "lawrencio/abaca-color-scanner"
PIPELINE_DIR = Path("abaca_pipeline")

# ── EXACTLY these 5 files go to the model repo — nothing else ────────────────
# Single MLP — only mlp_a. b/c/d are old Quad ensemble files, not needed.
MODEL_INFERENCE_FILES = [
    "model_mlp_a.joblib",
    "scaler_knn.joblib",
    "label_encoder.joblib",
    "model_config.json",
    "rhs_colors.csv",
]

# ── Required app files → Space repo ──────────────────────────────────────────
SPACE_APP_FILES = [
    "app.py",
    "features.py",
    "db.py",
    "segment.py",
    "download_models.py",
    "Dockerfile",
    "requirements.txt",
]

# ── Optional app files (only uploaded if they exist) ─────────────────────────
SPACE_OPTIONAL_FILES = [
    "sw.js",
]


def get_template_files(base_dir: Path) -> list:
    """Auto-discover all files in templates/ folder."""
    templates_dir = base_dir / "templates"
    if not templates_dir.exists():
        return []
    return [
        str(p.relative_to(base_dir)).replace("\\", "/")
        for p in sorted(templates_dir.rglob("*"))
        if p.is_file()
    ]


def human_size(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1_000_000_000: return f"{size/1e9:.2f} GB"
    if size >= 1_000_000:     return f"{size/1e6:.1f} MB"
    return f"{size/1000:.0f} KB"


def login_check() -> bool:
    """
    Auth priority:
      1. HF_TOKEN environment variable  (export HF_TOKEN=hf_...)
      2. huggingface-cli login cache     (huggingface-cli login)
    No --token flag needed if either is set.
    """
    import os
    try:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        api   = HfApi(token=token) if token else HfApi()
        user  = api.whoami()
        src   = "HF_TOKEN env var" if token else "huggingface-cli login cache"
        print(f"✅  Logged in as: {user['name']}  (via {src})")
        return True
    except ImportError:
        print("❌  huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception:
        print("❌  Not authenticated. Choose one:")
        print("    Option 1 (recommended): export HF_TOKEN=hf_...")
        print("    Option 2: huggingface-cli login")
        return False

def check_files(base_dir: Path) -> bool:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  PRE-DEPLOY CHECK")
    print(f"{sep}")

    all_ok = True

    # Model config info
    config_path = base_dir / PIPELINE_DIR / "model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"\n  Model info:")
        print(f"    Type     : {cfg.get('model_type', '?')}")
        print(f"    Classes  : {cfg.get('n_classes', '?')}")
        print(f"    Val acc  : {cfg.get('mlp_a_val_accuracy', '?')}%")
    else:
        print(f"\n  ⚠️  model_config.json not found in {PIPELINE_DIR}/")

    # ── Model files check ─────────────────────────────────────────────────────
    print(f"\n  → MODEL REPO: {MODEL_REPO}")
    print(f"    Destination in repo: abaca_pipeline/")
    print(f"    Files:")
    missing_models = []
    for fname in MODEL_INFERENCE_FILES:
        p = base_dir / PIPELINE_DIR / fname
        if p.exists():
            print(f"      ✅  {fname:<40} {human_size(p)}")
        else:
            print(f"      ❌  {fname:<40} ← MISSING")
            missing_models.append(fname)
            all_ok = False

    if missing_models:
        print(f"\n    ❌ {len(missing_models)} model file(s) missing")

    # ── Space files check ─────────────────────────────────────────────────────
    print(f"\n  → SPACE REPO: {SPACE_REPO}")
    print(f"    Required files:")
    for f in SPACE_APP_FILES:
        p = base_dir / f
        if p.exists():
            print(f"      ✅  {f:<40} {human_size(p)}")
        else:
            print(f"      ❌  {f:<40} ← MISSING")
            all_ok = False

    print(f"    Optional files:")
    for f in SPACE_OPTIONAL_FILES:
        p = base_dir / f
        if p.exists():
            print(f"      ✅  {f}")
        else:
            print(f"      ⚠️  {f}  (not found — will skip)")

    templates = get_template_files(base_dir)
    print(f"    Templates: {len(templates)} files in templates/")
    if not templates:
        print(f"      ⚠️  templates/ folder not found")

    print(f"\n{sep}")
    if all_ok:
        print(f"  ✅  All required files present — safe to deploy")
    else:
        print(f"  ❌  Fix missing files before deploying")
    print(f"{sep}\n")
    return all_ok


# ── UPLOAD MODELS ─────────────────────────────────────────────────────────────
def upload_model_files(base_dir: Path) -> bool:
    """
    Upload ONLY the 5 inference files to lawrencio/abaca-models
    under abaca_pipeline/ subfolder in the repo.

    Repo structure after upload:
        abaca_pipeline/
            model_mlp_a.joblib   ← Single MLP (only this one)
            scaler_knn.joblib
            label_encoder.joblib
            model_config.json
            rhs_colors.csv
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except ImportError:
        print("❌  huggingface_hub not installed.")
        return False

    files_to_upload = []
    for fname in MODEL_INFERENCE_FILES:
        p = base_dir / PIPELINE_DIR / fname
        if p.exists():
            files_to_upload.append((p, f"abaca_pipeline/{fname}"))
        else:
            print(f"  ⚠️  Skip (not found): {fname}")

    if not files_to_upload:
        print("❌  No model files found — aborting.")
        return False

    print(f"\nUploading {len(files_to_upload)} model files → {MODEL_REPO}/abaca_pipeline/\n")

    success, failed = 0, []
    for local_path, repo_path in files_to_upload:
        print(f"  ⬆️  {repo_path:<55} ({human_size(local_path)}) ...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"Update: {local_path.name}",
            )
            print("✅")
            success += 1
        except Exception as e:
            err = str(e)
            if "No files have been modified" in err:
                print("ℹ️  Already up to date")
                success += 1
            else:
                print(f"❌  {e}")
                failed.append(repo_path)

    print(f"\n  Uploaded: {success}/{len(files_to_upload)}", end="")
    if failed:
        print(f"  |  Failed: {len(failed)}")
        for f in failed:
            print(f"    ❌ {f}")
    else:
        print()
    return len(failed) == 0


# ── UPLOAD SPACE ──────────────────────────────────────────────────────────────
def upload_space_files(base_dir: Path) -> bool:
    """
    Upload app code to the Space repo.
    NEVER uploads model files — those stay in abaca-models repo.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
    except ImportError:
        print("❌  huggingface_hub not installed.")
        return False

    all_files = list(SPACE_APP_FILES)
    for f in SPACE_OPTIONAL_FILES:
        if (base_dir / f).exists():
            all_files.append(f)
    all_files += get_template_files(base_dir)

    # Safety guard: never let model files sneak into the Space repo
    model_fnames = set(MODEL_INFERENCE_FILES)
    all_files = [f for f in all_files if Path(f).name not in model_fnames]

    print(f"\nUploading {len(all_files)} app files → {SPACE_REPO}\n")

    success, skipped, failed = 0, 0, []
    for rel_path in all_files:
        local_path = base_dir / rel_path
        if not local_path.exists():
            print(f"  ⚠️  Skip (not found): {rel_path}")
            skipped += 1
            continue
        print(f"  ⬆️  {rel_path:<55} ({human_size(local_path)}) ...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=rel_path.replace("\\", "/"),
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"Deploy: {Path(rel_path).name}",
            )
            print("✅")
            success += 1
        except Exception as e:
            err = str(e)
            if "No files have been modified" in err:
                print("ℹ️  No change")
                success += 1
            else:
                print(f"❌  {e}")
                failed.append(rel_path)

    print(f"\n  Uploaded: {success}  |  Skipped: {skipped}  |  Failed: {len(failed)}")
    if failed:
        for f in failed:
            print(f"    ❌ {f}")
    return len(failed) == 0


# ── SINGLE FILE UPLOAD ────────────────────────────────────────────────────────
def upload_single_files(base_dir: Path, files: list) -> bool:
    """
    Upload specific files by name.
    Auto-detects destination repo:
      - Model files (*.joblib, rhs_colors.csv, model_config.json) → MODEL_REPO
      - Everything else → SPACE_REPO
    """
    import os
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    api   = HfApi(token=token) if token else HfApi()

    model_names = set(MODEL_INFERENCE_FILES)
    success, failed = 0, []

    for fname in files:
        local_path = base_dir / fname
        if not local_path.exists():
            print(f"  ❌  {fname} — file not found locally")
            failed.append(fname)
            continue

        # Auto-detect destination
        basename = Path(fname).name
        is_model = (
            basename in model_names or
            basename.endswith(".joblib") or
            basename in ("rhs_colors.csv", "model_config.json")
        )

        if is_model:
            repo_id   = MODEL_REPO
            repo_type = "model"
            repo_path = f"abaca_pipeline/{basename}"
        else:
            repo_id   = SPACE_REPO
            repo_type = "space"
            repo_path = str(Path(fname)).replace("\\", "/")

        print(f"  ⬆️  {fname:<45} → {repo_id}/{repo_path} ...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Update: {basename}",
            )
            print(f"✅  ({human_size(local_path)})")
            success += 1
        except Exception as e:
            err = str(e)
            if "No files have been modified" in err:
                print("ℹ️  No change")
                success += 1
            else:
                print(f"❌  {e}")
                failed.append(fname)

    print(f"\n  Done: {success} uploaded, {len(failed)} failed")
    return len(failed) == 0


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args     = sys.argv[1:]
    base_dir = Path(__file__).parent

    print(f"\n{'='*65}")
    print(f"  ABACA SCANNER — DEPLOY TO HUGGING FACE")
    print(f"  Model repo : {MODEL_REPO}")
    print(f"  Space repo : {SPACE_REPO}")
    print(f"{'='*65}")

    if "--check" in args:
        check_files(base_dir)
        return

    # ── --files: upload specific files ────────────────────────────────────────
    if "--files" in args:
        idx = args.index("--files")
        files = [a for a in args[idx+1:] if not a.startswith("--")]
        if not files:
            print("❌  --files requires at least one filename.")
            print("    Example: python deploy_to_hf.py --files app.py features.py")
            print("    Example: python deploy_to_hf.py --files abaca_pipeline/label_encoder.joblib")
            return
        if not login_check():
            return
        print(f"\n  Uploading {len(files)} specific file(s) ...\n")
        upload_single_files(base_dir, files)
        print(f"\n{'='*65}\n")
        return

    if not check_files(base_dir):
        print("❌  Fix missing files before deploying.")
        return

    if not login_check():
        return

    do_models = "--models" in args or "--all" in args
    # Default (no flags) = app only
    do_app = "--app" in args or "--all" in args or (
        "--models" not in args and "--all" not in args
    )

    if do_models:
        print("\n[1] Uploading model files → abaca-models repo ...")
        ok = upload_model_files(base_dir)
        if not ok:
            print("\n❌  Model upload had failures.")

    if do_app:
        label = "[2]" if do_models else "[1]"
        print(f"\n{label} Uploading app code → Space repo ...")
        upload_space_files(base_dir)

    print(f"\n{'='*65}")
    print(f"  DEPLOY COMPLETE")
    print(f"{'='*65}")

    if do_models:
        print(f"\n  ⚠️  Restart the Space to pull the new models:")
        print(f"  https://huggingface.co/spaces/{SPACE_REPO} → Settings → Restart Space")
    else:
        print(f"\n  Space will restart automatically (~90 seconds).")
        print(f"  https://lawrencio-abaca-color-scanner.hf.space")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()