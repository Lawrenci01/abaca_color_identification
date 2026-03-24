"""
cleanup_space_repo.py — Abaca Color Scanner
=============================================
Removes files that should NOT be in the Space repo:
  - abaca_pipeline/ folder (models belong in abaca-models repo only)
  - abaca_ui.html (old file)
  - index.html at root (should be in templates/ only)
  - deploy_to_hf.py (local dev script, not needed in Space)

Run from your project root:
    python cleanup_space_repo.py
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationDelete

SPACE_REPO = "lawrencio/abaca-color-scanner"
REPO_TYPE  = "space"

# ── Files/folders to DELETE from the Space repo ───────────────────────────────
# Add any path you want removed. Folders must be listed file by file
# (HF API deletes files, not folders — folder disappears when empty).
FILES_TO_DELETE = [
    # Old/wrong files at root
    "abaca_ui.html",
    "index.html",
    "deploy_to_hf.py",

    # abaca_pipeline/ — models belong in abaca-models repo, not here
    # Single MLP files
    "abaca_pipeline/model_mlp_a.joblib",
    # Old Quad MLP files (if still present)
    "abaca_pipeline/model_mlp_b.joblib",
    "abaca_pipeline/model_mlp_c.joblib",
    "abaca_pipeline/model_mlp_d.joblib",
    # Supporting files
    "abaca_pipeline/scaler_knn.joblib",
    "abaca_pipeline/label_encoder.joblib",
    "abaca_pipeline/model_config.json",
    "abaca_pipeline/rhs_colors.csv",
    "abaca_pipeline/augmented_manifest.csv",
    "abaca_pipeline/features_cache.npy",
    "abaca_pipeline/labels_cache.joblib",
]


def human_size(size_bytes):
    if size_bytes >= 1_000_000: return f"{size_bytes/1e6:.1f} MB"
    return f"{size_bytes/1000:.0f} KB"


def get_all_space_files(api, repo_id):
    """List all files currently in the Space repo."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="space")
        return set(files)
    except Exception as e:
        print(f"❌  Could not list repo files: {e}")
        return set()


def main():
    import os
    token = os.environ.get("HF_TOKEN")

    print(f"\n{'='*60}")
    print(f"  SPACE REPO CLEANUP")
    print(f"  Repo : {SPACE_REPO}")
    print(f"{'='*60}\n")

    try:
        api  = HfApi(token=token) if token else HfApi()
        user = api.whoami()
        print(f"✅  Logged in as: {user['name']}\n")
    except Exception as e:
        print(f"❌  Auth failed: {e}")
        print(f"    Set HF_TOKEN env var or run: huggingface-cli login")
        return

    # Get current files in the repo
    print("📋  Fetching current file list from Space repo ...")
    existing = get_all_space_files(api, SPACE_REPO)
    print(f"    Found {len(existing)} files in repo\n")

    # Figure out what actually needs deleting
    to_delete = [f for f in FILES_TO_DELETE if f in existing]
    not_found = [f for f in FILES_TO_DELETE if f not in existing]

    if not_found:
        print(f"  ℹ️  Already gone (skipping):")
        for f in not_found:
            print(f"     {f}")
        print()

    if not to_delete:
        print("✅  Nothing to delete — Space repo is already clean!\n")
        return

    print(f"  🗑️  Will delete {len(to_delete)} file(s):")
    for f in to_delete:
        print(f"     ✗  {f}")

    print(f"\n  Proceeding with deletion ...\n")

    # Build delete operations
    operations = [CommitOperationDelete(path_in_repo=f) for f in to_delete]

    try:
        api.create_commit(
            repo_id=SPACE_REPO,
            repo_type=REPO_TYPE,
            operations=operations,
            commit_message=f"Cleanup: remove abaca_pipeline/ and stale files from Space repo",
        )
        print(f"  ✅  Deleted {len(to_delete)} file(s) in one commit.\n")

    except Exception as e:
        err = str(e)
        if "No files have been modified" in err:
            print("  ℹ️  No changes — files already removed.\n")
            return
        # Fallback: delete one by one
        print(f"  ⚠️  Batch delete failed ({e}). Trying one by one ...\n")
        success, failed = 0, []
        for f in to_delete:
            print(f"  🗑️  {f} ...", end=" ", flush=True)
            try:
                api.delete_file(
                    path_in_repo=f,
                    repo_id=SPACE_REPO,
                    repo_type=REPO_TYPE,
                    commit_message=f"Cleanup: remove {f}",
                )
                print("✅")
                success += 1
            except Exception as e2:
                err2 = str(e2)
                if "404" in err2 or "not found" in err2.lower():
                    print("ℹ️  Already gone")
                    success += 1
                else:
                    print(f"❌  {e2}")
                    failed.append(f)

        print(f"\n  Deleted: {success}  |  Failed: {len(failed)}")
        if failed:
            for f in failed:
                print(f"    ✗ {f}")

    print(f"{'='*60}")
    print(f"  CLEANUP COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Space repo should now only contain:")
    print(f"  app.py, features.py, db.py, segment.py,")
    print(f"  download_models.py, Dockerfile, requirements.txt,")
    print(f"  sw.js, static/, templates/")
    print(f"\n  Models live in: https://huggingface.co/lawrencio/abaca-models")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()