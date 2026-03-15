"""
train_model.py — Single MLP (fast, ~1-2 hrs at N_AUGMENTS=300)

Why single MLP instead of Quad ensemble:
- 876 classes, 248 features
- 84% of adjacent grade steps have ΔE >= 5 — pure ΔE matching handles these
- Only 15% of classes need MLP to distinguish (ΔE 1-5)
- 1% have ΔE < 1 — no model can reliably separate these
- Single MLP achieves ~91-92% val accuracy vs 93.82% for Quad
- Training time: ~1-2 hrs vs ~18-30 hrs, same real-world Top-5 accuracy

Backup policy:
- ALWAYS backs up existing models before overwriting (never deletes)
- Preserves old a/b/c/d quad models in timestamped folder
- Folder: abaca_pipeline/backup/model_backup_YYYYMMDD_HHMMSS/
"""
import csv, json, time, shutil, datetime
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from features import extract_features

PIPELINE_DIR = Path("abaca_pipeline")
MANIFEST     = PIPELINE_DIR / "augmented_manifest.csv"
X_CACHE      = PIPELINE_DIR / "features_cache.npy"
Y_CACHE      = PIPELINE_DIR / "labels_cache.joblib"

# ── Single MLP config ─────────────────────────────────────────────────────────
# 248 → 1024 → 768 → 512 → N_classes
# Medium depth: handles hard ΔE 1-5 cases, trains in ~1-2 hrs at N=300
MLP_LAYERS     = (1024, 768, 512)
MLP_SEED       = 42
MLP_ACTIVATION = 'relu'
MLP_ALPHA      = 0.001


def backup_old_models():
    """
    Copy ALL existing model files to a timestamped backup folder.
    Backs up a, b, c, d if they exist — old Quad ensemble is fully preserved.
    Never deletes anything.
    """
    model_files = (
        list(PIPELINE_DIR.glob("model_mlp_*.joblib")) +
        [PIPELINE_DIR / "scaler_knn.joblib",
         PIPELINE_DIR / "label_encoder.joblib",
         PIPELINE_DIR / "model_config.json"]
    )
    model_files = [f for f in model_files if f.exists()]

    if not model_files:
        print("  No existing models found — skipping backup.")
        return

    timestamp      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_backup_dir = PIPELINE_DIR / "backup" / f"model_backup_{timestamp}"
    run_backup_dir.mkdir(parents=True, exist_ok=True)

    for f in model_files:
        shutil.copy2(f, run_backup_dir / f.name)

    print(f"  Backed up {len(model_files)} files → {run_backup_dir.resolve()}")
    for f in model_files:
        print(f"    {f.name:<35} ({f.stat().st_size/1e6:.1f} MB)")


def load_training_data(use_cache=True):
    """Load features from cache or extract from augmented images."""
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}\n"
            f"Run: python augment_dataset.py first"
        )

    if use_cache and X_CACHE.exists() and Y_CACHE.exists():
        print(f"  Loading features from cache ...")
        print(f"  NOTE: Delete {X_CACHE.name} if you re-ran augment_dataset.py")
        X        = np.load(X_CACHE)
        y_labels = joblib.load(Y_CACHE)
        print(f"  Loaded {len(X):,} samples from cache")
        return X, y_labels

    with open(MANIFEST, encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    train_rows = [m for m in manifest if m['split'] == 'train']
    print(f"  Training images : {len(train_rows):,}")

    X, y_labels = [], []
    errors      = 0
    error_log   = []

    for i, row in enumerate(train_rows):
        if i % 10000 == 0:
            print(f"  [{i:>6}/{len(train_rows):>6}] extracting features ...")
        path = Path(row['path'])
        if not path.exists():
            errors += 1
            if len(error_log) < 10:
                error_log.append(f"    Missing : {path}")
            continue
        try:
            X.append(extract_features(Image.open(path)))
            y_labels.append(row['class_label'])
        except Exception as e:
            errors += 1
            if len(error_log) < 10:
                error_log.append(f"    Error   : {path} — {e}")

    if errors:
        print(f"  Warning: Skipped {errors} images")
        for line in error_log:
            print(line)
        if errors > 10:
            print(f"    ... and {errors - 10} more")

    X = np.array(X, dtype=np.float32)
    print(f"  Caching features → {X_CACHE} ...")
    np.save(X_CACHE, X)
    joblib.dump(y_labels, Y_CACHE)

    return X, y_labels


def main():
    print(f"{'='*60}")
    print(f"  TRAINING — Single MLP  {MLP_LAYERS}")
    print(f"{'='*60}")
    t_start = time.time()

    # ── [1/4] Backup ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Backing up existing models ...")
    backup_old_models()

    # ── [2/4] Load data ───────────────────────────────────────────────────────
    print(f"\n[2/4] Loading training data ...")
    t0 = time.time()
    X, y_labels = load_training_data()
    le           = LabelEncoder()
    y_int        = le.fit_transform(y_labels)
    n_classes    = len(le.classes_)
    print(f"  Feature size : {X.shape[1]}")
    print(f"  Classes      : {n_classes}")
    print(f"  Samples      : {len(X):,}")
    print(f"  Done in      : {(time.time()-t0)/60:.1f} min")

    # ── [3/4] Scale ───────────────────────────────────────────────────────────
    print(f"\n[3/4] Scaling features ...")
    t0       = time.time()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    del X
    print(f"  Done in : {(time.time()-t0):.1f}s")

    # ── [4/4] Train ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Training MLP  seed={MLP_SEED}  layers={MLP_LAYERS} ...")
    t0  = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=MLP_LAYERS,
        activation=MLP_ACTIVATION,
        solver='adam',
        alpha=MLP_ALPHA,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        tol=1e-4,
        random_state=MLP_SEED,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )
    mlp.fit(X_scaled, y_int)

    val_acc   = mlp.best_validation_score_ * 100
    train_acc = (mlp.predict(X_scaled) == y_int).mean() * 100
    elapsed   = (time.time() - t0) / 60
    arch      = '->'.join(['248'] + [str(l) for l in MLP_LAYERS] + [str(n_classes)])

    print(f"\n  Train accuracy : {train_acc:.2f}%")
    print(f"  Val accuracy   : {val_acc:.2f}%  <- real indicator")
    print(f"  Training time  : {elapsed:.1f} min")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n  Saving models ...")
    # Saved as model_mlp_a.joblib so app.py loads it without any changes
    joblib.dump(mlp,    PIPELINE_DIR / "model_mlp_a.joblib",   compress=3)
    joblib.dump(scaler, PIPELINE_DIR / "scaler_knn.joblib",    compress=3)
    joblib.dump(le,     PIPELINE_DIR / "label_encoder.joblib", compress=3)

    config = {
        'feature_size':         int(X_scaled.shape[1]),
        'swatch_size':          96,
        'model_type':           'SingleMLP_v1',
        'n_classes':            n_classes,
        'class_labels':         list(le.classes_),
        'architecture':         arch,
        'mlp_a_architecture':   arch,
        'seed':                 MLP_SEED,
        'activation':           MLP_ACTIVATION,
        'alpha':                MLP_ALPHA,
        'training_samples':     int(X_scaled.shape[0]),
        'mlp_a_train_accuracy': round(train_acc, 2),
        'mlp_a_val_accuracy':   round(val_acc, 2),
        # Keep ensemble_weights for app.py compatibility — only mlp_a used
        'ensemble_weights':     {'mlp_a': 1.0},
    }
    with open(PIPELINE_DIR / "model_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Architecture   : {arch}")
    print(f"  Train accuracy : {train_acc:.2f}%")
    print(f"  Val accuracy   : {val_acc:.2f}%")
    print(f"  Training time  : {elapsed:.1f} min")
    print(f"  Total time     : {(time.time()-t_start)/60:.1f} min")
    print(f"  Saved to       : {PIPELINE_DIR.resolve()}")
    print(f"  Old models in  : {(PIPELINE_DIR / 'backup').resolve()}")
    print(f"\n  Next step: python evaluate.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()