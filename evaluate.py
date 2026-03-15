"""
evaluate.py — Single MLP Evaluation (876 classes)

Updated for Single MLP:
- Only model_mlp_a.joblib required (b/c/d optional)
- Report shows single model accuracy + Top-3 + Top-5
- All charts and report files still generated in report/
- Backward compatible — still works if b/c/d exist (shows individual accuracies)
"""
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from features import extract_features

PIPELINE_DIR = Path("abaca_pipeline")
REPORT_DIR   = Path("report")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models ...")

    # mlp_a is required
    for fname in ["model_mlp_a.joblib", "scaler_knn.joblib", "label_encoder.joblib"]:
        if not (PIPELINE_DIR / fname).exists():
            raise FileNotFoundError(
                f"Missing: {PIPELINE_DIR/fname}\n"
                f"Run: python train_model.py first"
            )

    mlp_a  = joblib.load(PIPELINE_DIR / "model_mlp_a.joblib")
    scaler = joblib.load(PIPELINE_DIR / "scaler_knn.joblib")
    le     = joblib.load(PIPELINE_DIR / "label_encoder.joblib")

    # mlp_b/c/d optional — load if present
    def _load_opt(name):
        p = PIPELINE_DIR / name
        return joblib.load(p) if p.exists() else None

    mlp_b = _load_opt("model_mlp_b.joblib")
    mlp_c = _load_opt("model_mlp_c.joblib")
    mlp_d = _load_opt("model_mlp_d.joblib")

    # Detect mode
    quad_mode = all([mlp_b, mlp_c, mlp_d])
    mode_label = "Quad MLP Ensemble" if quad_mode else "Single MLP"
    print(f"  Mode     : {mode_label}")
    print(f"  Classes  : {len(le.classes_)}")

    # Load weights from config
    config_path = PIPELINE_DIR / "model_config.json"
    WEIGHT_A = WEIGHT_B = WEIGHT_C = WEIGHT_D = 0.25
    model_type = "SingleMLP_v1"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        w = cfg.get('ensemble_weights', {})
        WEIGHT_A = w.get('mlp_a', 1.0 if not quad_mode else 0.28)
        WEIGHT_B = w.get('mlp_b', 0.27)
        WEIGHT_C = w.get('mlp_c', 0.22)
        WEIGHT_D = w.get('mlp_d', 0.23)
        model_type = cfg.get('model_type', model_type)
        arch = cfg.get('mlp_a_architecture', 'unknown')
        print(f"  Architecture: {arch}")

    # ── Load test data ────────────────────────────────────────────────────────
    manifest_path = PIPELINE_DIR / "augmented_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("augmented_manifest.csv not found. Run augment_dataset.py first.")

    with open(manifest_path, encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))

    test_rows = [m for m in manifest if m['split'] == 'test']
    print(f"\nEvaluating on {len(test_rows)} test images ...")

    X_test, y_true_codes = [], []
    for i, row in enumerate(test_rows):
        if i % 1000 == 0:
            print(f"  Extracting features [{i}/{len(test_rows)}] ...")
        path = Path(row['path'])
        if not path.exists():
            continue
        try:
            feat = extract_features(Image.open(path))
            X_test.append(feat)
            y_true_codes.append(row['class_label'])
        except Exception:
            pass

    X_test     = np.array(X_test, dtype=np.float32)
    y_true_int = le.transform(y_true_codes)
    X_scaled   = scaler.transform(X_test)
    print(f"  Loaded {len(X_test)} test samples")

    # ── Predictions ───────────────────────────────────────────────────────────
    print("\nRunning predictions ...")
    proba_a = mlp_a.predict_proba(X_scaled)
    acc_a   = (np.argmax(proba_a, axis=1) == y_true_int).mean()
    print(f"  MLP-A accuracy : {acc_a*100:.2f}%")

    if quad_mode:
        proba_b = mlp_b.predict_proba(X_scaled)
        proba_c = mlp_c.predict_proba(X_scaled)
        proba_d = mlp_d.predict_proba(X_scaled)
        acc_b   = (np.argmax(proba_b, axis=1) == y_true_int).mean()
        acc_c   = (np.argmax(proba_c, axis=1) == y_true_int).mean()
        acc_d   = (np.argmax(proba_d, axis=1) == y_true_int).mean()
        ensemble_proba = (WEIGHT_A * proba_a + WEIGHT_B * proba_b +
                          WEIGHT_C * proba_c + WEIGHT_D * proba_d)
        print(f"  MLP-B accuracy : {acc_b*100:.2f}%")
        print(f"  MLP-C accuracy : {acc_c*100:.2f}%")
        print(f"  MLP-D accuracy : {acc_d*100:.2f}%")
    else:
        ensemble_proba = proba_a
        acc_b = acc_c = acc_d = None

    y_pred_int   = np.argmax(ensemble_proba, axis=1)
    y_pred_codes = le.inverse_transform(y_pred_int)
    ensemble_acc = (y_pred_int == y_true_int).mean()
    print(f"  Final accuracy : {ensemble_acc*100:.2f}%")

    # Top-3 / Top-5
    top3_correct = sum(
        1 for i, ti in enumerate(y_true_int)
        if ti in np.argsort(ensemble_proba[i])[-3:]
    )
    top5_correct = sum(
        1 for i, ti in enumerate(y_true_int)
        if ti in np.argsort(ensemble_proba[i])[-5:]
    )
    top3_acc = top3_correct / len(y_true_int)
    top5_acc = top5_correct / len(y_true_int)
    print(f"  Top-3 accuracy : {top3_acc*100:.2f}%")
    print(f"  Top-5 accuracy : {top5_acc*100:.2f}%")

    # Classification metrics
    precision = precision_score(y_true_int, y_pred_int, average='macro', zero_division=0)
    recall    = recall_score(   y_true_int, y_pred_int, average='macro', zero_division=0)
    f1        = f1_score(       y_true_int, y_pred_int, average='macro', zero_division=0)

    # ── Delta-E ───────────────────────────────────────────────────────────────
    colors_db = {}
    with open(PIPELINE_DIR / "rhs_colors.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            colors_db[row['class_label']] = (
                float(row['Lab_L']), float(row['Lab_a']), float(row['Lab_b'])
            )

    delta_es, wrong_delta_es = [], []
    for tc, pc in zip(y_true_codes, y_pred_codes):
        if tc in colors_db and pc in colors_db:
            tL, ta, tb = colors_db[tc]
            pL, pa, pb = colors_db[pc]
            de = np.sqrt((tL-pL)**2 + (ta-pa)**2 + (tb-pb)**2)
            delta_es.append(de)
            if tc != pc:
                wrong_delta_es.append(de)

    delta_es       = np.array(delta_es)
    wrong_delta_es = np.array(wrong_delta_es) if wrong_delta_es else np.array([0.0])

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nGenerating charts ...")

    # 1. Confusion matrix
    present_classes = sorted(set(list(y_true_codes) + list(y_pred_codes)))
    present_idx     = [list(le.classes_).index(c) for c in present_classes]
    cm       = confusion_matrix(y_true_int, y_pred_int, labels=present_idx)
    n        = len(present_classes)
    fig_size = max(14, n * 0.22)
    fig, ax  = plt.subplots(figsize=(min(fig_size, 44), min(fig_size, 44)))
    sns.heatmap(cm, ax=ax,
                xticklabels=present_classes, yticklabels=present_classes,
                cmap='Blues', linewidths=0.3,
                annot=(n <= 40), fmt='d' if n <= 40 else '',
                cbar_kws={'shrink': 0.6})
    ax.set_xlabel('Predicted Class', fontsize=9)
    ax.set_ylabel('True Class',      fontsize=9)
    ax.set_title(f'Confusion Matrix — {mode_label} ({len(le.classes_)} classes)',
                 fontsize=11, fontweight='bold')
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("  confusion_matrix.png")

    # 2. Delta-E distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(delta_es, bins=40, color='#2563EB', edgecolor='white', alpha=0.85)
    ax.axvline(5,  color='green',  linestyle='--', linewidth=1.5, label='dE=5')
    ax.axvline(10, color='orange', linestyle='--', linewidth=1.5, label='dE=10')
    ax.axvline(20, color='red',    linestyle='--', linewidth=1.5, label='dE=20')
    ax.set_xlabel('Delta-E Color Distance (lower = better)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Delta-E Distribution — {mode_label}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "delta_e_distribution.png", dpi=150)
    plt.close()
    print("  delta_e_distribution.png")

    # 3. Model accuracy chart
    if quad_mode:
        models = ['MLP-A', 'MLP-B', 'MLP-C', 'MLP-D', 'Ensemble', 'Top-3', 'Top-5']
        accs   = [acc_a*100, acc_b*100, acc_c*100, acc_d*100,
                  ensemble_acc*100, top3_acc*100, top5_acc*100]
        clrs   = ['#60A5FA', '#818CF8', '#A78BFA', '#C084FC',
                  '#10B981', '#3B82F6', '#06B6D4']
        title  = 'Quad MLP Ensemble — Model Accuracy'
    else:
        models = ['MLP-A', 'Top-3', 'Top-5']
        accs   = [acc_a*100, top3_acc*100, top5_acc*100]
        clrs   = ['#10B981', '#3B82F6', '#06B6D4']
        title  = 'Single MLP — Model Accuracy'

    fig, ax = plt.subplots(figsize=(max(8, len(models)*2), 4))
    bars = ax.bar(models, accs, color=clrs, edgecolor='white', linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(90, color='red', linestyle='--', linewidth=1.5, label='Target 90%')
    ax.set_ylim(0, min(100, max(accs) + 15))
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print("  model_comparison.png")

    # 4. Color comparison grid
    rgb_db = {}
    with open(PIPELINE_DIR / "rhs_colors.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rgb_db[row['class_label']] = (int(row['R']), int(row['G']), int(row['B']))

    sample_n  = min(80, len(y_true_codes))
    cols_grid = 10
    rows_grid = (sample_n + cols_grid - 1) // cols_grid
    fig, axes = plt.subplots(rows_grid * 2, cols_grid,
                             figsize=(cols_grid * 1.2, rows_grid * 2.4))

    for idx in range(sample_n):
        r_idx     = idx // cols_grid
        c_idx     = idx % cols_grid
        true_code = y_true_codes[idx]
        pred_code = y_pred_codes[idx]
        correct   = (true_code == pred_code)
        tr, tg, tb = rgb_db.get(true_code, (128, 128, 128))
        pr, pg, pb = rgb_db.get(pred_code, (128, 128, 128))
        for row_offset, (r, g, b), label in [
            (0, (tr, tg, tb), true_code),
            (1, (pr, pg, pb), pred_code),
        ]:
            ax = axes[r_idx * 2 + row_offset, c_idx]
            ax.set_facecolor(f'#{r:02x}{g:02x}{b:02x}')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=4, pad=1,
                         color='green' if correct else 'red')
            for spine in ax.spines.values():
                spine.set_edgecolor('green' if correct else 'red')
                spine.set_linewidth(1.5 if not correct else 0.5)

    for idx in range(sample_n, rows_grid * cols_grid):
        r_idx = idx // cols_grid
        c_idx = idx % cols_grid
        for ro in [0, 1]:
            axes[r_idx * 2 + ro, c_idx].axis('off')

    plt.suptitle('True (top) vs Predicted (bottom) — green=correct, red=wrong',
                 fontsize=9, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(REPORT_DIR / "color_comparison_grid.png", dpi=150)
    plt.close()
    print("  color_comparison_grid.png")

    # 5. Per-class metrics CSV
    report_dict = classification_report(
        y_true_int, y_pred_int,
        target_names=le.classes_, output_dict=True, zero_division=0
    )
    pc_rows = []
    for code in le.classes_:
        if code in report_dict:
            m = report_dict[code]
            pc_rows.append({
                'class_label': code,
                'precision':   round(m['precision'], 4),
                'recall':      round(m['recall'],    4),
                'f1':          round(m['f1-score'],  4),
                'support':     int(m['support'])
            })
    with open(REPORT_DIR / "per_class_metrics.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['class_label','precision','recall','f1','support'])
        w.writeheader()
        w.writerows(pc_rows)
    print("  per_class_metrics.csv")

    # ── Text report ───────────────────────────────────────────────────────────
    confused = {}
    for tc, pc in zip(y_true_codes, y_pred_codes):
        if tc != pc:
            key = (tc, pc)
            confused[key] = confused.get(key, 0) + 1
    top_confused = sorted(confused.items(), key=lambda x: -x[1])[:15]

    w5  = (delta_es <  5).mean() * 100
    w10 = (delta_es < 10).mean() * 100
    w20 = (delta_es < 20).mean() * 100
    sep = "=" * 66

    # Build accuracy block depending on mode
    if quad_mode:
        acc_lines = [
            f"  MLP-A (individual)     : {acc_a*100:.2f}%",
            f"  MLP-B (individual)     : {acc_b*100:.2f}%",
            f"  MLP-C (individual)     : {acc_c*100:.2f}%",
            f"  MLP-D (individual)     : {acc_d*100:.2f}%",
            f"  Ensemble (A+B+C+D)     : {ensemble_acc*100:.2f}%",
        ]
        ensemble_label = f"Ensemble: MLP-A({WEIGHT_A}) + MLP-B({WEIGHT_B}) + MLP-C({WEIGHT_C}) + MLP-D({WEIGHT_D})"
    else:
        acc_lines = [
            f"  MLP-A accuracy         : {acc_a*100:.2f}%",
        ]
        ensemble_label = f"Model: Single MLP  Architecture: {cfg.get('mlp_a_architecture','248->1024->768->512->N')}"

    report_lines = [
        sep,
        "  ABACA FIBER COLOR IDENTIFICATION — EVALUATION REPORT",
        f"  {ensemble_label}",
        f"  Features: {X_test.shape[1]} dims  |  Classes: {len(le.classes_)}",
        sep,
        "",
        "DATASET SUMMARY",
        f"  Test images evaluated  : {len(X_test)}",
        f"  Unique RHS classes     : {len(le.classes_)}",
        f"  Feature size           : {X_test.shape[1]} per image",
        "",
        sep,
        "ACCURACY",
        sep,
    ] + acc_lines + [
        f"  Top-3 Accuracy         : {top3_acc*100:.2f}%",
        f"  Top-5 Accuracy         : {top5_acc*100:.2f}%",
        f"  Goal (Top-5 >= 90%)    : {'MET ✅' if top5_acc*100 >= 90 else 'NOT MET ❌'} ({top5_acc*100:.2f}%)",
        "",
        sep,
        f"CLASSIFICATION METRICS  (macro-averaged across all {len(le.classes_)} classes)",
        sep,
        f"  Precision              : {precision:.4f}  ({precision*100:.2f}%)",
        f"  Recall                 : {recall:.4f}  ({recall*100:.2f}%)",
        f"  F1 Score               : {f1:.4f}  ({f1*100:.2f}%)",
        "",
        sep,
        "DELTA-E  (perceptual color distance, lower = better)",
        sep,
        f"  Mean  dE  (all)        : {delta_es.mean():.2f}",
        f"  Median dE (all)        : {np.median(delta_es):.2f}",
        f"  Mean  dE  (wrong only) : {wrong_delta_es.mean():.2f}",
        f"  Median dE (wrong only) : {np.median(wrong_delta_es):.2f}",
        f"  Max   dE               : {delta_es.max():.2f}",
        "",
        f"  Within dE <  5  (imperceptible) : {w5:.1f}%",
        f"  Within dE < 10  (very close)    : {w10:.1f}%",
        f"  Within dE < 20  (acceptable)    : {w20:.1f}%",
        "",
        sep,
        "TOP 15 MOST COMMON MISCLASSIFICATIONS",
        sep,
    ]

    for (tc, pc), cnt in top_confused:
        de_pair = 0.0
        if tc in colors_db and pc in colors_db:
            tL, ta, tb = colors_db[tc]
            pL, pa, pb = colors_db[pc]
            de_pair = np.sqrt((tL-pL)**2 + (ta-pa)**2 + (tb-pb)**2)
        report_lines.append(
            f"  True: {tc:<22} Predicted: {pc:<22} Count: {cnt:>3}   dE: {de_pair:.1f}"
        )

    report_lines += [
        "",
        sep,
        "OUTPUT FILES  (all saved to report/)",
        sep,
        "  evaluation_report.txt",
        "  confusion_matrix.png",
        "  delta_e_distribution.png",
        "  model_comparison.png",
        "  color_comparison_grid.png",
        "  per_class_metrics.csv",
        sep,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(REPORT_DIR / "evaluation_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved → {REPORT_DIR}/evaluation_report.txt")
    print(f"Charts saved → {REPORT_DIR}/")


if __name__ == "__main__":
    main()