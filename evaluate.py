"""
FINE-TUNED STEP 4 — Evaluate KNN + RF + SVM Ensemble with 165 features
Uses the full 3-model ensemble (same as inference_server.py).
Fixes:
  - Uses real KNN+RF+SVM ensemble instead of SVM-only
  - Reports Delta-E on wrong predictions only (median was always 0.00)
  - Updated report labels
"""

import csv
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
    confusion_matrix, r2_score, classification_report
)
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
import warnings
warnings.filterwarnings('ignore')


# ── Feature extraction — imported from shared module ──────────────────────
from features import rgb_to_lab, extract_features


def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom  = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask   = denom > 1.0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def main():
    pipeline_dir = Path("abaca_pipeline")
    report_dir   = pipeline_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models (KNN + RF + SVM + Ridge) ...")
    required = ["model_knn.joblib", "model_rf.joblib", "model_svm.joblib",
                "scaler_knn.joblib", "scaler.joblib", "model_regressor.joblib",
                "label_encoder.joblib"]
    for fname in required:
        if not (pipeline_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing: {pipeline_dir/fname}\n"
                f"Run: python run_pipeline.py --train first"
            )
    knn        = joblib.load(pipeline_dir / "model_knn.joblib")
    scaler_knn = joblib.load(pipeline_dir / "scaler_knn.joblib")
    rf         = joblib.load(pipeline_dir / "model_rf.joblib")
    svm_pipe   = joblib.load(pipeline_dir / "model_svm.joblib")
    scaler     = joblib.load(pipeline_dir / "scaler.joblib")
    regressor  = joblib.load(pipeline_dir / "model_regressor.joblib")
    le         = joblib.load(pipeline_dir / "label_encoder.joblib")
    print(f"✅  Models loaded — {len(le.classes_)} RHS classes")

    with open(pipeline_dir / "augmented_manifest.csv") as f:
        manifest = list(csv.DictReader(f))
    test_rows = [m for m in manifest if m['split'] == 'test']
    print(f"Evaluating on {len(test_rows)} test images ...")

    X_test, y_true_codes, y_true_rgb = [], [], []
    for i, row in enumerate(test_rows):
        if i % 1000 == 0:
            print(f"  Extracting features [{i}/{len(test_rows)}] ...")
        path = Path(row['path'])
        if not path.exists():
            continue
        try:
            feat = extract_features(Image.open(path))
            X_test.append(feat)
            y_true_codes.append(row['rhs_code'])
            y_true_rgb.append([float(row['R']), float(row['G']), float(row['B'])])
        except Exception:
            pass

    X_test     = np.array(X_test,     dtype=np.float32)
    y_true_rgb = np.array(y_true_rgb, dtype=np.float32)
    y_true_int = le.transform(y_true_codes)

    # ── Full KNN + RF + SVM ensemble (matches inference_server.py) ───────────
    print("Running KNN + RF + SVM ensemble predictions ...")
    X_scaled_knn = scaler_knn.transform(X_test)   # KNN needs its own scaler
    knn_proba    = knn.predict_proba(X_scaled_knn)
    rf_proba     = rf.predict_proba(X_test)        # RF uses raw features
    svm_proba    = svm_pipe.predict_proba(X_test)  # SVM has internal scaler

    # Weighted ensemble: KNN=0.25, RF=0.35, SVM=0.40
    ensemble_proba = 0.25 * knn_proba + 0.35 * rf_proba + 0.40 * svm_proba
    y_pred_int     = np.argmax(ensemble_proba, axis=1)
    y_pred_codes   = le.inverse_transform(y_pred_int)
    ensemble_acc   = (y_pred_int == y_true_int).mean()

    # Individual model accuracies for comparison
    svm_acc = (np.argmax(svm_proba, axis=1) == y_true_int).mean()
    rf_acc  = (np.argmax(rf_proba,  axis=1) == y_true_int).mean()
    knn_acc = (np.argmax(knn_proba, axis=1) == y_true_int).mean()
    print(f"  KNN accuracy : {knn_acc*100:.2f}%")
    print(f"  RF  accuracy : {rf_acc*100:.2f}%")
    print(f"  SVM accuracy : {svm_acc*100:.2f}%")
    print(f"  Ensemble     : {ensemble_acc*100:.2f}%")

    # Top-3 accuracy
    top3_correct = 0
    for i, true_idx in enumerate(y_true_int):
        top3_preds = np.argsort(ensemble_proba[i])[-3:]
        if true_idx in top3_preds:
            top3_correct += 1
    top3_accuracy = top3_correct / len(y_true_int)

    # Classification metrics
    precision = precision_score(y_true_int, y_pred_int, average='macro', zero_division=0)
    recall    = recall_score(   y_true_int, y_pred_int, average='macro', zero_division=0)
    f1        = f1_score(       y_true_int, y_pred_int, average='macro', zero_division=0)

    # Delta-E
    colors_db = {}
    with open(pipeline_dir / "rhs_colors.csv") as f:
        for row in csv.DictReader(f):
            colors_db[row['rhs_code']] = (float(row['Lab_L']), float(row['Lab_a']), float(row['Lab_b']))
    delta_es      = []  # all predictions
    wrong_delta_es = [] # wrong predictions only (median was always 0.00 before this fix)
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

    # Regression
    X_scaled   = scaler.transform(X_test)
    y_pred_rgb = regressor.predict(X_scaled)
    mape_val   = smape(y_true_rgb, y_pred_rgb)
    r2_val     = r2_score(y_true_rgb, y_pred_rgb)

    # ── Confusion matrix ───────────────────────────────────────────────────
    present_classes = sorted(set(list(y_true_codes) + list(y_pred_codes)))
    present_idx     = [list(le.classes_).index(c) for c in present_classes]
    cm = confusion_matrix(y_true_int, y_pred_int, labels=present_idx)
    n  = len(present_classes)
    fig_size = max(14, n * 0.22)
    fig, ax  = plt.subplots(figsize=(min(fig_size, 44), min(fig_size, 44)))
    sns.heatmap(cm, ax=ax,
                xticklabels=present_classes, yticklabels=present_classes,
                cmap='Blues', linewidths=0.3,
                annot=(n <= 40), fmt='d' if n <= 40 else '',
                cbar_kws={'shrink': 0.6})
    ax.set_xlabel('Predicted RHS Code', fontsize=9)
    ax.set_ylabel('True RHS Code',      fontsize=9)
    ax.set_title('Confusion Matrix — KNN+RF+SVM Ensemble', fontsize=11, fontweight='bold')
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("   Confusion matrix saved.")

    # ── Delta-E chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(delta_es, bins=40, color='#2563EB', edgecolor='white', alpha=0.85)
    ax.axvline(5,  color='green',  linestyle='--', linewidth=1.5, label='ΔE=5  (imperceptible)')
    ax.axvline(10, color='orange', linestyle='--', linewidth=1.5, label='ΔE=10 (very close)')
    ax.axvline(20, color='red',    linestyle='--', linewidth=1.5, label='ΔE=20 (acceptable)')
    ax.set_xlabel('Delta-E Color Distance (lower = better)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Color Distance — True vs Predicted RHS Code', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(report_dir / "delta_e_distribution.png", dpi=150)
    plt.close()
    print("   Delta-E chart saved.")

    # ── Accuracy bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    models  = ['Old KNN\n(baseline)', 'Prev\nSVM', 'KNN\nonly',
               'RF\nonly', 'SVM\nonly', 'Ensemble\nKNN+RF+SVM', 'Top-3\nAccuracy']
    accs    = [17.0, 64.55, knn_acc*100, rf_acc*100,
               svm_acc*100, ensemble_acc*100, top3_accuracy*100]
    clrs    = ['#EF4444', '#F59E0B', '#60A5FA', '#34D399',
               '#A78BFA', '#10B981', '#3B82F6']
    bars    = ax.bar(models, accs, color=clrs, edgecolor='white', linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(0, min(100, max(accs) + 12))
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title('Model Accuracy — Individual vs Ensemble', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / "model_comparison.png", dpi=150)
    plt.close()
    print("   Model comparison chart saved.")

    # ── RGB scatter ────────────────────────────────────────────────────────
    channels  = ['R', 'G', 'B']
    colors_ch = ['#E74C3C', '#27AE60', '#2980B9']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (ch, col) in enumerate(zip(channels, colors_ch)):
        ax = axes[i]
        ax.scatter(y_true_rgb[:, i], y_pred_rgb[:, i],
                   alpha=0.35, s=12, color=col, edgecolors='none')
        mn = min(y_true_rgb[:, i].min(), y_pred_rgb[:, i].min())
        mx = max(y_true_rgb[:, i].max(), y_pred_rgb[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        ch_r2   = r2_score(y_true_rgb[:, i], y_pred_rgb[:, i])
        ch_mape = smape(y_true_rgb[:, i], y_pred_rgb[:, i])
        ax.set_title(f'{ch}  R²={ch_r2:.3f}  sMAPE={ch_mape:.1f}%', fontsize=9)
        ax.set_xlabel(f'True {ch}', fontsize=8)
        ax.set_ylabel(f'Predicted {ch}', fontsize=8)
    plt.suptitle('RGB Regression — True vs Predicted', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(report_dir / "rgb_regression_scatter.png", dpi=150)
    plt.close()
    print("   Regression scatter saved.")

    # ── Color comparison grid ──────────────────────────────────────────────
    sample_n  = min(80, len(y_true_codes))
    cols_grid = 10
    rows_grid = (sample_n + cols_grid - 1) // cols_grid
    fig, axes = plt.subplots(rows_grid * 2, cols_grid,
                             figsize=(cols_grid * 1.2, rows_grid * 2.4))
    for idx in range(sample_n):
        r_idx = idx // cols_grid
        c_idx = idx % cols_grid
        tr, tg, tb = [int(v) for v in y_true_rgb[idx]]
        pr, pg, pb = [int(np.clip(v, 0, 255)) for v in y_pred_rgb[idx]]
        correct = (y_pred_codes[idx] == y_true_codes[idx])
        for row_offset, (r, g, b), label in [
            (0, (tr, tg, tb), y_true_codes[idx]),
            (1, (pr, pg, pb), y_pred_codes[idx]),
        ]:
            ax = axes[r_idx * 2 + row_offset, c_idx]
            ax.set_facecolor(f'#{r:02x}{g:02x}{b:02x}')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=4, pad=1, color='green' if correct else 'red')
            for spine in ax.spines.values():
                spine.set_edgecolor('green' if correct else 'red')
                spine.set_linewidth(1.5 if not correct else 0.5)
    for idx in range(sample_n, rows_grid * cols_grid):
        r_idx = idx // cols_grid
        c_idx = idx % cols_grid
        for ro in [0, 1]:
            axes[r_idx * 2 + ro, c_idx].axis('off')
    plt.suptitle('True (top) vs Predicted (bottom) — green = correct',
                 fontsize=9, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(report_dir / "color_comparison_grid.png", dpi=150)
    plt.close()
    print("   Color comparison grid saved.")

    # ── Per-class metrics ──────────────────────────────────────────────────
    report_dict = classification_report(
        y_true_int, y_pred_int,
        target_names=le.classes_, output_dict=True, zero_division=0)
    pc_rows = []
    for code in le.classes_:
        if code in report_dict:
            m = report_dict[code]
            pc_rows.append({'rhs_code': code,
                            'precision': round(m['precision'], 4),
                            'recall':    round(m['recall'],    4),
                            'f1':        round(m['f1-score'],  4),
                            'support':   int(m['support'])})
    with open(report_dir / "per_class_metrics.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rhs_code','precision','recall','f1','support'])
        w.writeheader(); w.writerows(pc_rows)

    # ── Text report ────────────────────────────────────────────────────────
    confused = {}
    for tc, pc in zip(y_true_codes, y_pred_codes):
        if tc != pc:
            key = (tc, pc); confused[key] = confused.get(key, 0) + 1
    top_confused = sorted(confused.items(), key=lambda x: -x[1])[:10]

    report_lines = [
        "=" * 65,
        "  ABACA FIBER COLOR IDENTIFICATION — EVALUATION REPORT",
        "  Ensemble: KNN(0.25) + RF(0.35) + SVM(0.40) | 165 features",
        "=" * 65,
        "",
        "DATASET SUMMARY",
        f"  Test images evaluated : {len(X_test)}",
        f"  Unique RHS classes    : {len(le.classes_)}",
        f"  Feature size          : {X_test.shape[1]} per image",
        "",
        "CLASSIFICATION METRICS (Individual Models)",
        f"  KNN Accuracy  : {knn_acc:.4f}  ({knn_acc*100:.2f}%)",
        f"  RF  Accuracy  : {rf_acc:.4f}  ({rf_acc*100:.2f}%)",
        f"  SVM Accuracy  : {svm_acc:.4f}  ({svm_acc*100:.2f}%)",
        "",
        "CLASSIFICATION METRICS (Ensemble KNN+RF+SVM)",
        f"  Ensemble Accuracy : {ensemble_acc:.4f}  ({ensemble_acc*100:.2f}%)",
        f"  Top-3 Accuracy    : {top3_accuracy:.4f}  ({top3_accuracy*100:.2f}%)",
        f"  Precision         : {precision:.4f}",
        f"  Recall            : {recall:.4f}",
        f"  F1 Score          : {f1:.4f}",
        f"  Previous SVM only : 64.55%  (before ensemble)",
        f"  Old fake KNN      : ~17%    (was actually RF mislabeled)",
        "",
        "COLOR DISTANCE METRICS (Delta-E, lower is better)",
        f"  Mean Delta-E (all)           : {delta_es.mean():.2f}",
        f"  Mean Delta-E (wrong only)    : {wrong_delta_es.mean():.2f}",
        f"  Median Delta-E (wrong only)  : {np.median(wrong_delta_es):.2f}",
        f"  Note: Median was always 0.00 before — fixed to show wrong preds only",
        f"  Within dE < 5  (imperceptible) : {(delta_es<5).mean()*100:.1f}%",
        f"  Within dE < 10 (very close)    : {(delta_es<10).mean()*100:.1f}%",
        f"  Within dE < 20 (acceptable)    : {(delta_es<20).mean()*100:.1f}%",
        "",
        "REGRESSION METRICS (predicted RGB values)",
        f"  sMAPE : {mape_val:.2f}%",
        f"  R²    : {r2_val:.4f}",
        "",
        "TOP-10 MOST COMMON MISCLASSIFICATIONS",
    ]
    for (tc, pc), cnt in top_confused:
        report_lines.append(f"  True: {tc:8s}  Predicted: {pc:8s}  Count: {cnt}")
    report_lines += [
        "",
        "OUTPUT FILES",
        "  confusion_matrix.png       — class confusion heatmap",
        "  delta_e_distribution.png   — color distance histogram",
        "  model_comparison.png       — before vs after fine-tuning",
        "  rgb_regression_scatter.png — R/G/B true vs predicted",
        "  color_comparison_grid.png  — visual true/predicted swatches",
        "  per_class_metrics.csv      — per-RHS-code metrics",
        "=" * 65,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open(report_dir / "evaluation_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n✅  Report saved to {report_dir}/")


if __name__ == "__main__":
    main()