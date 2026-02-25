"""
ABACA COLOR SCANNER — CPU Edition
===================================
What changed from the previous version:
  1. GrabCut segmentation (new segment.py) runs before color extraction
     → removes background/hands/shadows from the crop automatically
     → ~0.5s on CPU, no GPU needed

  2. Dominant color now explicitly excludes masked-out gray pixels
     → K-means sees only real fiber pixels

  3. White balance capped at 1.3x (was 1.8x)
     → preserves true fiber hue, less over-correction

  4. Your existing KNN + SVM models are kept as-is
     → No retraining needed
     → Same feature extraction pipeline as before

  5. Segmentation result shown in UI
     → Green badge: "Fiber isolated" + coverage %
     → Orange badge: "Fallback mode" if GrabCut wasn't confident

  6. Photo guide screen, live color preview, box quality bars,
     minimum size warning — all from previous version kept.

Files needed (unchanged):
  abaca_pipeline/model_knn.joblib
  abaca_pipeline/model_svm.joblib
  abaca_pipeline/label_encoder.joblib
  abaca_pipeline/rhs_colors.csv
  segment.py  ← new file (same folder as this script)

Install new dependency:
  pip install opencv-python-headless
"""

import io, json, socket, warnings
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
warnings.filterwarnings('ignore')

from skimage.feature import local_binary_pattern
from skimage.filters import gabor

PIPELINE_DIR = Path("abaca_pipeline")

# ── Security settings ──────────────────────────────────────────────────────────
MAX_IMAGE_BYTES   = 10_000_000   # 10 MB — reject oversized uploads
MAX_REQUESTS_PER_MIN = 30        # per IP — basic rate limiting
RATE_WINDOW_SECS  = 60

# Simple in-memory rate limiter {ip: [timestamp, ...]}
import time, collections
_rate_log = collections.defaultdict(list)

def _check_rate_limit(ip: str) -> bool:
    """Returns True if request is allowed, False if rate limit exceeded."""
    now   = time.time()
    calls = _rate_log[ip]
    # Remove calls older than the window
    _rate_log[ip] = [t for t in calls if now - t < RATE_WINDOW_SECS]
    if len(_rate_log[ip]) >= MAX_REQUESTS_PER_MIN:
        return False
    _rate_log[ip].append(now)
    return True


# ── Load models ────────────────────────────────────────────────────────────────
def load_models():
    import csv

    # Check all required files exist before loading
    required = [
        "model_knn.joblib", "model_svm.joblib",
        "scaler_knn.joblib", "label_encoder.joblib", "rhs_colors.csv"
    ]
    for fname in required:
        fpath = PIPELINE_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"\n\u274c  Missing: {fpath}\n"
                f"   Run: python run_pipeline.py --train  (to retrain models)"
            )

    knn        = joblib.load(PIPELINE_DIR / "model_knn.joblib")   # real KNN k=7
    scaler_knn = joblib.load(PIPELINE_DIR / "scaler_knn.joblib")  # KNN needs its own scaler
    svm        = joblib.load(PIPELINE_DIR / "model_svm.joblib")   # SVM RBF
    le         = joblib.load(PIPELINE_DIR / "label_encoder.joblib")

    colors_db = {}
    with open(PIPELINE_DIR / "rhs_colors.csv") as f:
        for row in csv.DictReader(f):
            colors_db[row["rhs_code"]] = {
                "L": float(row["Lab_L"]),
                "a": float(row["Lab_a"]),
                "b": float(row["Lab_b"]),
                "R": int(row["R"]),
                "G": int(row["G"]),
                "B": int(row["B"]),
            }

    print(f"✅ KNN (k=7) + SVM (RBF C=20) loaded")
    print(f"✅ {len(le.classes_)} RHS classes")
    print(f"✅ {len(colors_db)} colors in database")
    return knn, scaler_knn, svm, le, colors_db


# ── Color math ─────────────────────────────────────────────────────────────────
# ── Feature extraction — imported from shared module ──────────────────────
from features import rgb_to_lab, extract_features


# ── Missing color utility functions ───────────────────────────────────────────
def auto_white_balance(img: Image.Image, max_scale: float = 1.3) -> Image.Image:
    rgb = np.array(img.convert('RGB'), dtype=np.float32)
    r_mean = float(rgb[:, :, 0].mean())
    g_mean = float(rgb[:, :, 1].mean())
    b_mean = float(rgb[:, :, 2].mean())
    gray_mean = (r_mean + g_mean + b_mean) / 3.0
    if gray_mean < 1e-6:
        return img.convert('RGB')
    sr = float(np.clip(gray_mean / (r_mean + 1e-6), 1.0 / max_scale, max_scale))
    sg = float(np.clip(gray_mean / (g_mean + 1e-6), 1.0 / max_scale, max_scale))
    sb = float(np.clip(gray_mean / (b_mean + 1e-6), 1.0 / max_scale, max_scale))
    if max(abs(sr - 1.0), abs(sg - 1.0), abs(sb - 1.0)) < 0.05:
        return img.convert('RGB')
    ch_spread = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
    if ch_spread > 60:
        return img.convert('RGB')
    balanced = rgb.copy()
    balanced[:, :, 0] = np.clip(rgb[:, :, 0] * sr, 0, 255)
    balanced[:, :, 1] = np.clip(rgb[:, :, 1] * sg, 0, 255)
    balanced[:, :, 2] = np.clip(rgb[:, :, 2] * sb, 0, 255)
    return Image.fromarray(balanced.astype(np.uint8))

def extract_dominant_color(img: Image.Image, n_clusters: int = 5) -> tuple:
    from sklearn.cluster import KMeans
    rgb = np.array(img.convert('RGB'), dtype=np.float32)
    pixels = rgb.reshape(-1, 3)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    brightness = (r + g + b) / 3.0

    # Exclude GrabCut background fill (128,128,128 ± 30)
    gray_mask  = (np.abs(r - 128) < 30) & (np.abs(g - 128) < 30) & (np.abs(b - 128) < 30)
    # Exclude pure white / near-white background
    white_mask = (r > 230) & (g > 230) & (b > 230)
    # Exclude dark holes / punch-holes in color cards (brightness < 40)
    black_mask = brightness < 40
    # Exclude glare/specular highlights (brightness > 210)
    glare_mask = brightness > 210

    valid = pixels[~(gray_mask | white_mask | black_mask | glare_mask)]
    if len(valid) < 50:
        # Fallback: only drop gray background and pure black/white
        valid = pixels[~(gray_mask | (brightness < 20) | (brightness > 245))]
    if len(valid) < 50:
        valid = pixels
    if len(valid) > 5000:
        idx = np.random.choice(len(valid), 5000, replace=False)
        valid = valid[idx]
    k = min(n_clusters, len(valid))
    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(valid)
    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[np.argmax(counts)]
    return int(round(dominant[0])), int(round(dominant[1])), int(round(dominant[2]))

def delta_e(L1, a1, b1, L2, a2, b2) -> float:
    return float(np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2))

def de_label(de: float) -> tuple:
    if de < 1.0:    return "Imperceptible",   "#4ade80"
    elif de < 2.0:  return "Very close",      "#86efac"
    elif de < 3.5:  return "Close match",     "#fbbf24"
    elif de < 5.0:  return "Moderate diff",   "#f97316"
    elif de < 10.0: return "Noticeable diff", "#ef4444"
    else:           return "Very different",  "#dc2626"


# ── KNN + SVM ensemble ─────────────────────────────────────────────────────────
def ensemble_predict(feat, knn, scaler_knn, svm, le, n_top=15):
    """
    2-model ensemble: KNN + SVM (RF removed to reduce memory usage).
    KNN needs its own scaler (scaler_knn) — it was trained on scaled features.
    SVM uses its internal scaler (built into the Pipeline).
    Weights: KNN=0.40, SVM=0.60
    Delta-E still handles 85% of final score — ML impact unchanged.
    """
    f_raw    = feat.reshape(1, -1)
    f_scaled = scaler_knn.transform(f_raw)  # KNN needs scaled input

    knn_proba = knn.predict_proba(f_scaled)[0]   # real KNN k=7
    svm_proba = svm.predict_proba(f_raw)[0]       # SVM Pipeline has internal scaler

    combined = 0.40 * knn_proba + 0.60 * svm_proba
    top_idx  = np.argsort(combined)[::-1][:n_top]
    return le.inverse_transform(top_idx), combined[top_idx]


# ── Pure Delta-E matching ──────────────────────────────────────────────────────
def deltae_match(L, a, b, colors_db, n=10):
    results = [(code,
                delta_e(L, a, b, c['L'], c['a'], c['b']),
                f"#{c['R']:02x}{c['G']:02x}{c['B']:02x}")
               for code, c in colors_db.items()]
    return sorted(results, key=lambda x: x[1])[:n]


# ── Hybrid scoring ─────────────────────────────────────────────────────────────
def hybrid_score(ml_codes, ml_probs, dL, da, db, colors_db, de_cands):
    """
    Hybrid score = ML weight * ml_score + DE weight * de_score

    Since KNN/SVM have no abaca-specific training images yet, their
    probabilities are unreliable. Delta-E is the ground truth for color
    distance so we weight it much higher: 15% ML + 85% ΔE.

    When training images are added later, raise ML weight back toward 35%.
    """
    cands    = {}
    max_prob = max(ml_probs) if len(ml_probs) > 0 else 1.0
    for code, prob in zip(ml_codes, ml_probs):
        c = colors_db.get(code)
        if not c: continue
        de = delta_e(dL, da, db, c['L'], c['a'], c['b'])
        cands[code] = {'ml': float(prob)/(max_prob+1e-9), 'de': de,
                       'hex': f"#{c['R']:02x}{c['G']:02x}{c['B']:02x}"}
    for code, de, hx in de_cands[:10]:
        if code not in cands:
            cands[code] = {'ml': 0.0, 'de': de, 'hex': hx}
        else:
            cands[code]['de'] = de
    max_de = max(v['de'] for v in cands.values()) if cands else 50.0
    for code in cands:
        de_score = 1.0 - cands[code]['de'] / (max_de + 1e-9)
        # 15% ML + 85% ΔE — ΔE dominates since ML has no fiber training yet
        cands[code]['hybrid'] = 0.15 * cands[code]['ml'] + 0.85 * de_score
    return sorted(cands.items(), key=lambda x: x[1]['hybrid'], reverse=True)


# ── Full prediction pipeline ───────────────────────────────────────────────────
def predict(img: Image.Image, knn, scaler_knn, svm, le, colors_db):
    # 1. GrabCut segmentation — isolate fiber pixels
    try:
        from segment import segment_fiber
        masked_img, seg_found, seg_coverage, seg_mask = segment_fiber(img)
    except Exception:
        masked_img   = img.convert('RGB')
        seg_found    = False
        seg_coverage = 1.0

    if seg_found:
        # GrabCut succeeded — use the masked image (background = gray 128)
        work_img = masked_img
    else:
        # GrabCut failed — use center 60% crop instead of full raw image.
        # The center crop avoids hands, card edges, and cluttered backgrounds
        # that pollute color extraction when segmentation cannot isolate fiber.
        raw = img.convert('RGB')
        w, h = raw.size
        cx, cy = w // 2, h // 2
        cw, ch = int(w * 0.60), int(h * 0.60)
        left   = max(0, cx - cw // 2)
        top    = max(0, cy - ch // 2)
        right  = min(w, cx + cw // 2)
        bottom = min(h, cy + ch // 2)
        work_img = raw.crop((left, top, right, bottom))
        print(f"  ⚠️  GrabCut failed — using center 60% crop fallback "
              f"({work_img.width}×{work_img.height}px)")

    # 2. White balance (skipped for saturated colors)
    wb_img = auto_white_balance(work_img)
    # Check if WB actually changed anything (saturation-aware WB may skip)
    arr_before = np.array(work_img.convert('RGB'), dtype=np.float32)
    arr_after  = np.array(wb_img, dtype=np.float32)
    wb_applied = bool(np.abs(arr_before - arr_after).mean() > 0.5)

    # 3. Dominant color (ignores masked background)
    dom_r, dom_g, dom_b     = extract_dominant_color(wb_img)
    dom_L, dom_a, dom_b_val = rgb_to_lab(dom_r, dom_g, dom_b)

    # 4. Feature extraction + KNN/SVM ensemble
    feat                    = extract_features(wb_img)
    ml_codes, ml_probs      = ensemble_predict(feat, knn, scaler_knn, svm, le, n_top=15)

    # 5. Pure Delta-E candidates
    de_cands = deltae_match(dom_L, dom_a, dom_b_val, colors_db, n=10)

    # 6. Hybrid re-ranking
    ranked = hybrid_score(ml_codes, ml_probs, dom_L, dom_a, dom_b_val,
                          colors_db, de_cands)

    # 7. Build top 5 with ΔE gap info
    top5 = []
    for code, info in ranked[:5]:
        c = colors_db.get(code, {})
        if not c: continue
        de = info['de']
        dl, dc = de_label(de)
        top5.append({
            'rhs_code':   code,
            'match_score': round(info['hybrid'] * 100, 1),
            'delta_e':    round(de, 2),
            'de_label':   dl,
            'de_color':   dc,
            'hex':        info['hex'],
        })

    # Also compute ΔE for all codes in the 59A/60A range so user
    # knows how close the runner-up was
    pdb     = de_cands[0] if de_cands else (None, 99, '#888')
    pdb2    = de_cands[1] if len(de_cands) > 1 else (None, 99, '#888')
    best    = top5[0] if top5 else {}
    matched = colors_db.get(best.get('rhs_code', ''), {})

    return {
        'rhs_code':     best.get('rhs_code', '?'),
        'match_score':  best.get('match_score', 0),
        'delta_e':      best.get('delta_e', 0),
        'de_label':     best.get('de_label', ''),
        'de_color':     best.get('de_color', '#fff'),
        'pure_de_code': pdb[0],
        'pure_de_val':  round(pdb[1], 2),
        'pure_de_2nd':  pdb2[0],
        'pure_de_2nd_val': round(pdb2[1], 2),
        'dominant_hex': f'#{dom_r:02x}{dom_g:02x}{dom_b:02x}',
        'dominant_rgb': {'R': dom_r, 'G': dom_g, 'B': dom_b},
        'dominant_lab': {'L': round(dom_L,2), 'a': round(dom_a,2), 'b': round(dom_b_val,2)},
        'matched_hex':  (f"#{matched['R']:02x}{matched['G']:02x}{matched['B']:02x}"
                         if matched else '#888'),
        'wb_applied':   wb_applied,
        'seg_found':    seg_found,
        'seg_coverage': round(seg_coverage * 100, 1),
        'top_5': top5,
    }


# ── HTML UI ────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Abaca Scanner</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0f0f0f;--card:#1a1a1a;--card2:#222;--border:#2e2e2e;
  --text:#f0f0f0;--sub:#777;--accent:#d4a853;--green:#4ade80;--r:16px;
  --warn:#fbbf24;--danger:#ef4444;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent;}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',system-ui,sans-serif;
  min-height:100vh;padding-bottom:60px;}
.hdr{padding:14px 18px 0;display:flex;align-items:center;gap:12px;}
.logo{width:42px;height:42px;background:linear-gradient(135deg,#d4a853,#6b4c10);
  border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-size:22px;flex-shrink:0;}
.hdr h1{font-size:.95rem;letter-spacing:.06em;color:var(--accent);font-weight:700;}
.hdr p{font-size:.68rem;color:var(--sub);}
.pad{padding:0 18px;}
.btn{flex:1;padding:13px;border-radius:10px;border:1px solid var(--border);
  background:var(--card2);color:var(--text);font-size:.88rem;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:6px;font-weight:500;
  transition:opacity .15s;}
.btn:active{opacity:.75;}
.btn.primary{background:var(--accent);color:#000;border-color:var(--accent);font-weight:700;}
.btn.green{background:#166534;color:#4ade80;border-color:#166534;font-weight:700;}
.btn.danger{background:#7f1d1d;color:#ef4444;border-color:#7f1d1d;}
.btn-row{display:flex;gap:10px;}
#file-cam,#file-upload{display:none;}

/* Guide */
#guide-screen{padding:14px 18px 0;}
.guide-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  overflow:hidden;margin-bottom:10px;}
.guide-title{padding:10px 14px;font-size:.75rem;font-weight:700;letter-spacing:.05em;
  text-transform:uppercase;color:var(--accent);border-bottom:1px solid var(--border);}
.guide-steps{padding:12px 14px;display:flex;flex-direction:column;gap:10px;}
.gs{display:flex;gap:12px;align-items:flex-start;}
.gs-num{width:26px;height:26px;border-radius:50%;background:var(--accent);color:#000;
  font-size:.75rem;font-weight:700;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;margin-top:1px;}
.gs-text{font-size:.82rem;line-height:1.5;}
.gs-text strong{color:var(--accent);} .gs-text em{color:var(--warn);font-style:normal;}
.do-dont{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:0 14px 12px;}
.dd-block{border-radius:10px;padding:10px;}
.dd-block.do{background:#052e16;border:1px solid #166534;}
.dd-block.dont{background:#1f0606;border:1px solid #7f1d1d;}
.dd-title{font-size:.7rem;font-weight:700;text-transform:uppercase;margin-bottom:6px;}
.dd-block.do .dd-title{color:#4ade80;} .dd-block.dont .dd-title{color:#ef4444;}
.dd-item{font-size:.75rem;color:var(--sub);margin-bottom:4px;}
.dd-item:last-child{margin:0;}

/* Preview */
#prev-section{display:none;padding:14px 18px 0;}
.preview-wrap{position:relative;width:100%;border-radius:var(--r);overflow:hidden;
  border:1px solid var(--border);background:#000;touch-action:none;}
#preview{width:100%;display:block;max-height:340px;object-fit:contain;}
#focus-box{position:absolute;border:2.5px solid #00ff88;border-radius:6px;
  box-shadow:0 0 0 9999px rgba(0,0,0,.5);cursor:move;touch-action:none;}
.corner{position:absolute;width:18px;height:18px;background:#00ff88;
  border-radius:3px;z-index:2;}
.corner.tl{top:-5px;left:-5px;cursor:nw-resize;}
.corner.tr{top:-5px;right:-5px;cursor:ne-resize;}
.corner.bl{bottom:-5px;left:-5px;cursor:sw-resize;}
.corner.br{bottom:-5px;right:-5px;cursor:se-resize;}
#focus-dot{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:10px;height:10px;border-radius:50%;background:#00ff88;pointer-events:none;}
#live-bar{position:absolute;bottom:0;left:0;right:0;padding:5px 10px;
  display:flex;align-items:center;gap:8px;background:rgba(0,0,0,.65);pointer-events:none;}
#live-sw{width:18px;height:18px;border-radius:4px;flex-shrink:0;
  border:1px solid rgba(255,255,255,.2);}
#live-hex{font-size:.7rem;font-family:monospace;color:#fff;font-weight:700;}
#live-hint{font-size:.63rem;color:rgba(255,255,255,.45);flex:1;transition:color .2s;}
.focus-hint{text-align:center;font-size:.72rem;color:#00ff88;margin-top:6px;opacity:.8;}
.retake-btn{width:100%;margin-top:8px;background:var(--card2);border:1px solid var(--border);
  color:var(--sub);padding:10px;border-radius:10px;font-size:.82rem;cursor:pointer;}

/* Crop warning */
#crop-warn{display:none;margin:8px 0 0;background:#1c1202;border:1px solid #92400e;
  border-radius:10px;padding:10px 12px;font-size:.8rem;color:var(--warn);}

/* Quality bars */
#shot-guide{display:none;margin:10px 0 0;background:var(--card);
  border:1px solid var(--border);border-radius:var(--r);overflow:hidden;}
.sg-title{padding:8px 14px;font-size:.75rem;font-weight:700;letter-spacing:.05em;
  text-transform:uppercase;color:var(--accent);border-bottom:1px solid var(--border);}
.sg-bars{padding:10px 14px;display:flex;flex-direction:column;gap:8px;}
.sg-row{display:flex;align-items:center;gap:10px;}
.sg-icon{font-size:1rem;flex-shrink:0;width:22px;text-align:center;}
.sg-info{flex:1;}
.sg-label{font-size:.75rem;font-weight:600;}
.sg-bar-wrap{background:#2a2a2a;border-radius:99px;height:5px;margin-top:3px;overflow:hidden;}
.sg-bar{height:100%;border-radius:99px;transition:width .4s,background .4s;}
.sg-val{font-size:.68rem;color:var(--sub);flex-shrink:0;min-width:40px;text-align:right;}
#shot-status{margin:8px 14px 12px;padding:9px 12px;border-radius:10px;
  font-size:.8rem;font-weight:600;}
#shot-status.good{background:#052e16;color:#4ade80;border:1px solid #166534;}
#shot-status.warn{background:#1c1202;color:#fbbf24;border:1px solid #92400e;}

/* Crop preview */
#crop-section{display:none;margin:10px 0 0;}
.crop-title{font-size:.78rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;color:#00ff88;}
.crop-sub{font-size:.7rem;color:var(--sub);margin-top:2px;margin-bottom:8px;}
.crop-pw{border-radius:var(--r);overflow:hidden;border:2px solid #00ff88;
  background:#000;position:relative;display:inline-block;}
.crop-pw-wrap{text-align:center;margin-bottom:0;}
#crop-canvas{display:block;max-width:100%;height:auto;image-rendering:auto;}
.crop-zoom{position:absolute;top:10px;right:10px;background:rgba(0,0,0,.75);
  border:1px solid #00ff88;color:#00ff88;font-size:.7rem;font-weight:700;
  padding:3px 8px;border-radius:99px;pointer-events:none;}
.crop-stats{display:flex;gap:8px;margin-top:8px;}
.cs{flex:1;background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:8px 10px;text-align:center;}
.csv{font-family:monospace;font-size:.9rem;font-weight:700;color:var(--accent);}
.csl{font-size:.63rem;color:var(--sub);margin-top:2px;}
.crop-btn-row{display:flex;gap:10px;margin-top:10px;}
#crop-btn{display:none;margin:10px 0 0;width:100%;padding:14px;border-radius:10px;
  border:none;background:var(--accent);color:#000;font-size:1rem;font-weight:700;cursor:pointer;}

/* Quality modal */
#quality-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);
  z-index:100;align-items:flex-end;padding:20px;}
.qm{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:18px;width:100%;}
.qm-title{font-size:1rem;font-weight:700;color:var(--warn);margin-bottom:6px;}
.qm-body{font-size:.82rem;color:var(--sub);line-height:1.5;margin-bottom:14px;}
.qm-btns{display:flex;gap:10px;}

/* Spinner */
#spinner{display:none;text-align:center;padding:40px 18px;}
.ring{width:48px;height:48px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;
  animation:spin .8s linear infinite;margin:0 auto 14px;}
@keyframes spin{to{transform:rotate(360deg);}}
#spinner p{color:var(--sub);font-size:.85rem;}
#spin-step{font-size:.75rem;color:var(--accent);margin-top:6px;}

/* Results */
#result{display:none;padding:0 18px;}
.arch-badges{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 12px;}
.ab{background:var(--card2);border:1px solid var(--accent);border-radius:99px;
  padding:4px 10px;font-size:.65rem;color:var(--accent);}

/* Segmentation badge */
.seg-badge{display:flex;align-items:center;gap:10px;border-radius:10px;
  padding:10px 12px;margin-bottom:12px;font-size:.78rem;}
.seg-badge.found{background:#0a1f12;border:1px solid #166534;color:#4ade80;}
.seg-badge.fallback{background:#1c1202;border:1px solid #92400e;color:var(--warn);}

/* Crop used */
.crop-used{display:flex;align-items:center;gap:10px;background:#0a1f12;
  border:1px solid #166534;border-radius:10px;padding:10px 12px;
  margin-bottom:12px;font-size:.78rem;color:#4ade80;}
.cut{width:44px;height:44px;border-radius:8px;flex-shrink:0;border:1px solid #166534;}

/* Color result */
.dom-block{background:var(--card);border:1px solid var(--border);
  border-radius:var(--r);overflow:hidden;margin-bottom:12px;}
.dom-swatch{height:80px;position:relative;}
.dom-lbl{position:absolute;bottom:8px;left:12px;font-size:.68rem;
  color:rgba(255,255,255,.7);background:rgba(0,0,0,.4);padding:2px 8px;border-radius:99px;}
.dom-info{padding:12px 14px;display:flex;justify-content:space-between;align-items:center;}
.dom-code{font-size:1.8rem;font-weight:700;letter-spacing:.04em;color:var(--accent);}
.dom-sub{font-size:.72rem;color:var(--sub);margin-top:2px;}
.dom-right{text-align:right;}
.dom-hex{font-size:.85rem;font-weight:600;font-family:monospace;}
.dom-rgb{font-size:.68rem;color:var(--sub);margin-top:2px;}

.compare-row{display:flex;gap:10px;margin-bottom:12px;}
.sw-block{flex:1;border-radius:12px;overflow:hidden;border:1px solid var(--border);}
.sw-color{height:64px;}
.sw-label{background:var(--card2);padding:7px 8px;text-align:center;
  font-size:.65rem;color:var(--sub);}
.sw-label strong{display:block;font-size:.75rem;color:var(--text);font-family:monospace;}

.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;}
.mc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:13px;}
.mc-lbl{font-size:.65rem;color:var(--sub);text-transform:uppercase;
  letter-spacing:.05em;margin-bottom:5px;}
.mc-val{font-family:monospace;font-size:1.45rem;font-weight:700;line-height:1;}
.mc-sub{font-size:.65rem;margin-top:4px;}
.bar-wrap{background:var(--border);border-radius:99px;height:5px;
  margin-top:7px;overflow:hidden;}
.bar-fill{height:100%;border-radius:99px;
  background:linear-gradient(90deg,var(--accent),var(--green));}

.lab-row{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:12px;margin-bottom:12px;display:flex;justify-content:space-around;}
.li .lv{font-family:monospace;font-size:1rem;font-weight:700;text-align:center;}
.li .ll{font-size:.63rem;color:var(--sub);margin-top:2px;text-align:center;}

.sec-lbl{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;
  color:var(--sub);margin-bottom:8px;}
.t5{display:flex;align-items:center;gap:10px;background:var(--card);
  border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin-bottom:8px;}
.t5.r1{border-color:var(--accent);}
.t5-sw{width:48px;height:48px;border-radius:8px;flex-shrink:0;
  border:1px solid rgba(255,255,255,.08);}
.t5-body{flex:1;}
.t5-code{font-family:monospace;font-size:1rem;font-weight:700;}
.t5-bw{background:#2a2a2a;border-radius:99px;height:4px;margin:5px 0;overflow:hidden;}
.t5-b{height:100%;border-radius:99px;background:var(--accent);}
.t5-m{font-size:.68rem;color:var(--sub);}
.t5-c{font-family:monospace;font-size:1rem;font-weight:700;
  color:var(--accent);flex-shrink:0;}
.mc-lbl[title]{cursor:help;text-decoration:underline dotted;text-underline-offset:3px;}
</style>
</head>
<body>

<div class="hdr">
  <div class="logo">🌿</div>
  <div>
    <h1>ABACA SCANNER</h1>
    <p>GrabCut Segmentation · KNN+RF+SVM · Delta-E · CPU Edition</p>
  </div>
</div>

<!-- GUIDE SCREEN -->
<div id="guide-screen" class="pad">
  <div class="guide-card" style="margin-top:14px;">
    <div class="guide-title">📋 Photo Guide — Read Before Scanning</div>
    <div class="guide-steps">
      <div class="gs">
        <div class="gs-num">1</div>
        <div class="gs-text"><strong>Distance:</strong> Hold phone
          <em>15–25 cm</em> from fiber. Too close = blurry. Too far = box too small.</div>
      </div>
      <div class="gs">
        <div class="gs-num">2</div>
        <div class="gs-text"><strong>Lighting:</strong> Use
          <em>soft natural daylight</em>. No direct sun, no yellow indoor bulbs,
          no shadows across the fiber.</div>
      </div>
      <div class="gs">
        <div class="gs-num">3</div>
        <div class="gs-text"><strong>Background:</strong> Place fiber on a
          <em>plain white or gray surface</em>. GrabCut auto-removes background
          but plain BG helps it work better.</div>
      </div>
      <div class="gs">
        <div class="gs-num">4</div>
        <div class="gs-text"><strong>Frame it:</strong>
          <em>No hands, no punch-hole, no glare inside the green box.</em>
          Cover the flat colored surface only — the black hole and shiny patches
          will corrupt the color reading.</div>
      </div>
    </div>
    <div class="do-dont">
      <div class="dd-block do">
        <div class="dd-title">✅ Do</div>
        <div class="dd-item">☀️ Soft natural light</div>
        <div class="dd-item">📐 Box on fiber only</div>
        <div class="dd-item">🖼 Plain white/gray BG</div>
        <div class="dd-item">📏 15–25 cm distance</div>
        <div class="dd-item">📦 Box &gt; 10% of image</div>
      </div>
      <div class="dd-block dont">
        <div class="dd-title">❌ Don't</div>
        <div class="dd-item">💡 Yellow indoor light</div>
        <div class="dd-item">✋ Hands in scan box</div>
        <div class="dd-item">⚫ Punch-hole in box</div>
        <div class="dd-item">✨ Glare/shine in box</div>
        <div class="dd-item">🔍 Box &lt; 10% of image</div>
        <div class="dd-item">📱 Blurry / shaky shot</div>
      </div>
    </div>
  </div>

  <div class="btn-row" style="margin-top:4px;">
    <button class="btn primary" id="btn-cam" onclick="openCamera()">📷 Take Photo</button>
    <button class="btn" onclick="document.getElementById('file-upload').click()">🖼 Upload</button>
  </div>
  <input type="file" id="file-cam" accept="image/*"
    capture="environment" onchange="handleImage(this)">
  <input type="file" id="file-upload" accept="image/*" onchange="handleImage(this)">

  <!-- Camera stream overlay (desktop fallback) -->
  <div id="cam-overlay" style="display:none;position:fixed;inset:0;background:#000;
    z-index:200;flex-direction:column;align-items:center;justify-content:center;">
    <video id="cam-video" autoplay playsinline
      style="max-width:100%;max-height:80vh;border-radius:12px;"></video>
    <div style="display:flex;gap:12px;margin-top:16px;">
      <button onclick="snapPhoto()"
        style="padding:14px 32px;background:#d4a853;color:#000;border:none;
        border-radius:10px;font-size:1rem;font-weight:700;cursor:pointer;">
        📸 Capture
      </button>
      <button onclick="closeCamera()"
        style="padding:14px 24px;background:#333;color:#fff;border:none;
        border-radius:10px;font-size:1rem;cursor:pointer;">
        ✕ Cancel
      </button>
    </div>
    <canvas id="cam-canvas" style="display:none;"></canvas>
  </div>
</div>

<!-- PREVIEW + FOCUS BOX -->
<div id="prev-section" class="pad">
  <div style="margin-top:14px;">
    <div class="preview-wrap" id="preview-wrap">
      <img id="preview" alt="" draggable="false">
      <div id="focus-box">
        <div class="corner tl" data-corner="tl"></div>
        <div class="corner tr" data-corner="tr"></div>
        <div class="corner bl" data-corner="bl"></div>
        <div class="corner br" data-corner="br"></div>
        <div id="focus-dot"></div>
      </div>
      <div id="live-bar">
        <div id="live-sw"></div>
        <span id="live-hex">#——</span>
        <span id="live-hint">Live avg · avoid hole &amp; glare</span>
      </div>
    </div>
    <div class="focus-hint">
      🎯 Tap image to move box · Drag corners to resize · Green box = scan area
    </div>

    <div id="crop-warn">
      ⚠️ Box covers only <span id="cpw">—</span>% of image — too small for
      accurate results. <strong>Drag corners to cover more of the fiber. Avoid the hole and shiny/white patches.</strong>
    </div>

    <div id="shot-guide">
      <div class="sg-title">📊 Box Quality Check</div>
      <div id="shot-status" class="good">✅ Ready</div>
      <div class="sg-bars">
        <div class="sg-row">
          <div class="sg-icon" id="ic-bright">☀️</div>
          <div class="sg-info">
            <div class="sg-label">Brightness</div>
            <div class="sg-bar-wrap"><div class="sg-bar" id="bar-bright"></div></div>
          </div>
          <div class="sg-val" id="val-bright">—</div>
        </div>
        <div class="sg-row">
          <div class="sg-icon" id="ic-sharp">🔍</div>
          <div class="sg-info">
            <div class="sg-label">Sharpness</div>
            <div class="sg-bar-wrap"><div class="sg-bar" id="bar-sharp"></div></div>
          </div>
          <div class="sg-val" id="val-sharp">—</div>
        </div>
        <div class="sg-row">
          <div class="sg-icon" id="ic-size">📐</div>
          <div class="sg-info">
            <div class="sg-label">Box coverage</div>
            <div class="sg-bar-wrap"><div class="sg-bar" id="bar-size"></div></div>
          </div>
          <div class="sg-val" id="val-size">—</div>
        </div>
        <div class="sg-row">
          <div class="sg-icon" id="ic-fiber">🌿</div>
          <div class="sg-info">
            <div class="sg-label">Fiber pixels</div>
            <div class="sg-bar-wrap"><div class="sg-bar" id="bar-fiber"></div></div>
          </div>
          <div class="sg-val" id="val-fiber">—</div>
        </div>
      </div>
      <button onclick="autoFindFiber()" style="width:calc(100% - 28px);margin:4px 14px 12px;
        padding:9px;border-radius:8px;border:1px solid var(--accent);
        background:transparent;color:var(--accent);font-size:.8rem;
        font-weight:600;cursor:pointer;">
        🎯 Auto-find fiber region
      </button>
    </div>

    <button id="crop-btn" onclick="showCrop()">✂️ Preview Crop & Zoom</button>

    <div id="crop-section">
      <div class="crop-title">✂️ Cropped Region — Confirm Before Analyzing</div>
      <div class="crop-sub">
        This goes to GrabCut segmentation → KNN+RF+SVM → Delta-E
      </div>
      <div class="crop-pw-wrap">
        <div class="crop-pw">
          <canvas id="crop-canvas"></canvas>
          <div class="crop-zoom" id="zoom-badge">—×</div>
        </div>
      </div>
      <div class="crop-stats">
        <div class="cs">
          <div class="csv" id="crop-w">—</div>
          <div class="csl">Width px</div>
        </div>
        <div class="cs">
          <div class="csv" id="crop-h">—</div>
          <div class="csl">Height px</div>
        </div>
        <div class="cs">
          <div class="csv" id="crop-pct">—</div>
          <div class="csl">% of image</div>
        </div>
      </div>
      <div class="crop-btn-row">
        <button class="btn" onclick="hideCrop()">🔙 Re-adjust</button>
        <button class="btn green" onclick="checkThenSubmit()">
          🔍 Analyze This Crop
        </button>
      </div>
    </div>

    <button class="retake-btn" onclick="resetUI()">↩ Start over</button>
  </div>
</div>

<!-- Quality modal -->
<div id="quality-modal" style="display:none;position:fixed;inset:0;
  background:rgba(0,0,0,.85);z-index:100;align-items:flex-end;padding:20px;">
  <div class="qm">
    <div class="qm-title">⚠️ Quality Warning</div>
    <div class="qm-body" id="qm-body">Issues detected.</div>
    <div class="qm-btns">
      <button class="btn" onclick="closeModal()">🔙 Go back</button>
      <button class="btn danger" onclick="forceSubmit()">⚡ Analyze anyway</button>
    </div>
  </div>
</div>

<!-- Spinner -->
<div id="spinner" class="pad">
  <div class="ring"></div>
  <p>Analyzing fiber…</p>
  <div id="spin-step">Running GrabCut segmentation…</div>
</div>

<!-- RESULTS -->
<div id="result">

  <div class="arch-badges pad">
    <span class="ab">GrabCut Seg</span>
    <span class="ab">KNN + RF + SVM</span>
    <span class="ab">Delta-E</span>
    <span class="ab">Hybrid Score</span>
  </div>

  <!-- Segmentation result -->
  <div class="seg-badge found pad" id="seg-badge">
    <span id="seg-icon">🎯</span>
    <div>
      <div style="font-weight:700" id="seg-title">Fiber isolated by GrabCut</div>
      <div style="font-size:.68rem;opacity:.7;margin-top:1px" id="seg-sub">—</div>
    </div>
  </div>

  <!-- Crop used -->
  <div class="crop-used pad">
    <canvas id="crop-thumb" class="cut" width="44" height="44"></canvas>
    <div>
      <div style="font-weight:700">Analyzed region</div>
      <div style="font-size:.68rem;opacity:.7;margin-top:1px"
        id="crop-used-info">—</div>
    </div>
  </div>

  <div class="pad">
    <!-- Dominant color + RHS code -->
    <div class="dom-block">
      <div class="dom-swatch" id="dom-swatch">
        <div class="dom-lbl">Dominant fiber color</div>
      </div>
      <div class="dom-info">
        <div>
          <div class="dom-code" id="rhs-code">—</div>
          <div class="dom-sub">Best RHS match (KNN+RF+SVM + ΔE hybrid)</div>
          <div id="pure-de-hint" style="font-size:.7rem;margin-top:3px;">—</div>
          <span id="wb-badge" style="display:none;font-size:.63rem;
            background:#1a3a2a;color:#4ade80;border:1px solid #166534;
            padding:2px 7px;border-radius:99px;margin-top:4px;display:inline-block">
            ⚖️ White balanced
          </span>
        </div>
        <div class="dom-right">
          <div class="dom-hex" id="dom-hex">—</div>
          <div class="dom-rgb" id="dom-rgb">—</div>
        </div>
      </div>
    </div>

    <!-- Color swatches comparison -->
    <div class="compare-row">
      <div class="sw-block">
        <div class="sw-color" id="in-sw"></div>
        <div class="sw-label"><strong id="in-hex">—</strong>Scanned color</div>
      </div>
      <div class="sw-block">
        <div class="sw-color" id="mt-sw"></div>
        <div class="sw-label"><strong id="mt-hex">—</strong>RHS reference</div>
      </div>
    </div>

    <!-- Metrics -->
    <div class="metrics">
      <div class="mc">
        <div class="mc-lbl" title="Match Score = 25% ML ensemble + 75% color distance (ΔE). Higher = better color match. Not a probability.">Match Score ⓘ</div>
        <div class="mc-val" id="conf-val" style="color:var(--green)">—</div>
        <div class="bar-wrap"><div class="bar-fill" id="conf-bar"></div></div>
        <div class="mc-sub" style="color:var(--sub)">KNN+RF+SVM + ΔE (lower ΔE = better)</div>
      </div>
      <div class="mc">
        <div class="mc-lbl">Delta-E (ΔE)</div>
        <div class="mc-val" id="de-val">—</div>
        <div class="mc-sub" id="de-lbl">Color distance</div>
      </div>
    </div>

    <!-- Lab values -->
    <div class="lab-row">
      <div class="li">
        <div class="lv" id="ll">—</div>
        <div class="ll">L* Light</div>
      </div>
      <div class="li">
        <div class="lv" id="la">—</div>
        <div class="ll">a* G↔R</div>
      </div>
      <div class="li">
        <div class="lv" id="lb">—</div>
        <div class="ll">b* B↔Y</div>
      </div>
    </div>

    <!-- Top 5 -->
    <div class="sec-lbl">🏆 Top 5 RHS matches</div>
    <div id="top5-list"></div>
    <button class="retake-btn" onclick="resetUI()" style="margin-top:4px;">
      ↩ Scan another fiber
    </button>
  </div>
</div>

<script>
let box={x:0,y:0,w:0,h:0}, dragging=null, dragStart=null;
let croppedBlob=null, liveTimer=null, qualIssues=[];
let camStream=null;
const gW=()=>document.getElementById('preview-wrap');
const gB=()=>document.getElementById('focus-box');
const gI=()=>document.getElementById('preview');

// ── Camera handling ────────────────────────────────────────────────────────────
function openCamera(){
  const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
  if(isMobile){
    // On mobile: use native file input with capture — opens camera app directly
    document.getElementById('file-cam').click();
    return;
  }
  // On desktop: use getUserMedia to stream webcam in-page
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    // Browser doesn't support getUserMedia — fall back to file input
    document.getElementById('file-cam').click();
    return;
  }
  navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
  }).then(stream => {
    camStream = stream;
    const video = document.getElementById('cam-video');
    video.srcObject = stream;
    document.getElementById('cam-overlay').style.display = 'flex';
  }).catch(err => {
    // Permission denied or no camera — fall back to file input
    console.warn('Camera error:', err);
    document.getElementById('file-cam').click();
  });
}

function snapPhoto(){
  const video = document.getElementById('cam-video');
  const canvas = document.getElementById('cam-canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  closeCamera();
  canvas.toBlob(blob => {
    // Wrap blob in a fake file input event
    const file = new File([blob], 'photo.jpg', {type:'image/jpeg'});
    const dt = new DataTransfer();
    dt.items.add(file);
    const inp = document.getElementById('file-upload');
    inp.files = dt.files;
    handleImage(inp);
  }, 'image/jpeg', 0.95);
}

function closeCamera(){
  if(camStream){ camStream.getTracks().forEach(t=>t.stop()); camStream=null; }
  document.getElementById('cam-overlay').style.display = 'none';
}

function applyBox(){
  const f=gB();
  f.style.left=box.x+'px'; f.style.top=box.y+'px';
  f.style.width=box.w+'px'; f.style.height=box.h+'px';
}

function getImgLayout(){
  const img=gI();
  const dW=img.clientWidth, dH=img.clientHeight;
  const natAR=img.naturalWidth/img.naturalHeight;
  const dAR=dW/dH;
  let iW,iH,iX,iY;
  if(natAR<dAR){iH=dH; iW=iH*natAR; iX=(dW-iW)/2; iY=0;}
  else         {iW=dW; iH=iW/natAR; iX=0; iY=(dH-iH)/2;}
  const sx=img.naturalWidth/iW;
  const sy=img.naturalHeight/iH;
  return {iW,iH,iX,iY,sx,sy};
}

function initBox(){
  const {iW,iH,iX,iY}=getImgLayout();
  const s=Math.round(Math.min(iW,iH)*0.50);
  box={x:Math.round(iX+(iW-s)/2), y:Math.round(iY+(iH-s)/2), w:s, h:s};
  applyBox();
  // After placing default box, try to auto-find the fiber region
  setTimeout(autoFindFiber, 100);
}

function autoFindFiber(){
  // Draw full image to an offscreen canvas to sample pixels
  const img = gI();
  if(!img.naturalWidth) return;
  const NW = img.naturalWidth, NH = img.naturalHeight;

  // Work at reduced resolution for speed (max 200px on longest side)
  const scale = Math.min(200/NW, 200/NH, 1);
  const sw = Math.round(NW*scale), sh = Math.round(NH*scale);
  const oc = document.createElement('canvas');
  oc.width=sw; oc.height=sh;
  oc.getContext('2d').drawImage(img,0,0,NW,NH,0,0,sw,sh);
  const data = oc.getContext('2d').getImageData(0,0,sw,sh).data;

  // Score each pixel: is it a "good fiber" pixel?
  // Good = not black/hole (<40), not white/paper (>220 all channels),
  //        not pure gray (very low saturation), not very dark
  const good = new Uint8Array(sw*sh);
  for(let i=0;i<sw*sh;i++){
    const r=data[i*4], g=data[i*4+1], b=data[i*4+2];
    const brightness=(r+g+b)/3;
    const maxC=Math.max(r,g,b), minC=Math.min(r,g,b);
    const saturation = maxC>0 ? (maxC-minC)/maxC : 0;
    // Good fiber pixel: reasonable brightness, has some color (not gray/white/black)
    if(brightness>40 && brightness<220 && saturation>0.08) good[i]=1;
  }

  // Find the largest rectangular region with high density of good pixels
  // Use a sliding window approach: try different box positions and sizes
  const minW=Math.round(sw*0.25), minH=Math.round(sh*0.25);
  const stepX=Math.max(1,Math.round(sw*0.05));
  const stepY=Math.max(1,Math.round(sh*0.05));

  // Build integral image for fast area sums
  const integral = new Float32Array((sw+1)*(sh+1));
  for(let y=0;y<sh;y++){
    for(let x=0;x<sw;x++){
      integral[(y+1)*(sw+1)+(x+1)] =
        good[y*sw+x]
        + integral[y*(sw+1)+(x+1)]
        + integral[(y+1)*(sw+1)+x]
        - integral[y*(sw+1)+x];
    }
  }
  function areaSum(x0,y0,x1,y1){
    return integral[(y1+1)*(sw+1)+(x1+1)]
          -integral[y0*(sw+1)+(x1+1)]
          -integral[(y1+1)*(sw+1)+x0]
          +integral[y0*(sw+1)+x0];
  }

  // Try boxes of ~50-80% of image size, find best density
  let bestScore=-1, bestX=0, bestY=0, bestW=sw, bestH=sh;
  const sizes=[0.75, 0.65, 0.55];
  for(const frac of sizes){
    const bw=Math.round(sw*frac), bh=Math.round(sh*frac);
    if(bw<minW||bh<minH) continue;
    for(let y=0;y+bh<=sh;y+=stepY){
      for(let x=0;x+bw<=sw;x+=stepX){
        const sum=areaSum(x,y,x+bw-1,y+bh-1);
        const density=sum/(bw*bh);
        if(density>bestScore){
          bestScore=density; bestX=x; bestY=y; bestW=bw; bestH=bh;
        }
      }
    }
  }

  // Only apply auto-crop if we found a clearly better region (>55% good pixels)
  if(bestScore < 0.55) return;

  // Convert back to display coordinates
  const {iW,iH,iX,iY} = getImgLayout();
  const dispScale = iW/sw;
  const nx = Math.round(iX + bestX*dispScale);
  const ny = Math.round(iY + bestY*dispScale);
  const nw = Math.round(bestW*dispScale);
  const nh = Math.round(bestH*dispScale);

  // Animate box to new position
  box={x:nx, y:ny, w:nw, h:nh};
  clampBox();
  applyBox();
  onChange();
}

function clampBox(){
  const W=gW().clientWidth, H=gW().clientHeight;
  box.w=Math.max(50,Math.min(box.w,W));
  box.h=Math.max(50,Math.min(box.h,H));
  box.x=Math.max(0,Math.min(box.x,W-box.w));
  box.y=Math.max(0,Math.min(box.y,H-box.h));
}

function pp(e,w){
  const r=w.getBoundingClientRect(), t=e.touches?e.touches[0]:e;
  return {x:t.clientX-r.left, y:t.clientY-r.top};
}

function onWD(e){
  if(e.target===gI()){
    const p=pp(e,gW());
    box.x=Math.max(0,Math.min(p.x-box.w/2,gW().clientWidth-box.w));
    box.y=Math.max(0,Math.min(p.y-box.h/2,gW().clientHeight-box.h));
    applyBox(); onChange();
  }
}

function onBD(e){
  e.stopPropagation();
  dragging=e.target.dataset.corner||'move';
  dragStart={...box,...pp(e,gW())};
  e.preventDefault();
}

function onMv(e){
  if(!dragging||!dragStart) return;
  e.preventDefault();
  const p=pp(e,gW()), dx=p.x-dragStart.x, dy=p.y-dragStart.y;
  if(dragging==='move'){
    box.x=dragStart.x+dx; box.y=dragStart.y+dy;
  } else if(dragging==='br'){
    box.w=Math.max(50,dragStart.w+dx); box.h=Math.max(50,dragStart.h+dy);
  } else if(dragging==='bl'){
    const nw=Math.max(50,dragStart.w-dx);
    box.x=dragStart.x+(dragStart.w-nw); box.w=nw;
    box.h=Math.max(50,dragStart.h+dy);
  } else if(dragging==='tr'){
    box.w=Math.max(50,dragStart.w+dx);
    const nh=Math.max(50,dragStart.h-dy);
    box.y=dragStart.y+(dragStart.h-nh); box.h=nh;
  } else if(dragging==='tl'){
    const nw=Math.max(50,dragStart.w-dx), nh=Math.max(50,dragStart.h-dy);
    box.x=dragStart.x+(dragStart.w-nw); box.y=dragStart.y+(dragStart.h-nh);
    box.w=nw; box.h=nh;
  }
  clampBox(); applyBox();
}

function onUp(){if(dragging){dragging=null;dragStart=null;onChange();}}

function onChange(){
  clearTimeout(liveTimer);
  liveTimer=setTimeout(()=>{updateQ();updateLive();},80);
}

function getCrop(){
  const img=gI();
  if(!img.naturalWidth) return null;
  const NW=img.naturalWidth, NH=img.naturalHeight;
  const {iX,iY,sx,sy}=getImgLayout();
  const bx=Math.max(0, Math.round((box.x-iX)*sx));
  const by=Math.max(0, Math.round((box.y-iY)*sy));
  const bw=Math.min(NW-bx, Math.round(box.w*sx));
  const bh=Math.min(NH-by, Math.round(box.h*sy));
  if(bw<10||bh<10) return null;
  const c=document.createElement('canvas');
  c.width=bw; c.height=bh;
  c.getContext('2d').drawImage(img,bx,by,bw,bh,0,0,bw,bh);
  return {canvas:c,bx,by,bw,bh,W:NW,H:NH};
}

function updateLive(){
  const r=getCrop(); if(!r) return;
  const {canvas:c,bw,bh}=r;
  const data=c.getContext('2d').getImageData(0,0,bw,bh).data;

  // Avg color from center 50% (skip edges)
  let rv=0,gv=0,bv=0,n=0;
  const x0=Math.floor(bw*.25),x1=Math.floor(bw*.75);
  const y0=Math.floor(bh*.25),y1=Math.floor(bh*.75);
  for(let y=y0;y<y1;y++) for(let x=x0;x<x1;x++){
    const i=(y*bw+x)*4; rv+=data[i]; gv+=data[i+1]; bv+=data[i+2]; n++;
  }
  if(!n) return;
  rv=Math.round(rv/n); gv=Math.round(gv/n); bv=Math.round(bv/n);
  const h=`#${rv.toString(16).padStart(2,'0')}${gv.toString(16).padStart(2,'0')}${bv.toString(16).padStart(2,'0')}`;
  document.getElementById('live-sw').style.background=h;
  document.getElementById('live-hex').textContent=h.toUpperCase();

  // Count dark/glare pixels across entire box to catch off-center holes
  let darkPx=0, glarePx=0, total=data.length/4;
  for(let i=0;i<data.length;i+=4){
    const pb=(data[i]+data[i+1]+data[i+2])/3;
    if(pb<50) darkPx++;
    if(pb>210) glarePx++;
  }
  const darkRatio=darkPx/total, glareRatio=glarePx/total;
  const hint=document.getElementById('live-hint');
  if(darkRatio > 0.15){
    hint.textContent='⚫ '+Math.round(darkRatio*100)+'% dark pixels — move box off the hole!';
    hint.style.color='#ef4444';
  } else if(glareRatio > 0.15){
    hint.textContent='✨ '+Math.round(glareRatio*100)+'% glare pixels — move box off the shine!';
    hint.style.color='#fbbf24';
  } else {
    hint.textContent='Live avg · avoid hole & glare';
    hint.style.color='rgba(255,255,255,.45)';
  }
}

function bc(p){return p>=70?'#4ade80':p>=40?'#fbbf24':'#ef4444';}
function setBar(id,pct,val){
  const b=document.getElementById('bar-'+id);
  b.style.width=pct+'%'; b.style.background=bc(pct);
  document.getElementById('val-'+id).textContent=val;
}
function setIcon(id,pct){
  const icons={bright:['🌑','⚠️','☀️'],sharp:['🌫️','⚠️','🔍'],size:['📦','⚠️','📐']};
  document.getElementById('ic-'+id).textContent=
    pct>=70?icons[id][2]:pct>=40?icons[id][1]:icons[id][0];
}

function updateQ(){
  const r=getCrop(); if(!r) return;
  const {canvas:c,bw,bh,W,H}=r;
  const data=c.getContext('2d').getImageData(0,0,bw,bh).data;
  let bSum=0;
  for(let i=0;i<data.length;i+=4) bSum+=(data[i]+data[i+1]+data[i+2])/3;
  const brightness=bSum/(bw*bh);
  let eS=0;
  for(let y=1;y<bh-1;y++) for(let x=1;x<bw-1;x++){
    const i=(y*bw+x)*4;
    eS+=Math.abs(data[i-4]-data[i+4])+Math.abs(data[i-bw*4]-data[i+bw*4]);
  }
  const sharpness=eS/(bw*bh), cov=bw*bh/(W*H)*100;
  let bp=brightness>=60&&brightness<=200?100:brightness>=30?40+(brightness-30)*2:10;
  bp=Math.round(Math.min(100,Math.max(0,bp)));
  let sp=Math.round(Math.min(100,(sharpness/30)*100));
  let szp=Math.round(Math.min(100,cov/20*100));

  // Count good fiber pixels (not black hole, not white/paper, not gray, not glare)
  let goodPx=0, darkPx=0, glarePx=0, totalPx=data.length/4;
  for(let i=0;i<data.length;i+=4){
    const pr=data[i],pg=data[i+1],pb=data[i+2];
    const bright=(pr+pg+pb)/3;
    const maxC=Math.max(pr,pg,pb), minC=Math.min(pr,pg,pb);
    const sat=maxC>0?(maxC-minC)/maxC:0;
    if(bright<40) darkPx++;
    else if(bright>215) glarePx++;
    else if(sat>0.08) goodPx++;  // has real color = likely fiber
  }
  const fiberPct=Math.round(goodPx/totalPx*100);
  const darkRatio=darkPx/totalPx, glareRatio=glarePx/totalPx;

  setBar('bright',bp,Math.round(brightness)+'/255');
  setBar('sharp',sp,sp+'%');
  setBar('size',szp,cov.toFixed(1)+'%');
  setBar('fiber',Math.min(100,fiberPct*1.4),fiberPct+'%');
  setIcon('bright',bp); setIcon('sharp',sp); setIcon('size',szp);
  document.getElementById('ic-fiber').textContent=fiberPct>=50?'🌿':fiberPct>=25?'⚠️':'❌';

  document.getElementById('cpw').textContent=cov.toFixed(1);
  document.getElementById('crop-warn').style.display=cov<10?'block':'none';
  qualIssues=[];
  if(bp<40) qualIssues.push('poor lighting (too dark or overexposed)');
  if(sp<40) qualIssues.push('blurry image — hold steady or move closer');
  if(cov<10) qualIssues.push('scan box too small (under 10% of image) — drag corners to cover more fiber');
  if(darkRatio>0.25) qualIssues.push('punch-hole inside box ('+Math.round(darkRatio*100)+'% dark pixels) — move box to flat color surface only');
  else if(glareRatio>0.25) qualIssues.push('glare/shine inside box ('+Math.round(glareRatio*100)+'% overexposed) — move box off the shine');
  else if(fiberPct<30) qualIssues.push('only '+fiberPct+'% fiber pixels in box — reposition box onto the fiber surface');

  const s=document.getElementById('shot-status');
  if(qualIssues.length>0){
    s.className='warn';
    s.textContent='⚠️ '+qualIssues[0].charAt(0).toUpperCase()+qualIssues[0].slice(1);
  } else {
    s.className='good';
    s.textContent='✅ '+fiberPct+'% fiber pixels — box looks good, tap Preview Crop';
  }
}

function showCrop(){
  const r=getCrop(); if(!r) return;
  const {canvas,bw,bh,W,H}=r;
  const cc=document.getElementById('crop-canvas');
  // Draw at FULL native resolution — no downsampling, no quality loss.
  // CSS max-width:100% handles fitting to screen without resampling pixels.
  cc.width  = bw;
  cc.height = bh;
  cc.style.width  = '';
  cc.style.height = '';
  cc.getContext('2d').drawImage(canvas, 0, 0, bw, bh, 0, 0, bw, bh);
  const zoomText = '1:1 full res';
  const pct=(bw*bh/(W*H)*100).toFixed(1);
  document.getElementById('zoom-badge').textContent=zoomText;
  document.getElementById('crop-w').textContent=bw;
  document.getElementById('crop-h').textContent=bh;
  document.getElementById('crop-pct').textContent=pct+'%';
  canvas.toBlob(b=>{croppedBlob=b;},'image/jpeg',.95);
  document.getElementById('crop-thumb')
    .getContext('2d').drawImage(canvas,0,0,bw,bh,0,0,44,44);
  document.getElementById('crop-used-info').textContent=
    `${bw}×${bh}px · ${pct}% of frame · ${zoomText} zoom`;
  ['shot-guide','crop-btn','crop-warn']
    .forEach(id=>document.getElementById(id).style.display='none');
  document.getElementById('crop-section').style.display='block';
  document.getElementById('crop-section')
    .scrollIntoView({behavior:'smooth',block:'nearest'});
}

function hideCrop(){
  document.getElementById('crop-section').style.display='none';
  document.getElementById('shot-guide').style.display='block';
  document.getElementById('crop-btn').style.display='block';
  updateQ(); croppedBlob=null;
}

function checkThenSubmit(){
  const r=getCrop();
  if(r){
    const cov=r.bw*r.bh/(r.W*r.H)*100;
    if(cov<10){
      document.getElementById('qm-body').textContent=
        'Box too small ('+cov.toFixed(1)+'% of image). Drag the corners to cover more fiber for an accurate reading.';
      document.getElementById('quality-modal').style.display='flex';
      return;
    }
  }
  if(qualIssues.length>0){
    document.getElementById('qm-body').textContent=
      'Issues: '+qualIssues.join('; ')+'. Results may be less accurate. Proceed?';
    document.getElementById('quality-modal').style.display='flex';
  } else { doSubmit(); }
}
function closeModal(){document.getElementById('quality-modal').style.display='none';}
function forceSubmit(){closeModal(); doSubmit();}

const STEPS=[
  'Running GrabCut segmentation…',
  'Applying white balance…',
  'Extracting dominant color…',
  'Running KNN classifier…',
  'Running SVM classifier…',
  'Matching RHS Delta-E…'
];
let spinI=null;
function startSpin(){
  let i=0; document.getElementById('spin-step').textContent=STEPS[0];
  spinI=setInterval(()=>{i=(i+1)%STEPS.length;
    document.getElementById('spin-step').textContent=STEPS[i];},800);
}
function stopSpin(){clearInterval(spinI);}

async function doSubmit(){
  if(!croppedBlob){
    const r=getCrop(); if(!r) return;
    r.canvas.toBlob(async b=>{croppedBlob=b; await _send();},'image/jpeg',.95);
  } else { await _send(); }
}

async function _send(){
  ['crop-section','crop-btn','shot-guide','crop-warn']
    .forEach(id=>document.getElementById(id).style.display='none');
  document.getElementById('spinner').style.display='block';
  document.getElementById('result').style.display='none';
  startSpin();
  const fd=new FormData();
  fd.append('image',croppedBlob,'crop.jpg');
  fd.append('pre_cropped','1');
  try{
    const res=await fetch('/predict',{method:'POST',body:fd});
    const d=await res.json();
    stopSpin();
    document.getElementById('spinner').style.display='none';
    if(d.error){alert('Error: '+d.error); return;}
    renderResult(d);
  } catch(e){
    stopSpin();
    document.getElementById('spinner').style.display='none';
    alert('Connection error: '+e.message);
  }
}

function renderResult(d){
  const sb=document.getElementById('seg-badge');
  if(d.seg_found){
    sb.className='seg-badge found pad';
    document.getElementById('seg-icon').textContent='🎯';
    document.getElementById('seg-title').textContent='Fiber isolated by GrabCut';
    const covWarn = d.seg_coverage < 25 ? ' ⚠️ Very low — try a larger box on the fiber' : ' · background removed';
    document.getElementById('seg-sub').textContent=
      `Fiber coverage: ${d.seg_coverage}% of crop${covWarn}`;
  } else {
    sb.className='seg-badge fallback pad';
    document.getElementById('seg-icon').textContent='⚠️';
    document.getElementById('seg-title').textContent='GrabCut fallback — center crop used';
    document.getElementById('seg-sub').textContent=
      'Center 60% of image used. Tip: plain background improves segmentation';
  }
  document.getElementById('dom-swatch').style.background=d.dominant_hex;
  document.getElementById('dom-hex').textContent=d.dominant_hex.toUpperCase();
  document.getElementById('dom-rgb').textContent=
    `R${d.dominant_rgb.R} G${d.dominant_rgb.G} B${d.dominant_rgb.B}`;
  document.getElementById('rhs-code').textContent=d.rhs_code;
  const pe=document.getElementById('pure-de-hint');
  const gap = d.pure_de_2nd_val && d.pure_de_val
    ? (d.pure_de_2nd_val - d.pure_de_val).toFixed(2)
    : null;
  const gapStr = gap ? ` · gap to #2: ΔE ${gap}` : '';
  if(d.pure_de_code && d.pure_de_code !== d.rhs_code){
    pe.textContent = `Pure ΔE best: ${d.pure_de_code} (ΔE ${d.pure_de_val})${gapStr}`;
    pe.style.color = '#fbbf24';
  } else {
    pe.textContent = `Pure ΔE: ${d.rhs_code} (ΔE ${d.pure_de_val})${gapStr}`;
    pe.style.color = '#4ade80';
  }
  document.getElementById('wb-badge').style.display=d.wb_applied?'inline-block':'none';
  document.getElementById('in-sw').style.background=d.dominant_hex;
  document.getElementById('mt-sw').style.background=d.matched_hex;
  document.getElementById('in-hex').textContent=d.dominant_hex.toUpperCase();
  document.getElementById('mt-hex').textContent=d.matched_hex.toUpperCase();
  document.getElementById('conf-val').textContent=d.match_score.toFixed(1)+'%';
  document.getElementById('conf-bar').style.width=d.match_score+'%';
  document.getElementById('de-val').textContent=d.delta_e.toFixed(2);
  document.getElementById('de-val').style.color=d.de_color;
  document.getElementById('de-lbl').textContent=d.de_label;
  document.getElementById('de-lbl').style.color=d.de_color;
  document.getElementById('ll').textContent=d.dominant_lab.L.toFixed(1);
  document.getElementById('la').textContent=d.dominant_lab.a.toFixed(1);
  document.getElementById('lb').textContent=d.dominant_lab.b.toFixed(1);
  document.getElementById('top5-list').innerHTML=d.top_5.map((t,i)=>`
    <div class="t5 ${i===0?'r1':''}">
      <div class="t5-sw" style="background:${t.hex}"></div>
      <div class="t5-body">
        <div class="t5-code">${t.rhs_code}</div>
        <div class="t5-bw">
          <div class="t5-b" style="width:${Math.min(t.match_score*2,100)}%"></div>
        </div>
        <div class="t5-m">
          <span style="color:${t.de_color}">ΔE ${t.delta_e} — ${t.de_label}</span>
        </div>
      </div>
      <div class="t5-c">${t.match_score.toFixed(1)}%</div>
    </div>`).join('');
  document.getElementById('result').style.display='block';
  document.getElementById('result').scrollIntoView({behavior:'smooth',block:'start'});
}

function handleImage(inp){
  const f=inp.files[0]; if(!f) return;
  croppedBlob=null;
  const img=gI(); img.src=URL.createObjectURL(f);
  document.getElementById('guide-screen').style.display='none';
  document.getElementById('prev-section').style.display='block';
  ['result','spinner','shot-guide','crop-btn','crop-section','crop-warn']
    .forEach(id=>document.getElementById(id).style.display='none');
  img.onload=()=>{
    initBox();
    document.getElementById('shot-guide').style.display='block';
    document.getElementById('crop-btn').style.display='block';
    updateQ(); updateLive();
  };
}

function resetUI(){
  croppedBlob=null;
  ['prev-section','shot-guide','crop-btn','crop-section',
   'result','spinner','crop-warn']
    .forEach(id=>document.getElementById(id).style.display='none');
  document.getElementById('guide-screen').style.display='block';
  document.getElementById('file-cam').value='';
  document.getElementById('file-upload').value='';
}

// Attach all pointer events
(function(){
  const w=gW(), f=gB();
  w.addEventListener('mousedown',onWD);
  w.addEventListener('touchstart',onWD,{passive:true});
  f.addEventListener('mousedown',onBD);
  f.addEventListener('touchstart',onBD,{passive:false});
  document.addEventListener('mousemove',onMv);
  document.addEventListener('touchmove',onMv,{passive:false});
  document.addEventListener('mouseup',onUp);
  document.addEventListener('touchend',onUp);
})();
</script>
</body>
</html>"""


# ── HTTP handler ───────────────────────────────────────────────────────────────
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    knn = scaler_knn = svm = le = colors_db = None

    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {fmt % args}")

    def do_GET(self):
        if self.path == '/':
            b = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != '/predict':
            self.send_response(404); self.end_headers(); return

        # ── Rate limit check ───────────────────────────────────────────────
        client_ip = self.address_string()
        if not _check_rate_limit(client_ip):
            self._json({
                'error': f'Rate limit exceeded — max {MAX_REQUESTS_PER_MIN} requests/min per device.'
            })
            print(f"  ⚠️  Rate limit hit from {client_ip}")
            return

        # ── Image size check — reject before reading body ──────────────────
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > MAX_IMAGE_BYTES:
            self._json({
                'error': f'Image too large ({content_length/1e6:.1f} MB). Max allowed: {MAX_IMAGE_BYTES/1e6:.0f} MB.'
            })
            print(f"  ⚠️  Rejected oversized upload: {content_length/1e6:.1f} MB from {client_ip}")
            return

        ct   = self.headers.get('Content-Type', '')
        body = self.rfile.read(content_length)

        bnd = None
        for p in ct.split(';'):
            p = p.strip()
            if p.startswith('boundary='):
                bnd = p[9:].strip('"').encode(); break
        if not bnd:
            self._json({'error': 'No boundary'}); return

        parts    = body.split(b'--' + bnd)
        img_data = None
        fields   = {}
        for part in parts[1:]:
            if b'\r\n\r\n' not in part: continue
            header, content = part.split(b'\r\n\r\n', 1)
            content    = content.rstrip(b'\r\n--')
            header_str = header.decode('utf-8', errors='replace')
            name = None
            for seg in header_str.split(';'):
                seg = seg.strip()
                if seg.startswith('name='):
                    name = seg[5:].strip('"')
            if name == 'image':
                img_data = content
            elif name == 'pre_cropped':
                try: fields[name] = content.decode().strip()
                except: pass

        if not img_data:
            self._json({'error': 'No image field'}); return

        try:
            img = Image.open(io.BytesIO(img_data))

            # Validate image dimensions — reject suspiciously tiny images
            min_dim = 30
            if img.width < min_dim or img.height < min_dim:
                self._json({
                    'error': f'Image too small ({img.width}×{img.height}px). '
                             f'Minimum size is {min_dim}×{min_dim}px.'
                })
                return

            result = predict(img, Handler.knn, Handler.scaler_knn,
                             Handler.svm, Handler.le, Handler.colors_db)
            self._json(result)
        except Exception as ex:
            import traceback; traceback.print_exc()
            self._json({'error': str(ex)})

    def _json(self, data):
        b = json.dumps(data, indent=2).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(b)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except: return "127.0.0.1"


def main():
    import os
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5050))  # Render sets PORT automatically
    print(f"\n🌿 Abaca Scanner — CPU Edition")
    print(f"   GrabCut Segmentation + KNN/SVM + Delta-E\n")
    print("Loading models...")
    Handler.knn, Handler.scaler_knn, Handler.svm, Handler.le, Handler.colors_db = load_models()
    ip = get_local_ip()
    print(f"\n🚀 Ready!")
    print(f"   PC    → http://localhost:{PORT}")
    print(f"   Phone → http://{ip}:{PORT}")
    print(f"\n   Ctrl+C to stop\n")
    server = HTTPServer((HOST, PORT), Handler)
    try:    server.serve_forever()
    except KeyboardInterrupt: print("\nStopped.")


if __name__ == "__main__":
    main()