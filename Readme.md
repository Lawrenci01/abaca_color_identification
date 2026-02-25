# 🌿 Abaca Color Scanner — CPU Edition

Scan abaca fiber and match it to the closest RHS (Royal Horticultural Society) color code using GrabCut segmentation, a KNN+RF+SVM ensemble, and Delta-E color distance.

---

## Project Structure

```
abaca_color_ai/
├── inference_server.py     # Main web server + UI
├── segment.py              # GrabCut fiber segmentation
├── features.py             # Shared 165-feature extraction
├── run_pipeline.py         # Train models from labeled images
├── evaluate.py             # Evaluate models + confusion matrix
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── .gitignore
└── abaca_pipeline/
    ├── model_knn.joblib        # Trained KNN model (k=7)
    ├── model_rf.joblib         # Trained Random Forest (100 trees)
    ├── model_svm.joblib        # Trained SVM (RBF kernel, C=20)
    ├── scaler_knn.joblib       # Feature scaler for KNN
    ├── label_encoder.joblib    # RHS code label encoder
    └── rhs_colors.csv          # 420 RHS color Lab/RGB values
```

---

## How It Works

### Pipeline (per scan)
```
Photo → GrabCut segmentation → White balance → Dominant color
      → 165-feature extraction → KNN+RF+SVM (15% weight)
      → Delta-E vs 420 RHS colors (85% weight)
      → Hybrid score → Top 5 RHS matches
```

### Feature Vector (165 dimensions)
- `[3]`  Mean Lab color (perceptual)
- `[3]`  Std RGB (texture roughness)
- `[12]` Lab per quadrant (spatial layout)
- `[96]` RGB histograms 32 bins × 3
- `[32]` HSV histograms 16H + 8S + 8V
- `[10]` LBP texture (micro-texture)
- `[8]`  Gabor texture (fiber direction)
- `[1]`  Delta-E std deviation (color consistency)

### Model Weights
| Model | Weight | Notes |
|-------|--------|-------|
| KNN (k=7) | 25% | Needs scaled features |
| RF (100 trees) | 35% | Scale-invariant |
| SVM (RBF C=20) | 40% | Internal scaler in pipeline |
| Delta-E | 85% of final | Overrides ML until retrained on abaca |

---

## Running Locally

```bash
pip install -r requirements.txt
python inference_server.py
```

Open: http://localhost:5050

---

## Training Models (requires labeled images)

Organize images into folders named by RHS code:
```
training_data/
  59A/
    photo1.jpg
    photo2.jpg
  60B/
    photo3.jpg
  ...
```

Then run:
```bash
python run_pipeline.py --train --data training_data/
```

---

## Evaluating Models (confusion matrix)

```bash
python evaluate.py --data test_data/
```

Outputs:
- Confusion matrix (PNG)
- Per-class accuracy
- Overall top-1 and top-3 accuracy

---

## Deploying to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys

**Important:** The `abaca_pipeline/` model files must be included in your repo (or use Render's persistent disk). Models are required at startup.

---

## Box Placement Tips for Accurate Scans

| ✅ Do | ❌ Don't |
|-------|---------|
| Cover flat fiber surface | Include punch-hole |
| Use soft natural light | Scan glare/shine |
| Box > 10% of image | Box < 10% of image |
| Plain background | Hands / labels in box |
| 🌿 Fiber bar > 50% | Submit when fiber bar is red |

---

## Delta-E Interpretation

| ΔE | Meaning |
|----|---------|
| < 1.0 | Imperceptible |
| < 2.0 | Very close |
| < 3.5 | Close match |
| < 5.0 | Moderate difference |
| < 10.0 | Noticeable difference |
| ≥ 10.0 | Very different |