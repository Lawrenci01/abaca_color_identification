# -*- coding: utf-8 -*-
import io, json, time, collections, warnings
import numpy as np
from pathlib import Path
from PIL import Image
import joblib

from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")

# ================= CONFIG =================

PIPELINE_DIR = Path("abaca_pipeline")
MAX_REQUESTS_PER_MIN = 30
RATE_WINDOW_SECS = 60
_rate_log = collections.defaultdict(list)

app = Flask(__name__, static_folder="static", static_url_path="/static")


# ================= CORS =================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ================= RATE LIMIT =================

def check_rate_limit(ip):
    now = time.time()
    _rate_log[ip] = [t for t in _rate_log[ip] if now - t < RATE_WINDOW_SECS]
    if len(_rate_log[ip]) >= MAX_REQUESTS_PER_MIN:
        return False
    _rate_log[ip].append(now)
    return True


# ================= LOAD MODELS =================

def load_models():
    import csv

    # Only mlp_a is required — b/c/d are optional (kept for backward compat)
    required = [
        PIPELINE_DIR / "model_mlp_a.joblib",
        PIPELINE_DIR / "scaler_knn.joblib",
        PIPELINE_DIR / "label_encoder.joblib",
        PIPELINE_DIR / "rhs_colors.csv",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    mlp_a = joblib.load(PIPELINE_DIR / "model_mlp_a.joblib")

    # Load b/c/d only if they exist — not required for single-MLP mode
    def _load_optional(path):
        return joblib.load(path) if path.exists() else None

    mlp_b = _load_optional(PIPELINE_DIR / "model_mlp_b.joblib")
    mlp_c = _load_optional(PIPELINE_DIR / "model_mlp_c.joblib")
    mlp_d = _load_optional(PIPELINE_DIR / "model_mlp_d.joblib")

    scaler = joblib.load(PIPELINE_DIR / "scaler_knn.joblib")
    le     = joblib.load(PIPELINE_DIR / "label_encoder.joblib")

    # Log what was loaded
    loaded = ["mlp_a"]
    if mlp_b: loaded.append("mlp_b")
    if mlp_c: loaded.append("mlp_c")
    if mlp_d: loaded.append("mlp_d")
    print(f"  Models loaded: {', '.join(loaded)}")

    colors_db = {}
    with open(PIPELINE_DIR / "rhs_colors.csv") as f:
        for row in csv.DictReader(f):
            colors_db[row["rhs_code"]] = {
                "L": float(row["Lab_L"]),
                "a": float(row["Lab_a"]),
                "b": float(row["Lab_b"]),
                "hex": "#{:02x}{:02x}{:02x}".format(
                    int(row["R"]), int(row["G"]), int(row["B"])
                ),
            }

    return mlp_a, mlp_b, mlp_c, mlp_d, scaler, le, colors_db


try:
    mlp_a, mlp_b, mlp_c, mlp_d, scaler, le, colors_db = load_models()
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️  Model load failed: {e}")
    mlp_a = mlp_b = mlp_c = mlp_d = scaler = le = colors_db = None

from features import predict


# ================= SERVE index.html =================

@app.route("/")
@app.route("/scanner")
@app.route("/history")
@app.route("/settings")
@app.route("/scan-detail")
def index():
    return send_from_directory("templates", "index.html")


# ================= ADMIN PAGE ROUTES =================

@app.route("/admin")
@app.route("/admin/users")
@app.route("/admin/history")
@app.route("/admin/settings")
def admin_index():
    return send_from_directory("templates", "index.html")


# ================= SERVICE WORKER =================

@app.route("/sw.js")
def service_worker():
    return send_from_directory(".", "sw.js", mimetype="application/javascript")


# ================= USER API ROUTES =================

@app.route("/api/login", methods=["POST"])
def api_login():
    from db import get_db
    data = request.json
    if not data or not data.get("username") or not data.get("pin"):
        return jsonify({"error": "username and pin are required"}), 400
    user = get_db().verify_user(data.get("username"), data.get("pin"))
    return jsonify(user if user else {"error": "Wrong credentials"})


@app.route("/api/register", methods=["POST"])
def api_register():
    from db import get_db
    data = request.json
    if not data or not data.get("username") or not data.get("pin"):
        return jsonify({"error": "username and pin are required"}), 400
    try:
        return jsonify(get_db().create_user(data.get("username"), data.get("pin")))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/scans")
def api_scans():
    from db import get_db
    user_id = request.args.get("user_id")
    scans = get_db().get_scans(user_id=user_id)

    # compute stats
    import datetime
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    today_scans = [s for s in scans if s.get("scanned_at", "").startswith(today)]
    verified = [s for s in scans if s.get("verified") == 1]
    des = [s["delta_e"] for s in scans if s.get("delta_e") is not None]
    avg_de = round(sum(des) / len(des), 2) if des else "—"

    return jsonify({
        "scans": scans,
        "stats": {
            "total": len(scans),
            "today": len(today_scans),
            "verified": len(verified),
            "avg_de": avg_de,
        }
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    from db import get_db
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    scan_id = get_db().save_scan(data)

    # ── Illuminant correction learning ────────────────────────────────────────
    # Every time a grader saves a verified scan, feed it to the illuminant
    # corrector so subsequent scans in the same session benefit from the
    # learned lighting offset.
    try:
        from features import record_grader_confirmation
        dom_lab = data.get("dominant_lab") or {}
        L = dom_lab.get("L") or data.get("dominant_lab_L")
        a = dom_lab.get("a") or data.get("dominant_lab_a")
        b = dom_lab.get("b") or data.get("dominant_lab_b")
        grade = data.get("rhs_grade") or data.get("rhs_code")
        if L is not None and a is not None and b is not None and grade:
            record_grader_confirmation(
                measured_lab     = (float(L), float(a), float(b)),
                correct_rhs_code = str(grade).strip().upper(),
                colors_db        = colors_db,
            )
    except Exception:
        pass  # never block save on correction failure

    return jsonify({"scan_id": scan_id, "ok": True})


@app.route("/api/reset-illuminant", methods=["POST"])
def api_reset_illuminant():
    """
    Reset the session illuminant correction.
    Call when the grader moves to a new lighting environment.
    The frontend can expose this as a "New lighting condition" button.
    """
    try:
        from features import reset_illuminant_correction
        reset_illuminant_correction()
    except Exception:
        pass
    return jsonify({"ok": True, "message": "Illuminant correction reset."})



@app.route("/api/stats")
def api_stats():
    from db import get_db
    stats = get_db().get_stats()
    return jsonify(stats)


# ================= ADMIN API ROUTES =================

@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    from db import get_db
    data = request.json
    if not data or not data.get("username") or not data.get("pin"):
        return jsonify({"error": "username and pin are required"}), 400
    user = get_db().verify_admin(data.get("username"), data.get("pin"))
    return jsonify(user if user else {"error": "Wrong credentials"})


@app.route("/api/admin/scans")
def api_admin_scans():
    from db import get_db
    scans = get_db().get_all_scans()
    return jsonify(scans)


@app.route("/api/admin/users")
def api_admin_users():
    from db import get_db
    users = get_db().get_all_users()
    return jsonify(users)


@app.route("/api/heartbeat", methods=["POST"])
def api_heartbeat():
    from db import get_db
    data = request.json or {}
    user_id = data.get("user_id")
    if user_id:
        get_db().update_last_seen(user_id)
    return jsonify({"ok": True})


@app.route("/api/admin/stats")
def api_admin_stats():
    from db import get_db
    return jsonify(get_db().get_admin_stats())


@app.route("/api/admin/delete-user/<string:user_id>", methods=["POST"])
def api_admin_delete_user(user_id):
    from db import get_db
    db = get_db()
    db.delete_user_scans(user_id)
    db.delete_user(user_id)
    return jsonify({"ok": True})


@app.route("/admin/delete-user/<string:user_id>", methods=["POST"])
def delete_user(user_id):
    from db import get_db
    db = get_db()
    db.delete_user_scans(user_id)
    db.delete_user(user_id)
    return jsonify({"ok": True})


@app.route("/api/admin/retrain", methods=["POST"])
def api_admin_retrain():
    """
    Full retrain pipeline using verified scans from Supabase.

    Flow:
      1. Fetch all verified scans from DB
      2. Synthesize fiber swatch images from their dominant_rgb values
      3. Load existing augmented manifest (swatch training data)
      4. Append verified scans as additional training samples
      5. Retrain Quad MLP ensemble (4 models: a, b, c, d)
      6. Hot-reload models into memory (no restart needed)
      7. Return accuracy report
    """
    import threading, csv, time, random, traceback
    import numpy as np
    from PIL import Image, ImageFilter
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from features import extract_features

    # ── 1. Fetch verified scans ───────────────────────────────────────────────
    try:
        db = get_db()
        all_scans = db.get_all_scans()
    except Exception as e:
        return jsonify({"ok": False, "message": f"DB error: {e}"}), 500

    verified = [
        s for s in all_scans
        if s.get("verified") == 1
           and s.get("dominant_rgb")
           and s.get("rhs_grade")
    ]

    if len(verified) < 10:
        return jsonify({
            "ok": False,
            "message": f"Not enough verified scans to retrain (have {len(verified)}, need at least 10).",
            "verified_count": len(verified)
        })

    # ── 2. Build correction map for report ───────────────────────────────────
    correction_map = {}
    for s in all_scans:
        predicted = s.get("rhs_grade", "")
        actual = (s.get("correction") or "").strip().upper()
        if predicted and actual and predicted != actual and s.get("verified") == 1:
            key = f"{predicted}→{actual}"
            correction_map[key] = correction_map.get(key, 0) + 1
    top_errors = sorted(correction_map.items(), key=lambda x: x[1], reverse=True)[:10]

    # ── 3. Synthesize swatch images from verified scan RGB values ─────────────
    def make_fiber_swatch(r, g, b, size=96):
        """Synthesize a realistic fiber swatch patch from an RGB value."""
        rng = random.Random(r * 1000 + g * 100 + b)
        arr = np.full((size, size, 3), [r, g, b], dtype=np.float32)
        # Add fiber strand texture (vertical brightness variation)
        for _ in range(rng.randint(8, 16)):
            x = rng.randint(0, size - 1)
            brightness = rng.uniform(0.88, 1.12)
            w = rng.randint(1, 2)
            for dx in range(w):
                col = min(x + dx, size - 1)
                arr[:, col] = np.clip(arr[:, col] * brightness, 0, 255)
        # Add subtle noise
        noise = np.random.default_rng(r + g + b).normal(0, 2.5, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
        return img.filter(ImageFilter.GaussianBlur(radius=0.4))

    # ── 4. Load existing swatch training data ─────────────────────────────────
    MANIFEST = PIPELINE_DIR / "augmented_manifest.csv"
    if not MANIFEST.exists():
        return jsonify({
            "ok": False,
            "message": "augmented_manifest.csv not found. Run augment_dataset.py first."
        }), 500

    try:
        with open(MANIFEST, encoding='utf-8') as f:
            manifest = list(csv.DictReader(f))
        train_rows = [m for m in manifest if m.get('split') == 'train']
    except Exception as e:
        return jsonify({"ok": False, "message": f"Manifest read error: {e}"}), 500

    # ── 5. Extract features from all training data ────────────────────────────
    t_start = time.time()
    X, y_codes = [], []
    errors = 0

    for row in train_rows:
        path = Path(row['path'])
        if not path.exists():
            errors += 1
            continue
        try:
            X.append(extract_features(Image.open(path)))
            y_codes.append(row['rhs_code'])
        except Exception:
            errors += 1

    # Inject verified real-world scans (augmented 3× for weight)
    injected = 0
    for s in verified:
        rgb = s.get("dominant_rgb", {})
        r = int(rgb.get("R", 0) if isinstance(rgb, dict) else 0)
        g = int(rgb.get("G", 0) if isinstance(rgb, dict) else 0)
        b = int(rgb.get("B", 0) if isinstance(rgb, dict) else 0)
        grade = s.get("correction") or s.get("rhs_grade")
        if not grade:
            continue
        grade = grade.strip().upper()
        try:
            swatch = make_fiber_swatch(r, g, b)
            feat = extract_features(swatch)
            # Add 3 slightly varied versions for better generalization
            for jitter in [(0, 0, 0), (3, -2, 2), (-2, 3, -3)]:
                jr, jg, jb = (
                    min(255, max(0, r + jitter[0])),
                    min(255, max(0, g + jitter[1])),
                    min(255, max(0, b + jitter[2]))
                )
                swatch_j = make_fiber_swatch(jr, jg, jb)
                X.append(extract_features(swatch_j))
                y_codes.append(grade)
            injected += 1
        except Exception:
            errors += 1

    if len(X) < 50:
        return jsonify({"ok": False, "message": f"Too few usable training samples ({len(X)})."}), 500

    X = np.array(X, dtype=np.float32)

    # ── 6. Fit scaler + Quad MLP ────────────────────────────────────────────
    try:
        le = LabelEncoder()
        y_int = le.fit_transform(y_codes)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        def build_mlp(seed, layer_sizes, activation='relu', alpha=0.001):
            return MLPClassifier(
                hidden_layer_sizes=layer_sizes,
                activation=activation, solver='adam', alpha=alpha,
                batch_size=256, learning_rate='adaptive', learning_rate_init=0.001,
                max_iter=500, tol=1e-4, random_state=seed, verbose=False,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
            )

        mlp_configs = [
            (42, 'a', (1024, 768, 512, 256), 'relu', 0.001),
            (7, 'b', (1024, 1024), 'relu', 0.001),
            (123, 'c', (512, 512, 512), 'tanh', 0.0005),
            (999, 'd', (768, 512, 384, 256), 'relu', 0.001),
        ]

        results = {}
        for seed, label, layers, activation, alpha in mlp_configs:
            mlp = build_mlp(seed, layers, activation, alpha)
            mlp.fit(X_scaled, y_int)
            acc = (mlp.predict(X_scaled) == y_int).mean()
            results[label] = (mlp, round(acc * 100, 2))

    except Exception as e:
        return jsonify({"ok": False, "message": f"Training error: {traceback.format_exc()}"}), 500

    # ── 7. Save models ────────────────────────────────────────────────────────
    try:
        for label in ['a', 'b', 'c', 'd']:
            joblib.dump(results[label][0], PIPELINE_DIR / f"model_mlp_{label}.joblib", compress=3)
        joblib.dump(scaler, PIPELINE_DIR / "scaler_knn.joblib", compress=3)
        joblib.dump(le, PIPELINE_DIR / "label_encoder.joblib", compress=3)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Save error: {e}"}), 500

    # ── 8. Hot-reload into live memory (no restart needed) ───────────────────
    global mlp_a, mlp_b, mlp_c, mlp_d
    try:
        mlp_a = results['a'][0]
        mlp_b = results['b'][0]
        mlp_c = results['c'][0]
        mlp_d = results['d'][0]
        # Reload scaler and le via the features module's predict function
        import features as _feat_mod
        _feat_mod._loaded_scaler = scaler
        _feat_mod._loaded_le = le
    except Exception:
        pass

    elapsed = round(time.time() - t_start, 1)

    return jsonify({
        "ok": True,
        "message": "Retraining complete. Quad MLP models updated live.",
        "verified_count": len(verified),
        "injected_scans": injected,
        "swatch_samples": len(train_rows) - errors,
        "total_training_samples": len(X),
        "skipped_errors": errors,
        "mlp_a_accuracy": results['a'][1],
        "mlp_b_accuracy": results['b'][1],
        "mlp_c_accuracy": results['c'][1],
        "mlp_d_accuracy": results['d'][1],
        "elapsed_seconds": elapsed,
        "classes": len(le.classes_),
        "top_misclassifications": [{"pattern": k, "count": v} for k, v in top_errors],
        "note": "Models hot-reloaded — no restart needed."
    })


@app.route("/api/admin/retrain/export", methods=["POST"])
def api_retrain_export():
    """
    Generate a downloadable PDF/CSV retrain report from the provided retrain result JSON.
    Accepts the retrain result as JSON body, returns a CSV file.
    """
    import csv, io as _io, datetime
    from flask import make_response

    data = request.get_json(force=True) or {}
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    buf = _io.StringIO()
    w = csv.writer(buf)

    # Header
    w.writerow(["Abaca Scanner — Model Retrain Report"])
    w.writerow(["Generated", now])
    w.writerow([])

    # Summary
    w.writerow(["SUMMARY"])
    w.writerow(["Metric", "Value"])
    w.writerow(["Verified Scans Used", data.get("verified_count", "—")])
    w.writerow(["Real Scans Injected", data.get("injected_scans", "—")])
    w.writerow(["Swatch Samples", data.get("swatch_samples", "—")])
    w.writerow(["Total Training Samples", data.get("total_training_samples", "—")])
    w.writerow(["Skipped (errors)", data.get("skipped_errors", "—")])
    w.writerow(["RHS Classes", data.get("classes", "—")])
    w.writerow(["Training Time (seconds)", data.get("elapsed_seconds", "—")])
    w.writerow([])

    # Accuracy
    w.writerow(["MODEL ACCURACY"])
    w.writerow(["Model", "Train Accuracy (%)"])
    w.writerow(["MLP-A", data.get("mlp_a_accuracy", "—")])
    w.writerow(["MLP-B", data.get("mlp_b_accuracy", "—")])
    w.writerow(["MLP-C", data.get("mlp_c_accuracy", "—")])
    w.writerow(["MLP-D", data.get("mlp_d_accuracy", "—")])
    avg = None
    try:
        avg = round(
            (data["mlp_a_accuracy"] + data["mlp_b_accuracy"] + data["mlp_c_accuracy"] + data["mlp_d_accuracy"]) / 4, 2)
    except Exception:
        pass
    if avg:
        w.writerow(["Ensemble Average", avg])
    w.writerow([])

    # Top misclassifications
    errors = data.get("top_misclassifications", [])
    if errors:
        w.writerow(["TOP GRADE CORRECTIONS APPLIED"])
        w.writerow(["Pattern (Predicted→Corrected)", "Count"])
        for e in errors:
            w.writerow([e.get("pattern", "—"), e.get("count", 0)])
        w.writerow([])

    w.writerow(["Note", data.get("note", "")])

    filename = f"abaca_retrain_report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return resp


# ================= ML PREDICT =================

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_route():
    if request.method == "OPTIONS":
        return "", 204

    if mlp_a is None:
        return jsonify({"error": "Models not loaded. Check server logs."}), 503

    if not check_rate_limit(request.remote_addr):
        return jsonify({"error": "Rate limit exceeded. Try again in a minute."}), 429

    try:
        # Support both raw bytes and multipart form
        if request.content_type and "multipart" in request.content_type:
            f = request.files.get("image")
            if not f:
                return jsonify({"error": "No image file in request"}), 400
            img_bytes = f.read()
        else:
            img_bytes = request.data
            if not img_bytes:
                return jsonify({"error": "No image data received"}), 400

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ── Scan quality gate ────────────────────────────────────────────────
        # Checks for hard technical failures (overexposed, too dark, no color).
        # Does NOT try to detect "not abaca" — that's not reliably possible.
        # Soft warnings are passed through so the UI can show them.
        from features import scan_quality_check
        quality = scan_quality_check(img)
        if not quality["scannable"]:
            return jsonify({
                "error":        "unscannable",
                "warnings":     quality["warnings"],
                "mean_lum":     quality["mean_lum"],
                "mean_sat":     quality["mean_sat"],
                "scannable":    False,
            }), 422

        # ── White balance correction ─────────────────────────────────────
        # Priority 1: use gains provided by client (from calibration card)
        # Priority 2: auto-detect and correct warm/red/cool cast from image
        wb_applied = False
        try:
            wb_r = request.form.get("wb_r")
            wb_g = request.form.get("wb_g")
            wb_b = request.form.get("wb_b")
            if wb_r and wb_g and wb_b:
                gr, gg, gb = float(wb_r), float(wb_g), float(wb_b)
                if 0.5 <= gr <= 2.5 and 0.5 <= gg <= 2.5 and 0.5 <= gb <= 2.5:
                    arr = np.array(img, dtype=np.float32)
                    arr[:, :, 0] = np.clip(arr[:, :, 0] * gr, 0, 255)
                    arr[:, :, 1] = np.clip(arr[:, :, 1] * gg, 0, 255)
                    arr[:, :, 2] = np.clip(arr[:, :, 2] * gb, 0, 255)
                    img = Image.fromarray(arr.astype(np.uint8))
                    wb_applied = True
        except (ValueError, TypeError):
            pass

        # ── Auto WB: correct neutral-scene casts but skip colored fiber ────────
        # CRITICAL: do NOT apply gray-world WB to red/purple-dominant images.
        # Abaca fiber in RHS groups 44–73 (red, red-purple) will have mean_R
        # well above mean_G and mean_B — this IS the fiber color, not a camera
        # cast.  Correcting it desaturates the fiber and shifts Lab toward dark
        # neutral codes (187A, N187A), causing systematic mis-grading.
        if not wb_applied:
            try:
                arr = np.array(img, dtype=np.float32)
                mean_r = arr[:, :, 0].mean()
                mean_g = arr[:, :, 1].mean()
                mean_b = arr[:, :, 2].mean()
                mean_all = (mean_r + mean_g + mean_b) / 3.0

                # Skip WB when fiber color dominates:
                # Red fiber:    R leads both G and B by > 25 units
                # Purple fiber: both R and B lead G (R>G, B>G)
                red_dominant    = (mean_r - mean_g) > 25 and (mean_r - mean_b) > 25
                purple_dominant = (mean_r > mean_g + 10) and (mean_b > mean_g + 5)

                if not (red_dominant or purple_dominant) and mean_all > 10:
                    gr = mean_all / mean_r if mean_r > 5 else 1.0
                    gg = mean_all / mean_g if mean_g > 5 else 1.0
                    gb = mean_all / mean_b if mean_b > 5 else 1.0
                    max_dev = max(abs(gr - 1.0), abs(gg - 1.0), abs(gb - 1.0))
                    if max_dev > 0.08:
                        gr = max(0.6, min(1.8, gr))
                        gg = max(0.6, min(1.8, gg))
                        gb = max(0.6, min(1.8, gb))
                        arr[:, :, 0] = np.clip(arr[:, :, 0] * gr, 0, 255)
                        arr[:, :, 1] = np.clip(arr[:, :, 1] * gg, 0, 255)
                        arr[:, :, 2] = np.clip(arr[:, :, 2] * gb, 0, 255)
                        img = Image.fromarray(arr.astype(np.uint8))
                        wb_applied = True
            except Exception:
                pass  # Never block prediction on WB failure

        result = predict(img, mlp_a, mlp_b, mlp_c, mlp_d, scaler, le, colors_db)
        result["wb_applied"] = wb_applied
        result["scan_warnings"] = quality.get("warnings", [])
        result["scan_quality"]  = {
            "mean_lum": quality["mean_lum"],
            "mean_sat": quality["mean_sat"],
            "lum_std":  quality["lum_std"],
        }

        # Sanitize all numeric fields so JS .toFixed() never crashes on null/undefined
        def _f(v, default=0.0):
            try:
                return round(float(v), 4)
            except:
                return default

        result["delta_e"] = _f(result.get("delta_e"), 0.0)
        result["match_score"] = _f(result.get("match_score"), 0.0)
        result["dominant_hex"] = result.get("dominant_hex") or "#888888"
        result["matched_hex"] = result.get("matched_hex") or "#888888"
        result["rhs_code"] = result.get("rhs_code") or result.get("rhs_grade") or "—"
        result["verdict"] = result.get("verdict") or "Unknown"
        result["verdict_color"] = result.get("verdict_color") or "#888888"
        result["dominant_rgb"] = result.get("dominant_rgb") or {"R": 0, "G": 0, "B": 0}
        result["dominant_lab"] = result.get("dominant_lab") or {"L": 0, "a": 0, "b": 0}
        result["seg_found"] = bool(result.get("seg_found", False))
        result["seg_coverage"] = _f(result.get("seg_coverage"), 0.0)
        result["wb_applied"] = bool(result.get("wb_applied", False))

        # Normalize top_5 — features.py may use different key names
        top5_raw = result.get("top_5") or result.get("top5") or []
        top5_norm = []
        for t in top5_raw:
            # rhs_code may come as "code", "rhs_code", "label", "class"
            rhs = (t.get("rhs_code") or t.get("code") or
                   t.get("label") or t.get("class") or
                   t.get("rhs_grade") or "—")
            # delta_e may come as "delta_e", "de", "deltaE", "distance"
            de_val = (t.get("delta_e") or t.get("de") or
                      t.get("deltaE") or t.get("distance") or 0.0)
            # match_score may come as "match_score", "score", "confidence", "prob"
            sc_val = (t.get("match_score") or t.get("score") or
                      t.get("confidence") or t.get("prob") or 0.0)
            de_f = _f(de_val, 0.0)
            sc_f = _f(sc_val, 0.0)
            # color coding for delta-e
            de_color = ("#1a8c45" if de_f < 2 else
                        "#c47c00" if de_f < 5 else "#c0001a")
            de_label = ("Excellent" if de_f < 2 else
                        "Good" if de_f < 5 else
                        "Fair" if de_f < 10 else "Poor")
            top5_norm.append({
                "rhs_code": rhs,
                "delta_e": de_f,
                "match_score": sc_f,
                "hex": t.get("hex") or t.get("color") or "#888888",
                "de_color": t.get("de_color") or de_color,
                "de_label": t.get("de_label") or de_label,
            })
        result["top_5"] = top5_norm

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= HEALTH CHECK =================

@app.route("/health")
def health():
    from db import get_db
    db = get_db()
    return jsonify({
        "status": "ok",
        "models_loaded": mlp_a is not None,
        "supabase": db._supabase is not None,
    })


# ================= MAIN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)