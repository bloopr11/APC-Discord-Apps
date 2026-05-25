# ==================================================
# ml_scorer.py  —  ML Adaptive Scoring
#
# Menggunakan scikit-learn (ringan, ~50MB) — cocok Railway free.
# TIDAK pakai TensorFlow/PyTorch.
#
# Pipeline:
#   1. Collect training data dari db_logger.get_training_data()
#   2. Train RandomForest (atau GradientBoosting) setiap N signal baru
#   3. Predict win probability untuk signal baru
#   4. Blend rule-based score + ML win_prob → final adaptive score
#
# Fallback: jika data < MIN_TRAIN_ROWS → pakai rule-based score saja
#
# ENV:
#   ML_MIN_ROWS = minimum rows untuk training (default 150)
#   ML_RETRAIN_EVERY = retrain setiap N signal baru (default 50)
# ==================================================

import os
import json
import time
import pickle
import numpy as np
from typing import Optional, Tuple

ML_MIN_ROWS      = int(os.getenv("ML_MIN_ROWS",      "2"))
ML_RETRAIN_EVERY = int(os.getenv("ML_RETRAIN_EVERY", "2"))
MODEL_PATH       = os.getenv("MODEL_PATH", "ml_model.pkl")

# ==================================================
# ML STATE  (in-memory, reload dari disk jika ada)
# ==================================================
_model        = None      # sklearn estimator
_model_meta   = {}        # {"trained_at", "n_rows", "accuracy", "features"}
_signal_count = 0         # jumlah signal sejak retrain terakhir
_is_ready     = False     # True jika model sudah siap

# Features yang dipakai (urutan harus konsisten!)
FEATURE_COLS = [
    "score",
    "cat_trend", "cat_momentum", "cat_smc",
    "cat_orderflow", "cat_fvg", "cat_mtf", "cat_session",
    "adx", "rsi", "vol_ratio",
]

# ==================================================
# SKLEARN IMPORT (optional — fallback jika tidak ada)
# ==================================================
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    import sklearn
    SKLEARN_OK = True
    print(f"✅ ML: scikit-learn {sklearn.__version__} tersedia")
except ImportError:
    SKLEARN_OK = False
    print("⚠️  scikit-learn tidak terinstall → ML scoring dinonaktifkan")
    print("   Install: pip install scikit-learn --break-system-packages")

# ==================================================
# FEATURE EXTRACTION  (dari signal data dict)
# ==================================================
def extract_features(data: dict) -> Optional[np.ndarray]:
    """
    Ekstrak feature vector dari signal dict.
    Return None jika data tidak lengkap.
    """
    cats = data.get("cats", {})
    tr   = data.get("trend", {})

    try:
        feats = np.array([
            float(data.get("score",       50)),
            float(cats.get("TREND",        0)),
            float(cats.get("MOMENTUM",     0)),
            float(cats.get("SMC",          0)),
            float(cats.get("ORDERFLOW",    0)),
            float(cats.get("FVG",          0)),
            float(cats.get("MTF",          0)),
            float(cats.get("SESSION",      0)),
            float(tr.get("adx",           20)),
            float(data.get("rsi",         50)),
            float(data.get("vol_ratio",    1)),
        ], dtype=np.float32)

        if np.any(np.isnan(feats)):
            feats = np.nan_to_num(feats, nan=0.0)

        return feats.reshape(1, -1)
    except Exception as e:
        print(f"[ML] extract_features error: {e}")
        return None

def _rows_to_Xy(rows: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert DB rows → (X, y) numpy arrays.
    Label: 1 = WIN (TP1 atau TP2), 0 = LOSS (SL1 atau SL2)
    """
    X_list, y_list = [], []

    for r in rows:
        outcome = r.get("outcome", "")
        if outcome not in ("TP1", "TP2", "SL1", "SL2"):
            continue   # skip OPEN atau MANUAL_CLOSE

        label = 1 if outcome in ("TP1", "TP2") else 0

        try:
            feat = [
                float(r.get("score",        50) or 50),
                float(r.get("cat_trend",     0) or 0),
                float(r.get("cat_momentum",  0) or 0),
                float(r.get("cat_smc",       0) or 0),
                float(r.get("cat_orderflow", 0) or 0),
                float(r.get("cat_fvg",       0) or 0),
                float(r.get("cat_mtf",       0) or 0),
                float(r.get("cat_session",   0) or 0),
                float(r.get("adx",          20) or 20),
                float(r.get("rsi",          50) or 50),
                float(r.get("vol_ratio",     1) or 1),
            ]
            X_list.append(feat)
            y_list.append(label)
        except Exception:
            continue

    if not X_list:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

# ==================================================
# TRAIN
# ==================================================
def train(rows: list) -> bool:
    """
    Train model dari rows (list of dict dari DB).
    Return True jika berhasil.
    """
    global _model, _model_meta, _is_ready

    if not SKLEARN_OK:
        return False

    X, y = _rows_to_Xy(rows)

    if len(X) < ML_MIN_ROWS:
        print(f"[ML] Training data {len(X)} < {ML_MIN_ROWS} min rows → skip")
        return False

    # ── class balance check ───────────────────────
    n_win  = int(y.sum())
    n_loss = len(y) - n_win
    if n_win < 10 or n_loss < 10:
        print(f"[ML] Class imbalance terlalu ekstrim (win={n_win}, loss={n_loss}) → skip")
        return False

    try:
        # ── Pipeline: scaler + GBM ────────────────
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators   = 100,
                max_depth      = 4,
                learning_rate  = 0.05,
                subsample      = 0.8,
                random_state   = 42,
            )),
        ])

        # cross-val 5-fold
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        pipe.fit(X, y)

        # feature importance (dari GBM dalam pipeline)
        importances = pipe.named_steps["clf"].feature_importances_
        feat_imp    = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in importances]))

        _model      = pipe
        _is_ready   = True
        _model_meta = {
            "trained_at":    time.time(),
            "n_rows":        len(X),
            "n_win":         n_win,
            "n_loss":        n_loss,
            "cv_accuracy":   round(float(cv_scores.mean()), 4),
            "cv_std":        round(float(cv_scores.std()),  4),
            "feature_imp":   feat_imp,
        }

        # simpan ke disk (best-effort — tidak fatal jika gagal)
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({"model": _model, "meta": _model_meta}, f)
            print(f"[ML] Model disimpan ke {MODEL_PATH}")
        except Exception as e:
            print(f"[ML] Gagal simpan model: {e}")

        print(
            f"[ML] ✅ Trained — rows={len(X)} win={n_win} loss={n_loss} "
            f"cv_acc={_model_meta['cv_accuracy']:.3f}±{_model_meta['cv_std']:.3f}"
        )
        return True

    except Exception as e:
        print(f"[ML] Training error: {e}")
        import traceback; traceback.print_exc()
        return False

def load_model() -> bool:
    """Muat model dari disk. Return True jika berhasil."""
    global _model, _model_meta, _is_ready

    if not SKLEARN_OK:
        return False

    try:
        with open(MODEL_PATH, "rb") as f:
            saved     = pickle.load(f)
            _model    = saved["model"]
            _model_meta = saved.get("meta", {})
            _is_ready = True
        print(f"[ML] Model dimuat dari {MODEL_PATH} (trained at "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(_model_meta.get('trained_at', 0)))})")
        return True
    except FileNotFoundError:
        print(f"[ML] Model file tidak ditemukan ({MODEL_PATH}) — perlu training dulu")
        return False
    except Exception as e:
        print(f"[ML] Gagal load model: {e}")
        return False

# ==================================================
# PREDICT WIN PROBABILITY
# ==================================================
def predict_win_prob(data: dict) -> Optional[float]:
    """
    Return win probability 0.0-1.0, atau None jika model belum siap.
    """
    if not _is_ready or _model is None:
        return None

    feats = extract_features(data)
    if feats is None:
        return None

    try:
        prob = float(_model.predict_proba(feats)[0][1])   # prob class=1 (WIN)
        return round(prob, 4)
    except Exception as e:
        print(f"[ML] predict error: {e}")
        return None

# ==================================================
# ADAPTIVE SCORE BLEND
# ==================================================
def adaptive_score(rule_score: int, data: dict) -> dict:
    """
    Blend rule-based score dengan ML win probability.

    Formula:
        jika ML ready:
            adaptive = rule_score * (1 - ML_WEIGHT) + ml_score * ML_WEIGHT
        jika tidak:
            adaptive = rule_score

    ML_WEIGHT dikecilkan saat data sedikit (confidence rendah).
    """
    win_prob = predict_win_prob(data)
    ml_ready = win_prob is not None

    if ml_ready:
        # Seberapa percaya kita pada ML — makin banyak data makin tinggi
        n_rows       = _model_meta.get("n_rows", ML_MIN_ROWS)
        ml_weight    = min(0.30, (n_rows / 500) * 0.30)   # max 30% weight di 500 rows

        ml_score     = win_prob * 100   # 0-100
        final_score  = rule_score * (1 - ml_weight) + ml_score * ml_weight
        final_score  = int(round(max(0, min(100, final_score))))

        # confidence label dari ML
        if   win_prob >= 0.72: ml_conf = "🔥 HIGH"
        elif win_prob >= 0.58: ml_conf = "⚠️ MODERATE"
        else:                  ml_conf = "❌ LOW"
    else:
        final_score = rule_score
        ml_weight   = 0.0
        ml_score    = 0.0
        ml_conf     = "N/A"

    return {
        "adaptive_score": final_score,
        "rule_score":     rule_score,
        "ml_ready":       ml_ready,
        "win_prob":       win_prob,
        "ml_score":       round(ml_score, 1) if ml_ready else None,
        "ml_weight":      round(ml_weight, 3),
        "ml_confidence":  ml_conf,
        "model_meta":     _model_meta if ml_ready else {},
    }

# ==================================================
# AUTO RETRAIN TRIGGER
# ==================================================
def maybe_retrain(db_rows: list):
    """
    Panggil setiap kali signal baru disimpan ke DB.
    Retrain jika sudah cukup signal baru.
    """
    global _signal_count

    _signal_count += 1

    if _signal_count >= ML_RETRAIN_EVERY:
        _signal_count = 0
        print(f"[ML] Auto retrain triggered ({ML_RETRAIN_EVERY} signals since last train)...")
        train(db_rows)

# ==================================================
# STATUS / EMBED
# ==================================================
def ml_status() -> str:
    """Return status string untuk Discord embed."""
    if not SKLEARN_OK:
        return "❌ scikit-learn tidak terinstall"
    if not _is_ready:
        return f"⏳ Belum siap (butuh ≥{ML_MIN_ROWS} outcomes di DB)"

    meta     = _model_meta
    acc      = meta.get("cv_accuracy", 0)
    n        = meta.get("n_rows", 0)
    trained  = time.strftime(
        "%d %b %H:%M",
        time.localtime(meta.get("trained_at", 0))
    )

    # top 3 important features
    fi    = meta.get("feature_imp", {})
    top3  = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
    top3s = " | ".join(f"{k}:{v:.3f}" for k, v in top3)

    return (
        f"✅ Ready | Acc: `{acc:.1%}` | Rows: `{n}` | "
        f"Trained: `{trained}`\n"
        f"Top features: {top3s}"
    )

def ml_embed_value(ml_result: dict) -> str:
    """Format ML result untuk satu field di Discord embed."""
    if not ml_result["ml_ready"]:
        return f"⏳ Belum ready (butuh ≥{ML_MIN_ROWS} trade outcomes)"

    bars = int(ml_result["win_prob"] * 10)
    bar  = "🟩" * bars + "⬜" * (10 - bars)

    return (
        f"{ml_result['ml_confidence']}  Win prob: `{ml_result['win_prob']*100:.1f}%`\n"
        f"{bar}\n"
        f"Rule: `{ml_result['rule_score']}` → Adaptive: `{ml_result['adaptive_score']}`  "
        f"(ML wt: `{ml_result['ml_weight']*100:.0f}%`)"
    )

# ==================================================
# STARTUP  — coba load model yang sudah ada
# ==================================================
load_model()
