"""
xai_grud.py — File 5/N  (هستهٔ علمیِ مقاله)
==================================================================
استخراج توضیح از GRU-D با پنج روش، اندازه‌گیری توافق بین آن‌ها روی کل
test، تخمین عدم‌قطعیت با MC-Dropout، و آزمون فرضیهٔ اصلی:

   «آیا روش‌های XAI روی نمونه‌هایی که مدل مطمئن و درست است بیشتر با هم
    توافق دارند تا نمونه‌هایی که نامطمئن یا غلط است؟»

پنج روش انتساب (همه در سطح ویژگی، با تجمیع روی زمان):
   1. Saliency              |∂p/∂x|
   2. SmoothGrad            میانگین saliency روی نسخه‌های نوفه‌دار
   3. Gradient × Input      |x · ∂p/∂x|
   4. Integrated Gradients   انتگرال گرادیان از baseline صفر
   5. Occlusion             |Δp| با صفرکردن هر ویژگی

چرا attention نیست: GRU-D ما سرِ attention ندارد؛ و مهم‌تر اینکه وزن
attention توضیحِ قابل‌اعتمادی نیست (همان نکته‌ای که sanity check شما
نشان می‌دهد). این پنج روش از خانواده‌های متفاوت‌اند و توافقشان آزمون‌پذیر.

ورودی: results/artifacts/grud_model.keras , grud_test_data.npz  (از فایل ۳)
خروجی: results/xai/  (per_patient_xai.json , agreement_matrix.csv ,
                      hypothesis.json)
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from typing import Dict, List

import numpy as np
import tensorflow as tf
import keras
from scipy.stats import spearmanr, pointbiserialr

# اطمینان از ثبت GRUDCell برای بارگذاری مدل
from grud_model import GRUDCell  # noqa: F401

RESULTS_DIR = "results"
XAI_DIR = os.path.join(RESULTS_DIR, "xai")
ART_DIR = os.path.join(RESULTS_DIR, "artifacts")

METHODS = ["saliency", "smoothgrad", "grad_input", "integrated_gradients", "occlusion"]


# ==================================================================
# بارگذاری
# ==================================================================

def load_assets():
    model = keras.saving.load_model(os.path.join(ART_DIR, "grud_model.keras"))
    d = np.load(os.path.join(ART_DIR, "grud_test_data.npz"), allow_pickle=True)
    X = d["X_test"].astype(np.float32)             # (N, T, 3F)
    y = d["y_test"].astype(int)
    pid = d["pid_test"].astype(str)
    feats = list(d["feature_names"])
    F = len(feats)
    return model, X, y, pid, feats, F


def predict_prob(model, X) -> np.ndarray:
    return model(X, training=False).numpy().ravel()


# ==================================================================
# روش‌های گرادیانی (روی F کانالِ اولِ ورودی = مقادیر)
# ==================================================================

def _grad(model, X):
    x = tf.convert_to_tensor(X, tf.float32)
    with tf.GradientTape() as t:
        t.watch(x)
        p = model(x, training=False)
    return t.gradient(p, x).numpy()               # (N, T, 3F)


def saliency(model, X, F):
    g = _grad(model, X)[:, :, :F]
    return np.abs(g).sum(axis=1)                   # (N, F)


def smoothgrad(model, X, F, n=25, sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    acc = np.zeros((X.shape[0], F))
    for _ in range(n):
        noise = np.zeros_like(X)
        noise[:, :, :F] = rng.normal(0, sigma, X[:, :, :F].shape)
        acc += np.abs(_grad(model, X + noise)[:, :, :F]).sum(axis=1)
    return acc / n


def grad_input(model, X, F):
    g = _grad(model, X)[:, :, :F]
    return np.abs(g * X[:, :, :F]).sum(axis=1)


def integrated_gradients(model, X, F, steps=32):
    X_base = X.copy(); X_base[:, :, :F] = 0.0
    diff = X - X_base                              # غیرصفر فقط در F کانال اول
    acc = np.zeros_like(X)
    for a in np.linspace(0, 1, steps):
        acc += _grad(model, X_base + a * diff)
    ig = diff * (acc / steps)
    return np.abs(ig[:, :, :F]).sum(axis=1)


def occlusion(model, X, F):
    p_full = predict_prob(model, X)
    imp = np.zeros((X.shape[0], F))
    for f in range(F):
        Xo = X.copy(); Xo[:, :, f] = 0.0           # صفر = میانگین (داده استاندارد)
        imp[:, f] = np.abs(p_full - predict_prob(model, Xo))
    return imp


def compute_all_methods(model, X, F) -> Dict[str, np.ndarray]:
    print("  محاسبهٔ روش‌ها...")
    out = {}
    out["saliency"] = saliency(model, X, F);                  print("    ✓ saliency")
    out["smoothgrad"] = smoothgrad(model, X, F);              print("    ✓ smoothgrad")
    out["grad_input"] = grad_input(model, X, F);              print("    ✓ grad_input")
    out["integrated_gradients"] = integrated_gradients(model, X, F); print("    ✓ IG")
    out["occlusion"] = occlusion(model, X, F);                print("    ✓ occlusion")
    return out


# ==================================================================
# توافق بین روش‌ها
# ==================================================================

def pairwise_agreement(attr: Dict[str, np.ndarray]) -> np.ndarray:
    """میانگین Spearman جفتیِ روش‌ها برای هر بیمار → (N,)."""
    N = next(iter(attr.values())).shape[0]
    pairs = list(combinations(METHODS, 2))
    per_patient = np.zeros(N)
    for i in range(N):
        rhos = []
        for a, b in pairs:
            r, _ = spearmanr(attr[a][i], attr[b][i])
            if not np.isnan(r):
                rhos.append(r)
        per_patient[i] = np.mean(rhos) if rhos else np.nan
    return per_patient


def agreement_matrix(attr: Dict[str, np.ndarray]) -> np.ndarray:
    """ماتریس میانگین توافق (روی بیماران) بین هر جفت روش → (5,5)."""
    M = len(METHODS)
    mat = np.eye(M)
    for ai, a in enumerate(METHODS):
        for bi, b in enumerate(METHODS):
            if bi <= ai:
                continue
            rhos = [spearmanr(attr[a][i], attr[b][i])[0]
                    for i in range(attr[a].shape[0])]
            rhos = [r for r in rhos if not np.isnan(r)]
            mat[ai, bi] = mat[bi, ai] = np.mean(rhos) if rhos else np.nan
    return mat


def bootstrap_mean_ci(x: np.ndarray, n_boot=2000, seed=42):
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ==================================================================
# عدم‌قطعیت MC-Dropout
# ==================================================================

def mc_dropout(model, X, n=50, seed=0) -> Dict[str, np.ndarray]:
    tf.random.set_seed(seed)
    samples = np.stack([model(X, training=True).numpy().ravel() for _ in range(n)], axis=0)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    eps = 1e-8
    entropy = -(mean * np.log(mean + eps) + (1 - mean) * np.log(1 - mean + eps))
    return {"mean": mean, "std": std, "entropy": entropy}


# ==================================================================
# اجرا
# ==================================================================

def run():
    os.makedirs(XAI_DIR, exist_ok=True)
    model, X, y, pid, feats, F = load_assets()
    print(f"  test: {X.shape}  ویژگی‌ها: {F}")

    attr = compute_all_methods(model, X, F)

    print("\n  توافق بین روش‌ها (Spearman)...")
    agr = pairwise_agreement(attr)
    a_mean, a_lo, a_hi = bootstrap_mean_ci(agr)
    print(f"    میانگین توافق جفتی روی کل test: {a_mean:.3f} (95% CI: {a_lo:.3f}–{a_hi:.3f})")

    mat = agreement_matrix(attr)
    import pandas as pd
    pd.DataFrame(mat, index=METHODS, columns=METHODS).round(3).to_csv(
        os.path.join(XAI_DIR, "agreement_matrix.csv"))
    print("    ماتریس توافق ذخیره شد.")

    print("\n  عدم‌قطعیت MC-Dropout (۵۰ پاس)...")
    unc = mc_dropout(model, X)
    prob = unc["mean"]
    pred = (prob >= 0.5).astype(int)
    correct = (pred == y).astype(int)

    # ── آزمون فرضیهٔ اصلی ──
    print("\n  ── آزمون فرضیه: توافق در برابر عدم‌قطعیت و صحت ──")
    valid = ~np.isnan(agr)
    r_unc, p_unc = spearmanr(agr[valid], unc["std"][valid])
    r_ent, p_ent = spearmanr(agr[valid], unc["entropy"][valid])
    r_cor, p_cor = pointbiserialr(correct[valid], agr[valid])
    agr_correct = agr[valid & (correct == 1)]
    agr_wrong = agr[valid & (correct == 0)]
    print(f"    توافق ↔ عدم‌قطعیت(std)    : Spearman ρ={r_unc:+.3f} (p={p_unc:.3g})")
    print(f"    توافق ↔ آنتروپی          : Spearman ρ={r_ent:+.3f} (p={p_ent:.3g})")
    print(f"    توافق ↔ صحت              : point-biserial r={r_cor:+.3f} (p={p_cor:.3g})")
    print(f"    میانگین توافق (درست‌ها)   : {np.nanmean(agr_correct):.3f}  (n={len(agr_correct)})")
    print(f"    میانگین توافق (غلط‌ها)    : {np.nanmean(agr_wrong):.3f}  (n={len(agr_wrong)})")

    # ── ذخیرهٔ per-patient ──
    patients = []
    for i in range(len(y)):
        patients.append({
            "pid": str(pid[i]),
            "y_true": int(y[i]),
            "prob": float(prob[i]),
            "pred": int(pred[i]),
            "correct": int(correct[i]),
            "uncertainty_std": float(unc["std"][i]),
            "entropy": float(unc["entropy"][i]),
            "xai_agreement": float(agr[i]) if not np.isnan(agr[i]) else None,
            "attributions": {m: attr[m][i].tolist() for m in METHODS},
        })
    with open(os.path.join(XAI_DIR, "per_patient_xai.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": feats, "methods": METHODS, "patients": patients},
                  f, indent=2, ensure_ascii=False)

    hyp = {
        "mean_pairwise_agreement": a_mean,
        "agreement_ci": [a_lo, a_hi],
        "agreement_vs_uncertainty_std": {"rho": float(r_unc), "p": float(p_unc)},
        "agreement_vs_entropy": {"rho": float(r_ent), "p": float(p_ent)},
        "agreement_vs_correctness": {"r": float(r_cor), "p": float(p_cor)},
        "mean_agreement_correct": float(np.nanmean(agr_correct)),
        "mean_agreement_wrong": float(np.nanmean(agr_wrong)),
        "agreement_matrix_methods": METHODS,
    }
    with open(os.path.join(XAI_DIR, "hypothesis.json"), "w", encoding="utf-8") as f:
        json.dump(hyp, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ ذخیره شد در {XAI_DIR}/ "
          f"(per_patient_xai.json, agreement_matrix.csv, hypothesis.json)")
    return hyp


if __name__ == "__main__":
    run()
