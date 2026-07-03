"""
faithfulness_grud.py — File 7/N
==================================================================
سنجش وفاداریِ (faithfulness) پنج روش XAI روی GRU-D.

پرسش: آیا ویژگی‌هایی که هر روش «مهم» می‌داند، واقعاً روی پیش‌بینیِ
مدل اثر دارند؟ این مستقل از sanity است — یک روش می‌تواند sanity را
رد کند ولی هنوز رفتارِ پیش‌بینی را خوب توضیح دهد (و این تضاد، خودش
یک نکتهٔ علمیِ ظریف است).

پنج معیار (هم‌راستا با T1–T5 فایل شما):
  1. Deletion AOPC      حذف تدریجیِ ویژگی‌های مهم → افتِ سریع‌ترِ پیش‌بینی بهتر است.
  2. Insertion AOPC     افزودن تدریجیِ ویژگی‌های مهم از baseline → رشدِ سریع‌تر بهتر.
  3. Comprehensiveness  حذف k٪ مهم → افتِ بزرگ‌ترِ پیش‌بینی بهتر است.
  4. Sufficiency        نگه‌داشتنِ فقط k٪ مهم → حفظِ بیشترِ پیش‌بینی بهتر (افتِ کمتر).
  5. Faithfulness corr  همبستگیِ انتساب با اثرِ واقعیِ حذفِ تک‌ویژگی → بالاتر بهتر.

baseline حذف = صفر (داده استاندارد است؛ صفر = میانگین).

ورودی: results/artifacts/grud_model.keras , grud_test_data.npz
خروجی: results/xai/faithfulness.json
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

from grud_model import GRUDCell  # noqa: F401
from xai_grud import load_assets, predict_prob, compute_all_methods, METHODS

XAI_DIR = os.path.join("results", "xai")
TOPK_FRAC = 0.20      # برای comprehensiveness / sufficiency


# ==================================================================
# توابع perturbation (روی F کانال اولِ ورودی = مقادیر)
# ==================================================================

def remove_features(X, F, feats_per_patient):
    """برای هر بیمار، ویژگی‌های داده‌شده را صفر کن."""
    Xp = X.copy()
    for i, fs in enumerate(feats_per_patient):
        if len(fs):
            Xp[i, :, list(fs)] = 0.0
    return Xp


def keep_only_features(X, F, feats_per_patient):
    """برای هر بیمار، همهٔ مقادیر را صفر کن جز ویژگی‌های داده‌شده."""
    Xp = X.copy()
    Xp[:, :, :F] = 0.0
    for i, fs in enumerate(feats_per_patient):
        if len(fs):
            Xp[i, :, list(fs)] = X[i, :, list(fs)]
    return Xp


# ==================================================================
# معیارها برای یک روش
# ==================================================================

def deletion_aopc(model, X, F, order, p_full):
    """حذف تدریجیِ مهم→کم‌اهمیت؛ AOPC = میانگینِ افتِ پیش‌بینی."""
    N = X.shape[0]
    drops = np.zeros((N, F))
    for j in range(1, F + 1):
        feats = [order[i, :j] for i in range(N)]
        p = predict_prob(model, remove_features(X, F, feats))
        drops[:, j - 1] = p_full - p
    return float(drops.mean())


def insertion_aopc(model, X, F, order, p_full):
    """افزودن تدریجیِ مهم→کم‌اهمیت از baseline؛ AOPC = میانگینِ رشدِ پیش‌بینی."""
    N = X.shape[0]
    base = keep_only_features(X, F, [[] for _ in range(N)])
    p_base = predict_prob(model, base)
    rises = np.zeros((N, F))
    for j in range(1, F + 1):
        feats = [order[i, :j] for i in range(N)]
        p = predict_prob(model, keep_only_features(X, F, feats))
        rises[:, j - 1] = p - p_base
    return float(rises.mean())


def comprehensiveness(model, X, F, order, p_full, k):
    feats = [order[i, :k] for i in range(X.shape[0])]
    p = predict_prob(model, remove_features(X, F, feats))
    return float(np.mean(p_full - p))         # بزرگ‌تر بهتر


def sufficiency(model, X, F, order, p_full, k):
    feats = [order[i, :k] for i in range(X.shape[0])]
    p = predict_prob(model, keep_only_features(X, F, feats))
    return float(np.mean(p_full - p))         # کوچک‌تر بهتر (افت کم)


def faithfulness_corr(attr, delta_p):
    """همبستگیِ انتساب با اثرِ واقعیِ حذفِ تک‌ویژگی، میانگین روی بیماران.

    هشدارِ circularity: ground-truth (delta_p) با صفرکردنِ تک‌ویژگی ساخته می‌شود
    که *دقیقاً* همان تعریفِ occlusion است. بنابراین faithfulness_corr برای occlusion
    به‌صورتِ تحلیلی برابرِ ۱.۰ است (corr(x,x)) و یک یافته نیست. این متریک برای
    occlusion گزارش نمی‌شود (NaN) تا جدول گمراه‌کننده نباشد.
    """
    rhos = []
    for i in range(attr.shape[0]):
        r, _ = spearmanr(attr[i], delta_p[i])
        if not np.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


# روش‌هایی که ground-truthِ delta_p برایشان circular است و باید از F-corr کنار بروند.
FAITHCORR_EXCLUDE = {"occlusion"}


# ==================================================================
# اجرا
# ==================================================================

def run():
    os.makedirs(XAI_DIR, exist_ok=True)
    model, X, y, pid, feats, F = load_assets()
    print(f"  test: {X.shape}  ویژگی‌ها: {F}")
    p_full = predict_prob(model, X)

    print("\n  محاسبهٔ انتساب‌ها...")
    attr = compute_all_methods(model, X, F)

    # اثرِ واقعیِ حذفِ تک‌ویژگی (مشترک بین همهٔ روش‌ها) — یک‌بار محاسبه
    print("\n  محاسبهٔ اثرِ حذفِ تک‌ویژگی (Δp برای هر ویژگی)...")
    delta_p = np.zeros((X.shape[0], F))
    for f in range(F):
        Xo = X.copy(); Xo[:, :, f] = 0.0
        delta_p[:, f] = np.abs(p_full - predict_prob(model, Xo))

    k = max(1, int(round(TOPK_FRAC * F)))
    print(f"  k برای comprehensiveness/sufficiency: {k} ویژگی (top {int(TOPK_FRAC*100)}٪)\n")

    results = {}
    print(f"  {'روش':22s} {'Del-AOPC↑':>9s} {'Ins-AOPC↑':>9s} "
          f"{'Compr↑':>8s} {'Suff↓':>7s} {'F-corr↑':>8s}")
    for m in METHODS:
        order = np.argsort(-attr[m], axis=1)        # مهم→کم‌اهمیت
        d = deletion_aopc(model, X, F, order, p_full)
        ins = insertion_aopc(model, X, F, order, p_full)
        comp = comprehensiveness(model, X, F, order, p_full, k)
        suff = sufficiency(model, X, F, order, p_full, k)
        # occlusion را از F-corr کنار می‌گذاریم (circular با delta_p)
        fc = float("nan") if m in FAITHCORR_EXCLUDE else faithfulness_corr(attr[m], delta_p)
        results[m] = {"deletion_aopc": d, "insertion_aopc": ins,
                      "comprehensiveness": comp, "sufficiency": suff,
                      "faithfulness_corr": fc}
        fc_str = "  n/a" if np.isnan(fc) else f"{fc:>8.3f}"
        print(f"  {m:22s} {d:>9.3f} {ins:>9.3f} {comp:>8.3f} {suff:>7.3f} {fc_str}")

    # رتبه‌بندیِ ترکیبی (هرچه Del/Ins/Compr/F-corr بزرگ‌تر و Suff کوچک‌تر، بهتر)
    # F-corrِ NaN (occlusion) به‌صورت ۰ در نمره لحاظ می‌شود تا رتبه‌بندی نشکند.
    def score(r):
        fc = r["faithfulness_corr"]
        fc = 0.0 if (fc is None or np.isnan(fc)) else fc
        return (r["deletion_aopc"] + r["insertion_aopc"] + r["comprehensiveness"]
                - r["sufficiency"] + fc)
    ranked = sorted(METHODS, key=lambda m: score(results[m]), reverse=True)
    print(f"\n  رتبه‌بندیِ وفاداری (از بهترین): {ranked}")

    out = {"topk_frac": TOPK_FRAC, "k": k, "per_method": results,
           "faithfulness_ranking": ranked,
           "faithcorr_excluded": sorted(FAITHCORR_EXCLUDE),
           "note": ("Del/Ins/Compr/F-corr: بزرگ‌تر بهتر؛ Suff: کوچک‌تر بهتر. "
                    "F-corr برای occlusion گزارش نمی‌شود (circular با ground-truth).")}
    with open(os.path.join(XAI_DIR, "faithfulness.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ ذخیره شد: {XAI_DIR}/faithfulness.json")
    return out


if __name__ == "__main__":
    run()