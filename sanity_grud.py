"""
sanity_grud.py — File 6/N
==================================================================
Sanity check روش‌های XAI (Adebayo et al., 2018) برای GRU-D.

اصل: یک روشِ توضیحِ *معتبر* باید به وزن‌های مدل وابسته باشد. اگر وزن‌های
مدل را تصادفی کنیم و توضیح تقریباً همان بماند، آن روش در واقع مدل را
توضیح نمی‌دهد (مثل یک لبه‌یاب که فقط به ورودی نگاه می‌کند) و باید رد شود.

دو تست:
  1. Model parameter randomization (full):
     مدلِ آموزش‌دیده در برابر یک مدلِ کاملاً تصادفی (همان معماری، وزن
     تصادفی). برای هر روش، شباهتِ توضیح‌ها را اندازه می‌گیریم.
     شباهتِ پایین = روش *قبول می‌شود* (به مدل وابسته است).
  2. Cascading randomization (لایه به لایه):
     از سرِ خروجی به سمت پایین، تدریجی لایه‌ها را تصادفی می‌کنیم؛
     توضیح باید هرچه بیشتر تصادفی می‌کنیم، بیشتر تغییر کند.

فرمول‌بندیِ درست (مطابق نسخهٔ V3 شما): شباهت = میانگین Spearman بین
توضیحِ مدلِ اصلی و توضیحِ مدلِ تصادفی‌شده، روی بیماران. *پایین‌تر بهتر.*

ورودی: results/artifacts/grud_model.keras , grud_test_data.npz
خروجی: results/xai/sanity.json
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import keras
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from grud_model import (  # noqa: F401  (ثبت + ساخت معماری + آموزش)
    GRUDCell, build_model, fit_obs_standardizer, make_grud_inputs, train_one,
    RANDOM_STATE,
)
from oasis_prognosis_data import build_prognosis_dataset, patient_level_split
from xai_grud import (
    load_assets, predict_prob, METHODS,
    saliency, smoothgrad, grad_input, integrated_gradients, occlusion,
)

XAI_DIR = os.path.join("results", "xai")


def all_methods(model, X, F) -> Dict[str, np.ndarray]:
    return {
        "saliency": saliency(model, X, F),
        "smoothgrad": smoothgrad(model, X, F),
        "grad_input": grad_input(model, X, F),
        "integrated_gradients": integrated_gradients(model, X, F),
        "occlusion": occlusion(model, X, F),
    }


def mean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """میانگین Spearman بین دو مجموعه توضیح (هر ردیف یک بیمار)."""
    rhos = []
    for i in range(a.shape[0]):
        r, _ = spearmanr(a[i], b[i])
        if not np.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


def randomized_like(model, T, F, seed=0) -> keras.Model:
    """یک مدلِ هم‌معماری با وزن‌های تصادفیِ تازه."""
    keras.utils.set_random_seed(seed)
    return build_model(T, F)


def verdict(sim: float) -> str:
    if np.isnan(sim):
        return "?"
    if sim < 0.30:
        return "PASS (قوی)"
    if sim < 0.60:
        return "WEAK"
    return "FAIL"


def run(csv_path: str = os.path.join("data", "integrated_data.csv")):
    os.makedirs(XAI_DIR, exist_ok=True)
    model, X, y, pid, feats, F = load_assets()
    T = X.shape[1]
    print(f"  test: {X.shape}  ویژگی‌ها: {F}")

    # توضیح‌های مدلِ آموزش‌دیده
    print("\n  توضیح‌های مدل آموزش‌دیده...")
    orig = all_methods(model, X, F)
    auc_trained = roc_auc_score(y, predict_prob(model, X))

    # ── تست ۱: randomization کامل (میانگین روی چند seed) ──
    # چرا چند seed: یک مدلِ تک-تصادفی ممکن است تصادفاً AUC غیرِ۰.۵ بدهد و تست را
    # ناپایدار کند. با میانگین‌گیری روی چند وزنِ تصادفیِ مستقل، هم شباهت پایدار
    # می‌شود و هم AUCِ تصادفیِ گزارش‌شده به مقدارِ موردانتظار (~۰.۵) نزدیک‌تر و
    # قابل‌دفاع‌تر است.
    print("\n── تست ۱: Model parameter randomization (کامل، میانگینِ چند seed) ──")
    RAND_SEEDS = [123, 321, 777, 999, 2024]
    auc_rands = []
    sim_accum = {m: [] for m in METHODS}
    for s in RAND_SEEDS:
        rand = randomized_like(model, T, F, seed=s)
        auc_rands.append(roc_auc_score(y, predict_prob(rand, X)))
        rand_attr = all_methods(rand, X, F)
        for m in METHODS:
            sim_accum[m].append(mean_similarity(orig[m], rand_attr[m]))
    auc_rand = float(np.mean(auc_rands))
    auc_rand_std = float(np.std(auc_rands))
    print(f"    AUC مدل آموزش‌دیده: {auc_trained:.3f}   |   "
          f"AUC مدل تصادفی (میانگینِ {len(RAND_SEEDS)} seed): "
          f"{auc_rand:.3f} ± {auc_rand_std:.3f}")
    print(f"    (AUC تصادفی باید نزدیک ۰.۵ باشد تا تست معتبر باشد)\n")

    full_results = {}
    full_results_std = {}
    print(f"    {'روش':22s} {'شباهت(اصلی↔تصادفی)':>22s}   حکم")
    for m in METHODS:
        sim = float(np.mean(sim_accum[m]))
        sd = float(np.std(sim_accum[m]))
        full_results[m] = sim
        full_results_std[m] = sd
        print(f"    {m:22s} {sim:>14.3f} ± {sd:.3f}     {verdict(sim)}")

    # ── تست ۲: cascading (تصادفی‌کردن تدریجیِ لایه‌های بالا) ──
    print("\n── تست ۲: Cascading randomization (از سرِ خروجی به پایین) ──")
    # ترتیب لایه‌های دارای وزن، از خروجی به ورودی
    weighted = [l for l in model.layers if l.get_weights()]
    cascade_layers = list(reversed(weighted))
    casc = keras.models.clone_model(model)
    casc.set_weights(model.get_weights())
    cascade_results = {m: [] for m in METHODS}
    stage_names = []
    rng = np.random.default_rng(7)
    for li, layer in enumerate(cascade_layers):
        # وزن‌های این لایه را با مقادیر تصادفی (هم‌شکل) جایگزین کن
        target = casc.layers[[l.name for l in casc.layers].index(layer.name)]
        new_w = [rng.normal(0, 0.1, w.shape).astype(w.dtype) for w in target.get_weights()]
        target.set_weights(new_w)
        stage_names.append(layer.name)
        attr_c = all_methods(casc, X, F)
        for m in METHODS:
            cascade_results[m].append(mean_similarity(orig[m], attr_c[m]))
    print(f"    مراحل (تجمعی): {stage_names}")
    for m in METHODS:
        seq = " → ".join(f"{v:.2f}" for v in cascade_results[m])
        print(f"    {m:22s}: {seq}")
    print("    (انتظار: با تصادفی‌شدن بیشتر، شباهت باید کاهش یابد)")

    # ── تست ۳: data randomization (آموزش روی برچسب‌های permute‌شده) ──
    # مطابق روش V3 شما (Test 8): مدلی که واقعاً «هیچ» یاد نگرفته، ولی trained است.
    print("\n── تست ۳: Data randomization (مدلِ آموزش‌دیده روی برچسب‌های permute) ──")
    ds = build_prognosis_dataset(csv_path, verbose=False)
    tr, te = patient_level_split(ds, test_size=0.3, random_state=RANDOM_STATE)
    mean, std = fit_obs_standardizer(tr.values, tr.mask)
    Xtr_full = make_grud_inputs(tr.values, tr.mask, tr.delta, mean, std)
    rng = np.random.default_rng(42)
    y_perm = rng.permutation(tr.y)
    print("    آموزش مدل روی برچسب‌های permute‌شده...")
    perm_model = train_one(Xtr_full, y_perm, T, F, seed=RANDOM_STATE)
    auc_perm = roc_auc_score(y, predict_prob(perm_model, X))   # روی برچسب‌های واقعی test
    print(f"    AUC مدلِ permute‌شده روی برچسب واقعی: {auc_perm:.3f} "
          f"(باید نزدیک ۰.۵ باشد → چیزی یاد نگرفته)")
    perm_attr = all_methods(perm_model, X, F)

    data_results = {}
    print(f"\n    {'روش':22s} {'شباهت(اصلی↔permute)':>20s}   حکم")
    for m in METHODS:
        sim = mean_similarity(orig[m], perm_attr[m])
        data_results[m] = sim
        print(f"    {m:22s} {sim:>18.3f}     {verdict(sim)}")
    print("    (شباهت پایین = روش به برچسب/مدل حساس است → معتبر)")

    out = {
        "auc_trained": float(auc_trained),
        "auc_randomized": float(auc_rand),
        "auc_randomized_std": float(auc_rand_std),
        "auc_randomized_seeds": [float(a) for a in auc_rands],
        "auc_permuted_label": float(auc_perm),
        "full_randomization_similarity": full_results,
        "full_randomization_similarity_std": full_results_std,
        "full_randomization_verdict": {m: verdict(full_results[m]) for m in METHODS},
        "data_randomization_similarity": data_results,
        "data_randomization_verdict": {m: verdict(data_results[m]) for m in METHODS},
        "cascading_stages": stage_names,
        "cascading_similarity": cascade_results,
        "note": ("شباهتِ پایین‌تر = روشِ معتبرتر (به مدل/برچسب وابسته است). "
                 "تست ۱ روی چند seed میانگین گرفته شده تا پایدار و معتبر باشد."),
    }
    with open(os.path.join(XAI_DIR, "sanity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ ذخیره شد: {XAI_DIR}/sanity.json")
    return out


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    run(csv)