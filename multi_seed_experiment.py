"""
multi_seed_experiment.py — File 8/N
==================================================================
اجراکنندهٔ چند-seed: کل آزمایش (split تازه + آموزش GRU-D + متریک‌ها +
توافق XAI + عدم‌قطعیت MC-Dropout + همبستگی‌ها) را K بار با seedهای
متفاوت تکرار می‌کند و هر عدد را با میانگین ± فاصلهٔ اطمینان گزارش می‌دهد.

چرا حیاتی است: در فایل ۵ دیدیم رابطهٔ «توافق ↔ صحت» روی یک مدل مرزی و
روی مدلی دیگر معنادار بود. این فایل تعیین می‌کند کدام یافته‌ها در همهٔ
seedها پایدارند (ادعای محکم) و کدام ناپایدار (باید محتاطانه گزارش شوند).

هر seed: split با random_state متفاوت → مدلِ تازه → ارزیابی کاملِ مستقل.

اجرا:
    python multi_seed_experiment.py data\\integrated_data.csv 3     # تست سریع
    python multi_seed_experiment.py data\\integrated_data.csv 10    # اجرای کامل (شب)

خروجی: results/multiseed_summary.json
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr, pointbiserialr, t as student_t

from oasis_prognosis_data import build_prognosis_dataset, patient_level_split
from grud_model import fit_obs_standardizer, make_grud_inputs, train_one
from lgbm_baseline import binary_metrics
from xai_grud import (
    METHODS, saliency, smoothgrad, grad_input, integrated_gradients, occlusion,
    pairwise_agreement, mc_dropout, predict_prob,
)

RESULTS_DIR = "results"
BASE_SEED = 42


def _attributions(model, X, F) -> Dict[str, np.ndarray]:
    return {
        "saliency": saliency(model, X, F),
        "smoothgrad": smoothgrad(model, X, F),
        "grad_input": grad_input(model, X, F),
        "integrated_gradients": integrated_gradients(model, X, F),
        "occlusion": occlusion(model, X, F),
    }


def run_one_seed(ds, seed: int) -> Dict[str, float]:
    """یک آزمایش کاملِ مستقل با یک seed."""
    tr, te = patient_level_split(ds, test_size=0.3, random_state=seed)
    T, F = tr.values.shape[1], tr.values.shape[2]

    mean, std = fit_obs_standardizer(tr.values, tr.mask)
    Xtr = make_grud_inputs(tr.values, tr.mask, tr.delta, mean, std)
    Xte = make_grud_inputs(te.values, te.mask, te.delta, mean, std)

    model = train_one(Xtr, tr.y, T, F, seed=seed)

    # عدم‌قطعیت MC-Dropout → mean به‌عنوان احتمال
    unc = mc_dropout(model, Xte, n=50, seed=seed)
    prob = unc["mean"]
    m = binary_metrics(te.y, prob)

    # توافق XAI
    attr = _attributions(model, Xte, F)
    agr = pairwise_agreement(attr)
    valid = ~np.isnan(agr)

    pred = (prob >= 0.5).astype(int)
    correct = (pred == te.y).astype(int)

    r_unc, p_unc = spearmanr(agr[valid], unc["std"][valid])
    r_ent, p_ent = spearmanr(agr[valid], unc["entropy"][valid])
    r_cor, p_cor = pointbiserialr(correct[valid], agr[valid])

    return {
        "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
        "sensitivity": m["sensitivity"], "specificity": m["specificity"],
        "mean_agreement": float(np.nanmean(agr)),
        "rho_agr_unc": float(r_unc), "p_agr_unc": float(p_unc),
        "rho_agr_ent": float(r_ent), "p_agr_ent": float(p_ent),
        "r_agr_correct": float(r_cor), "p_agr_correct": float(p_cor),
    }


def aggregate(rows: List[Dict[str, float]]) -> Dict:
    keys = list(rows[0].keys())
    K = len(rows)
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if K > 1 else 0.0
        if K > 1:
            half = student_t.ppf(0.975, K - 1) * sd / np.sqrt(K)
        else:
            half = 0.0
        agg[k] = {"mean": mean, "std": sd,
                  "ci95": [mean - half, mean + half],
                  "values": vals.tolist()}
    # شمارش seedهایی که هر همبستگی در آن‌ها معنادار بود (p<0.05)
    agg["_significance_counts"] = {
        "agr_unc_sig": int(sum(r["p_agr_unc"] < 0.05 for r in rows)),
        "agr_ent_sig": int(sum(r["p_agr_ent"] < 0.05 for r in rows)),
        "agr_correct_sig": int(sum(r["p_agr_correct"] < 0.05 for r in rows)),
        "K": K,
    }
    return agg


def run(csv_path: str, K: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ds = build_prognosis_dataset(csv_path, verbose=False)
    print(f"  دیتاست: N={len(ds)}  | اجرای {K} seed\n")

    rows = []
    for s in range(K):
        seed = BASE_SEED + s
        print(f"── seed {s+1}/{K} (random_state={seed}) ──")
        r = run_one_seed(ds, seed)
        rows.append(r)
        print(f"    AUC={r['roc_auc']:.3f}  PR-AUC={r['pr_auc']:.3f}  "
              f"sens={r['sensitivity']:.3f}  توافق={r['mean_agreement']:.3f}")
        print(f"    توافق↔عدم‌قطعیت ρ={r['rho_agr_unc']:+.3f} (p={r['p_agr_unc']:.3g})  |  "
              f"توافق↔صحت r={r['r_agr_correct']:+.3f} (p={r['p_agr_correct']:.3g})")

    agg = aggregate(rows)

    def show(key, label, fmt="{:+.3f}"):
        a = agg[key]
        lo, hi = a["ci95"]
        print(f"    {label:30s}: {fmt.format(a['mean'])} ± {a['std']:.3f}  "
              f"(95% CI: {fmt.format(lo)}–{fmt.format(hi)})")

    print("\n" + "=" * 64)
    print(f"  جمع‌بندیِ {K} seed (میانگین ± CI)")
    print("=" * 64)
    show("roc_auc", "ROC-AUC", "{:.3f}")
    show("pr_auc", "PR-AUC", "{:.3f}")
    show("sensitivity", "Sensitivity", "{:.3f}")
    show("specificity", "Specificity", "{:.3f}")
    show("mean_agreement", "میانگین توافق XAI", "{:.3f}")
    show("rho_agr_unc", "توافق ↔ عدم‌قطعیت (ρ)")
    show("rho_agr_ent", "توافق ↔ آنتروپی (ρ)")
    show("r_agr_correct", "توافق ↔ صحت (r)")

    sc = agg["_significance_counts"]
    print(f"\n  پایداریِ معناداری (تعداد seed با p<0.05 از {K}):")
    print(f"    توافق↔عدم‌قطعیت : {sc['agr_unc_sig']}/{K}")
    print(f"    توافق↔آنتروپی  : {sc['agr_ent_sig']}/{K}")
    print(f"    توافق↔صحت      : {sc['agr_correct_sig']}/{K}")

    out = {"K": K, "per_seed": rows, "aggregate": agg}
    with open(os.path.join(RESULTS_DIR, "multiseed_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ ذخیره شد: {RESULTS_DIR}/multiseed_summary.json")
    return out


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(csv, K)
