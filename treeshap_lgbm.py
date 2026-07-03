"""
treeshap_lgbm.py — File 9/N
==================================================================
TreeSHAP روی LightGBM (بهترین مدل) + مقایسهٔ بین‌مدلی با GRU-D.

دو هدف:
  1. یک XAIِ دقیق و بومی برای LightGBM (TreeSHAP دقیق است، نه تقریبی).
  2. توافق بین‌مدلی: آیا بهترین مدل (GBM) و مدل عمیق (GRU-D) سرِ
     مهم‌ترین ویژگی‌ها هم‌نظرند؟ این یک اعتبارسنجیِ هم‌گراییِ مدل‌محور است.

نگاشت ویژگی: LightGBM روی ۴ آمارهٔ هر ویژگی (last/mean/slope/missrate)
کار می‌کند؛ برای مقایسه با GRU-D، |SHAP| این ۴ آماره را جمع می‌کنیم تا
یک اهمیتِ سطح-ویژگی (۲۲ بُعدی) به‌دست آید — قابل‌مقایسه با GRU-D.

ورودی: data/integrated_data.csv  و  results/xai/per_patient_xai.json (فایل ۵)
خروجی: results/xai/treeshap_lgbm.json , treeshap_feature_importance.csv
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oasis_prognosis_data import build_prognosis_dataset, patient_level_split
from lgbm_baseline import build_tabular_features, make_lgbm, RANDOM_STATE

XAI_DIR = os.path.join("results", "xai")


def base_feature_of(col: str) -> str:
    """'FAQ_Score__last' → 'FAQ_Score' ؛ '__seqlen' → '__seqlen'."""
    return col.rsplit("__", 1)[0] if "__" in col and not col.startswith("__") else col


def run(csv_path: str):
    os.makedirs(XAI_DIR, exist_ok=True)
    import shap

    ds = build_prognosis_dataset(csv_path, verbose=False)
    tr, te = patient_level_split(ds, test_size=0.3, random_state=RANDOM_STATE)
    X_tr, cols = build_tabular_features(tr)
    X_te, _ = build_tabular_features(te)
    X_tr = pd.DataFrame(X_tr, columns=cols)
    X_te = pd.DataFrame(X_te, columns=cols)

    spw = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
    clf = make_lgbm(spw)
    clf.fit(X_tr, tr.y)
    print(f"  LightGBM آموزش دید روی {len(tr.y)} بیمار، {X_tr.shape[1]} ویژگیِ تجمیعی.")

    # ── TreeSHAP روی test ──
    print("  محاسبهٔ TreeSHAP...")
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):          # برخی نسخه‌ها لیست برمی‌گردانند
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (N, F, classes)
        sv = sv[:, :, -1]
    mean_abs = np.abs(sv).mean(axis=0)        # (89,)

    # ── نگاشت ۸۹ ستون → ۲۲ ویژگیِ پایه ──
    base_imp = defaultdict(float)
    for c, v in zip(cols, mean_abs):
        base_imp[base_feature_of(c)] += float(v)

    feat_names = ds.feature_names
    lgbm_vec = np.array([base_imp.get(f, 0.0) for f in feat_names])  # (22,)

    # جدول مرتب‌شده
    imp_df = pd.DataFrame({"feature": feat_names, "treeshap_importance": lgbm_vec}
                          ).sort_values("treeshap_importance", ascending=False)
    imp_df.to_csv(os.path.join(XAI_DIR, "treeshap_feature_importance.csv"), index=False)
    print("\n  Top-8 ویژگی (TreeSHAP، سطح-ویژگی):")
    for _, r in imp_df.head(8).iterrows():
        print(f"    {r['feature']:16s}: {r['treeshap_importance']:.4f}")

    # ── مقایسهٔ بین‌مدلی با GRU-D ──
    cross = {}
    xai_path = os.path.join(XAI_DIR, "per_patient_xai.json")
    if os.path.exists(xai_path):
        with open(xai_path, encoding="utf-8") as f:
            gx = json.load(f)
        methods = gx["methods"]
        gfeats = gx["feature_names"]
        # میانگین انتساب هر روش روی بیماران → (22,)
        grud_imp = {m: np.zeros(len(gfeats)) for m in methods}
        for p in gx["patients"]:
            for m in methods:
                grud_imp[m] += np.array(p["attributions"][m])
        for m in methods:
            grud_imp[m] /= len(gx["patients"])

        # هم‌ترتیب‌سازیِ ویژگی‌ها
        idx = [gfeats.index(f) for f in feat_names]
        print("\n  توافقِ بین‌مدلی (Spearman): TreeSHAP(LightGBM) ↔ هر روشِ GRU-D")
        for m in methods:
            r, p = spearmanr(lgbm_vec, grud_imp[m][idx])
            cross[m] = {"rho": float(r), "p": float(p)}
            print(f"    {m:22s}: ρ={r:+.3f}  (p={p:.3g})")
    else:
        print("\n  (per_patient_xai.json پیدا نشد — اول فایل ۵ را اجرا کنید "
              "تا مقایسهٔ بین‌مدلی انجام شود.)")

    out = {
        "top_features": imp_df.head(10).to_dict(orient="records"),
        "treeshap_importance": {f: float(v) for f, v in zip(feat_names, lgbm_vec)},
        "cross_model_agreement_with_grud": cross,
    }
    with open(os.path.join(XAI_DIR, "treeshap_lgbm.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ ذخیره شد: {XAI_DIR}/treeshap_lgbm.json , treeshap_feature_importance.csv")
    return out


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    run(csv)
