"""
lgbm_baseline.py — File 2/N
==================================================================
Baseline قویِ LightGBM برای مسئلهٔ «پیشرفت به دمانس».

چرا این فایل حیاتی است:
  روی دادهٔ بالینیِ کوچک و جدولی، gradient boosting اغلب از شبکه‌های
  عمیق بهتر عمل می‌کند. بدون این baseline، هر ادعای برتریِ مدل عمیق
  بی‌اعتبار است و داور رد می‌کند. این فایل عددِ مرجع را می‌سازد که
  GRU-D و بقیه باید از آن جلو بزنند.

روش:
  - هر بیمار → یک بردار ویژگیِ تجمیع‌شده (last / mean / slope / نرخ گمشدگی).
  - LightGBM گمشدگی (NaN) را بومی هندل می‌کند → بدون imputation، بدون نشتی.
  - ارزیابی سخت‌گیرانه:
      * Repeated Stratified K-Fold روی train (میانگین ± انحراف).
      * ارزیابی نهایی روی test کنارگذاشته‌شده با فاصلهٔ اطمینان bootstrap.
  - متریک‌ها متناسب با عدم‌توازن: ROC-AUC, PR-AUC, balanced acc,
    F1, MCC, sensitivity, specificity.

این فایل از oasis_prognosis_data.py (فایل ۱) استفاده می‌کند و باید
کنار آن در ریشهٔ پروژه باشد.

خروجی‌ها (در results/):
  - lgbm_cv_results.json     نتایج CV و test
  - lgbm_feature_importance.csv
  - lgbm_model.txt           مدل ذخیره‌شده
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, recall_score, confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from oasis_prognosis_data import (
    build_prognosis_dataset, patient_level_split, PrognosisDataset,
)

RANDOM_STATE = 42
RESULTS_DIR = "results"


# ==================================================================
# ساخت ویژگی‌های تجمیع‌شدهٔ هر بیمار (با حفظ NaN برای LightGBM)
# ==================================================================

def build_tabular_features(ds: PrognosisDataset
                           ) -> Tuple[np.ndarray, List[str]]:
    """
    برای هر بیمار و هر ویژگی f، چهار آماره از ویزیت‌های *مشاهده‌شده* می‌سازد:
      last_f   : آخرین مقدار مشاهده‌شده      (NaN اگر هرگز مشاهده نشده)
      mean_f   : میانگین مقادیر مشاهده‌شده    (NaN اگر هرگز)
      slope_f  : آخرین − اولین مشاهده‌شده     (0 اگر فقط یک مشاهده، NaN اگر هیچ)
      miss_f   : نرخ گمشدگی روی ویزیت‌های معتبر (0..1) — این هرگز NaN نیست
    به‌علاوهٔ seqlen به‌عنوان یک ویژگیِ بیمار.

    NaN عمداً حفظ می‌شود؛ LightGBM آن را بومی مدیریت می‌کند (بدون نشتی).
    """
    N, T, F = ds.values.shape
    names = ds.feature_names
    feats = []
    col_names: List[str] = []

    for f in range(F):
        col_names += [f"{names[f]}__last", f"{names[f]}__mean",
                      f"{names[f]}__slope", f"{names[f]}__missrate"]
    col_names.append("__seqlen")

    for i in range(N):
        sl = int(ds.seqlen[i])
        # timestepهای معتبر در انتهای توالی قرار دارند (pre-padding)
        valid_t = np.arange(T - sl, T)
        row = []
        for f in range(F):
            vals = ds.values[i, valid_t, f]
            m = ds.mask[i, valid_t, f].astype(bool)
            obs = vals[m]
            if obs.size == 0:
                last_v = np.nan; mean_v = np.nan; slope_v = np.nan
            elif obs.size == 1:
                last_v = obs[-1]; mean_v = obs[0]; slope_v = 0.0
            else:
                last_v = obs[-1]; mean_v = float(np.mean(obs))
                slope_v = float(obs[-1] - obs[0])
            miss_v = 1.0 - (obs.size / max(sl, 1))
            row += [last_v, mean_v, slope_v, miss_v]
        row.append(float(sl))
        feats.append(row)

    return np.array(feats, dtype=np.float64), col_names


# ==================================================================
# متریک‌ها
# ==================================================================

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """متریک‌های مناسب کلاس نامتوازن برای مسئلهٔ دودویی."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),  # AP کلاس مثبت
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
    }


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 metric: str = "roc_auc", n_boot: int = 1000,
                 seed: int = 42) -> Tuple[float, float, float]:
    """فاصلهٔ اطمینان ۹۵٪ bootstrap برای یک متریک threshold-free."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    fn = roc_auc_score if metric == "roc_auc" else average_precision_score
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:   # هر دو کلاس لازم است
            continue
        vals.append(fn(y_true[idx], y_prob[idx]))
    vals = np.array(vals)
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


# ==================================================================
# مدل
# ==================================================================

def make_lgbm(scale_pos_weight: float):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,   # جبران عدم‌توازن
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


# ==================================================================
# ارزیابی repeated CV روی train
# ==================================================================

def repeated_cv(X: pd.DataFrame, y: np.ndarray,
                n_splits: int = 5, n_repeats: int = 5) -> Dict:
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE)
    rows = []
    for tr, va in rskf.split(X, y):
        spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        clf = make_lgbm(spw)
        clf.fit(X.iloc[tr], y[tr])
        prob = clf.predict_proba(X.iloc[va])[:, 1]
        rows.append(binary_metrics(y[va], prob))
    df = pd.DataFrame(rows)
    summary = {f"{k}_mean": float(df[k].mean()) for k in df.columns}
    summary.update({f"{k}_std": float(df[k].std()) for k in df.columns})
    return {"per_fold": rows, "summary": summary,
            "n_splits": n_splits, "n_repeats": n_repeats}


# ==================================================================
# اجرا
# ==================================================================

def run(csv_path: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds = build_prognosis_dataset(csv_path)
    tr, te = patient_level_split(ds, test_size=0.3, random_state=RANDOM_STATE)

    X_tr, col_names = build_tabular_features(tr)
    X_te, _ = build_tabular_features(te)
    X_tr = pd.DataFrame(X_tr, columns=col_names)
    X_te = pd.DataFrame(X_te, columns=col_names)
    y_tr, y_te = tr.y, te.y

    print(f"\n  ویژگی‌های تجمیع‌شده: {X_tr.shape[1]} ستون "
          f"(۴ آماره × {len(ds.feature_names)} + seqlen)")
    print(f"  NaN در X_train: {int(np.isnan(X_tr.values).sum())} "
          f"(عمدی — LightGBM بومی هندل می‌کند)")

    # ── Repeated CV روی train ──
    print("\n── Repeated Stratified 5-Fold × 5 روی train ──")
    cv = repeated_cv(X_tr, y_tr)
    s = cv["summary"]
    for k in ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc",
              "sensitivity", "specificity"]:
        print(f"    {k:18s}: {s[k+'_mean']:.3f} ± {s[k+'_std']:.3f}")

    # ── آموزش روی کل train، ارزیابی روی test ──
    print("\n── ارزیابی نهایی روی test کنارگذاشته‌شده ──")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = make_lgbm(spw)
    clf.fit(X_tr, y_tr)
    prob_te = clf.predict_proba(X_te)[:, 1]
    test_m = binary_metrics(y_te, prob_te)

    auc_m, auc_lo, auc_hi = bootstrap_ci(y_te, prob_te, "roc_auc")
    ap_m, ap_lo, ap_hi = bootstrap_ci(y_te, prob_te, "pr_auc")
    print(f"    ROC-AUC : {test_m['roc_auc']:.3f}  "
          f"(bootstrap 95% CI: {auc_lo:.3f}–{auc_hi:.3f})")
    print(f"    PR-AUC  : {test_m['pr_auc']:.3f}  "
          f"(bootstrap 95% CI: {ap_lo:.3f}–{ap_hi:.3f})")
    for k in ["balanced_accuracy", "f1", "mcc", "sensitivity", "specificity"]:
        print(f"    {k:18s}: {test_m[k]:.3f}")

    # ── اهمیت ویژگی‌ها (gain) ──
    imp = pd.DataFrame({
        "feature": col_names,
        "gain_importance": clf.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain_importance", ascending=False)
    imp.to_csv(os.path.join(RESULTS_DIR, "lgbm_feature_importance.csv"),
               index=False)
    print("\n  Top-8 ویژگی (gain):")
    for _, r in imp.head(8).iterrows():
        print(f"    {r['feature']:28s}: {r['gain_importance']:.0f}")

    # ── ذخیره ──
    clf.booster_.save_model(os.path.join(RESULTS_DIR, "lgbm_model.txt"))
    out = {
        "cv": cv["summary"],
        "test": test_m,
        "test_ci": {
            "roc_auc": [auc_lo, auc_hi],
            "pr_auc": [ap_lo, ap_hi],
        },
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "n_features": X_tr.shape[1],
        "positive_rate": float(ds.y.mean()),
    }
    with open(os.path.join(RESULTS_DIR, "lgbm_cv_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ ذخیره شد در {RESULTS_DIR}/  "
          f"(lgbm_cv_results.json, lgbm_feature_importance.csv, lgbm_model.txt)")
    return out


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    run(csv)
