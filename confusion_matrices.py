"""
confusion_matrices.py — File 14/N
==================================================================
ماتریسِ درهم‌ریختگی (confusion matrix) برای هر سه مدل روی *همان* test set.

چرا این فایل ارزش دارد:
  ROC-AUC/PR-AUC معیارهای اصلی و threshold-free‌اند، اما داورِ بالینی
  می‌خواهد تعدادِ خامِ TP/FP/FN/TN را هم ببیند — به‌ویژه با کلاسِ مثبتِ
  کوچک (~۱۸.۷٪). این فایل confusion matrix را در آستانهٔ ۰.۵ (که صریحاً
  دلخواه است و اعلام می‌شود) برای هر سه مدل می‌سازد، تا مقایسه در یک نگاه
  ممکن باشد.

نکتهٔ انصاف: هر سه مدل از patient_level_split(test_size=0.3, random_state=42)
استفاده می‌کنند، یعنی *دقیقاً همان بیماران و همان برچسب‌ها* در test —
پس confusion matrixها مستقیماً قابل‌مقایسه‌اند.

آستانه: 0.5 (ثابت، دلخواه). چون معیارهای اصلی threshold-free‌اند، این
شکل مکمل است، نه معیارِ اصلی.

اجرا:
    python confusion_matrices.py data\\integrated_data.csv

خروجی:
    results/confusion_matrices.json
    results/figures/fig10_confusion_matrices.png
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oasis_prognosis_data import (
    build_prognosis_dataset, patient_level_split, LeakFreeImputerScaler,
)
from lgbm_baseline import build_tabular_features, make_lgbm
from grud_model import (
    fit_obs_standardizer, make_grud_inputs, train_one as train_grud,
)
import transformer_ablation as tfm

RANDOM_STATE = 42
RESULTS = "results"
FIG = os.path.join(RESULTS, "figures")
THRESHOLD = 0.5

# رنگِ هر مدل، هم‌راستا با fig1 پروژه
MODEL_COLOR = {"LightGBM": "#2c6fbb", "GRU-D": "#d1495b", "Transformer": "#7a8450"}


def cm_stats(y_true, y_prob, thr=THRESHOLD):
    """confusion matrix + معیارهای مشتق‌شده در آستانهٔ داده‌شده."""
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0     # recall/sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0     # specificity
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0      # precision/PPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv), "threshold": float(thr)}


# ==================================================================
# احتمال‌های هر مدل روی همان test set
# ==================================================================

def get_probs(csv_path):
    ds = build_prognosis_dataset(csv_path, verbose=False)
    tr, te = patient_level_split(ds, test_size=0.3, random_state=RANDOM_STATE)
    T, F = tr.values.shape[1], tr.values.shape[2]
    y_te = te.y
    probs = {}

    # ── LightGBM ──
    print("  آموزش LightGBM...")
    Xtr_t, cols = build_tabular_features(tr)
    Xte_t, _ = build_tabular_features(te)
    Xtr_t = pd.DataFrame(Xtr_t, columns=cols)
    Xte_t = pd.DataFrame(Xte_t, columns=cols)
    spw = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
    clf = make_lgbm(spw); clf.fit(Xtr_t, tr.y)
    probs["LightGBM"] = clf.predict_proba(Xte_t)[:, 1]

    # ── GRU-D ──
    print("  آموزش GRU-D...")
    mean, std = fit_obs_standardizer(tr.values, tr.mask)
    Xtr_g = make_grud_inputs(tr.values, tr.mask, tr.delta, mean, std)
    Xte_g = make_grud_inputs(te.values, te.mask, te.delta, mean, std)
    grud = train_grud(Xtr_g, tr.y, T, F, seed=RANDOM_STATE)
    probs["GRU-D"] = grud.predict(Xte_g, verbose=0).ravel()

    # ── Transformer ──
    print("  آموزش Transformer...")
    imp = LeakFreeImputerScaler().fit(tr.values, tr.mask)
    Xtr_f = imp.transform(tr.values, tr.mask)
    Xte_f = imp.transform(te.values, te.mask)
    Mtr = (tr.mask.sum(axis=2) > 0).astype(np.float32)
    Mte = (te.mask.sum(axis=2) > 0).astype(np.float32)
    tf_model = tfm.train_one(Xtr_f, Mtr, tr.y, T, F, seed=RANDOM_STATE)
    probs["Transformer"] = tf_model.predict([Xte_f, Mte], verbose=0).ravel()

    return y_te, probs


# ==================================================================
# رسم
# ==================================================================

def draw_panel(ax, cm, title, color):
    """یک پنلِ 2×2 برای یک مدل."""
    M = np.array([[cm["tn"], cm["fp"]],
                  [cm["fn"], cm["tp"]]], dtype=float)
    # نرمال‌سازیِ سطری برای ته‌رنگ (per true-class rate)
    row = M.sum(axis=1, keepdims=True)
    Mn = np.divide(M, row, out=np.zeros_like(M), where=row > 0)

    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0\n(stable)", "Pred 1\n(progress)"], fontsize=8.5)
    ax.set_yticklabels(["True 0\n(stable)", "True 1\n(progress)"], fontsize=8.5)
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for r in range(2):
        for c in range(2):
            txt = f"{cell_labels[r][c]}\n{int(M[r, c])}\n({Mn[r, c]*100:.0f}%)"
            ax.text(c, r, txt, ha="center", va="center", fontsize=9,
                    color="white" if Mn[r, c] > 0.55 else "#1a1a1a",
                    fontweight="medium")
    ax.set_title(f"{title}\nSens {cm['sensitivity']:.2f} · Spec {cm['specificity']:.2f} · "
                 f"PPV {cm['ppv']:.2f}", fontsize=10, color=color)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def run(csv_path):
    os.makedirs(FIG, exist_ok=True)
    print(f"  ساخت confusion matrix برای سه مدل (آستانه={THRESHOLD})\n")
    y_te, probs = get_probs(csv_path)

    order = ["LightGBM", "GRU-D", "Transformer"]
    results = {}
    print("\n  " + "=" * 58)
    print(f"  {'مدل':12s} {'TN':>5s} {'FP':>5s} {'FN':>5s} {'TP':>5s}  "
          f"{'Sens':>5s} {'Spec':>5s} {'PPV':>5s}")
    for name in order:
        cm = cm_stats(y_te, probs[name])
        results[name] = cm
        print(f"  {name:12s} {cm['tn']:>5d} {cm['fp']:>5d} {cm['fn']:>5d} "
              f"{cm['tp']:>5d}  {cm['sensitivity']:>5.2f} {cm['specificity']:>5.2f} "
              f"{cm['ppv']:>5.2f}")

    n_pos = int(y_te.sum()); n_neg = int(len(y_te) - n_pos)
    meta = {"threshold": THRESHOLD, "n_test": int(len(y_te)),
            "n_positive": n_pos, "n_negative": n_neg,
            "note": ("confusion matrix در آستانهٔ ثابتِ 0.5 (دلخواه). "
                     "معیارهای اصلیِ مقاله threshold-free‌اند (ROC/PR-AUC). "
                     "هر سه مدل روی همان test split ارزیابی شده‌اند.")}
    out = {"meta": meta, "per_model": results}
    with open(os.path.join(RESULTS, "confusion_matrices.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ── شکلِ سه‌پنلی ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, name in zip(axes, order):
        draw_panel(ax, results[name], name, MODEL_COLOR[name])
    fig.suptitle(f"Confusion matrices on the shared test set "
                 f"(N={len(y_te)}: {n_pos} progressors, {n_neg} stable; threshold=0.5)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_confusion_matrices.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✅ ذخیره شد: {RESULTS}/confusion_matrices.json , "
          f"{FIG}/fig10_confusion_matrices.png")
    return out


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    run(csv)