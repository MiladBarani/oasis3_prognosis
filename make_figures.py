"""
make_figures.py — File 10/N
==================================================================
تولید نمودارهای چاپی (PNG، 300 DPI) از فایل‌های نتایج در results/.
برچسب‌ها انگلیسی‌اند (هم برای رندرِ درست، هم برای مقالهٔ بین‌المللی).

پیش‌نیازها (هر کدام نبود، آن شکل رد می‌شود):
  results/lgbm_cv_results.json, grud_results.json, transformer_results.json,
  results/multiseed_summary.json,
  results/xai/{agreement_matrix.csv, sanity.json, faithfulness.json,
              per_patient_xai.json, treeshap_feature_importance.csv,
              treeshap_lgbm.json},
  results/artifacts/{grud_model.keras, grud_test_data.npz}

خروجی: results/figures/fig1..fig8 .png
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding="utf-8")

RES = "results"
XAI = os.path.join(RES, "xai")
FIG = os.path.join(RES, "figures")
ART = os.path.join(RES, "artifacts")

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": True,
})
C = {"lgbm": "#2c6fbb", "grud": "#d1495b", "tfm": "#7a8450", "acc": "#3a3a3a"}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Fig 1: model comparison ───────────────────────────────────
def fig1_models():
    lg, gr, tf = _load(f"{RES}/lgbm_cv_results.json"), _load(f"{RES}/multiseed_summary.json"), _load(f"{RES}/transformer_results.json")
    if not (lg and gr and tf):
        print("  [fig1] رد شد (فایل ناقص)"); return
    g = gr["aggregate"]
    models = ["LightGBM", "GRU-D", "Transformer"]
    roc = [lg["test"]["roc_auc"], g["roc_auc"]["mean"], tf["test"]["roc_auc"]]
    roc_err = [
        [lg["test"]["roc_auc"] - lg["test_ci"]["roc_auc"][0], lg["test_ci"]["roc_auc"][1] - lg["test"]["roc_auc"]],
        [g["roc_auc"]["mean"] - g["roc_auc"]["ci95"][0], g["roc_auc"]["ci95"][1] - g["roc_auc"]["mean"]],
        [tf["test"]["roc_auc"] - tf["test_ci"]["roc_auc"][0], tf["test_ci"]["roc_auc"][1] - tf["test"]["roc_auc"]],
    ]
    pr = [lg["test"]["pr_auc"], g["pr_auc"]["mean"], tf["test"]["pr_auc"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3); w = 0.38
    cols = [C["lgbm"], C["grud"], C["tfm"]]
    ax.bar(x - w/2, roc, w, yerr=np.array(roc_err).T, capsize=4,
           color=cols, label="ROC-AUC", edgecolor="white")
    ax.bar(x + w/2, pr, w, color=cols, alpha=0.55, label="PR-AUC", edgecolor="white")
    for i, (r, p, err_hi) in enumerate(zip(roc, pr, [e[1] for e in roc_err])):
        ax.text(i - w/2, r + err_hi + 0.015, f"{r:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, p + 0.012, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score (test)")
    K = gr.get("K", len(g["roc_auc"]["values"]))
    ax.set_title(f"Model comparison — progression-to-dementia\n(GRU-D: mean±95%CI over {K} seeds)")
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(f"{FIG}/fig1_model_comparison.png"); plt.close(fig)
    print("  [fig1]  Comparison of models (ROC/PR-AUC)")


# ── Fig 2 & 8: نیاز به مدل (ROC/PR + calibration) ─────────────
def _grud_probs():
    import keras
    from grud_model import GRUDCell  # noqa
    from xai_grud import mc_dropout
    if not os.path.exists(f"{ART}/grud_model.keras"):
        return None
    model = keras.saving.load_model(f"{ART}/grud_model.keras")
    d = np.load(f"{ART}/grud_test_data.npz", allow_pickle=True)
    X, y = d["X_test"].astype("float32"), d["y_test"].astype(int)
    prob = mc_dropout(model, X, n=50)["mean"]
    return y, prob


def fig2_curves(y, prob):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    fpr, tpr, _ = roc_curve(y, prob)
    pre, rec, _ = precision_recall_curve(y, prob)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(fpr, tpr, color=C["grud"], lw=2,
                 label=f"AUC={roc_auc_score(y, prob):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC — GRU-D"); axes[0].legend(frameon=False)
    axes[1].plot(rec, pre, color=C["grud"], lw=2,
                 label=f"AP={average_precision_score(y, prob):.3f}")
    axes[1].axhline(y.mean(), ls="--", color="gray", lw=1, label=f"baseline={y.mean():.2f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall — GRU-D"); axes[1].legend(frameon=False)
    fig.savefig(f"{FIG}/fig2_roc_pr.png"); plt.close(fig)
    print("  [fig2] ✓ منحنی ROC/PR")


def fig8_calibration(y, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ece = [], [], 0.0
    for i in range(bins):
        m = (prob >= edges[i]) & (prob < edges[i + 1] if i < bins - 1 else prob <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf, acc = prob[m].mean(), y[m].mean()
        xs.append(conf); ys.append(acc); ece += (m.sum() / len(y)) * abs(acc - conf)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color=C["grud"], lw=2, label=f"GRU-D (ECE={ece:.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration / reliability — GRU-D"); ax.legend(frameon=False)
    fig.savefig(f"{FIG}/fig8_calibration.png"); plt.close(fig)
    print("  [fig8] ✓ کالیبراسیون")


# ── Fig 3: agreement matrix ───────────────────────────────────
def fig3_agreement():
    p = f"{XAI}/agreement_matrix.csv"
    if not os.path.exists(p):
        print("  [fig3] رد شد"); return
    df = pd.read_csv(p, index_col=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(df.values, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(df))); ax.set_yticks(range(len(df)))
    ax.set_xticklabels(df.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(df.index, fontsize=8)
    for i in range(len(df)):
        for j in range(len(df)):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if df.values[i, j] < 0.8 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Cross-method explanation agreement (GRU-D)")
    fig.savefig(f"{FIG}/fig3_agreement_matrix.png"); plt.close(fig)
    print("  [fig3] ✓ Confidence agreement matrix (Spearman) between XAI methods")


# ── Fig 4: sanity ─────────────────────────────────────────────
def fig4_sanity():
    s = _load(f"{XAI}/sanity.json")
    if not s:
        print("  [fig4] رد شد"); return
    methods = list(s["full_randomization_similarity"].keys())
    full = [s["full_randomization_similarity"][m] for m in methods]
    data = [s["data_randomization_similarity"][m] for m in methods]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(methods)); w = 0.38
    ax.bar(x - w/2, full, w, color="#444", label="Weight randomization", edgecolor="white")
    ax.bar(x + w/2, data, w, color="#999", label="Label randomization", edgecolor="white")
    ax.axhline(0.3, ls="--", color="green", lw=1, label="strong-pass threshold")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Similarity to original (lower = better)")
    ax.set_title("Sanity checks (Adebayo) — lower means more model-dependent")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(f"{FIG}/fig4_sanity.png"); plt.close(fig)
    print("  [fig4]  sanity")


# ── Fig 5: faithfulness ───────────────────────────────────────
def fig5_faithfulness():
    f = _load(f"{XAI}/faithfulness.json")
    if not f:
        print("  [fig5] رد شد"); return
    methods = list(f["per_method"].keys())
    metrics = ["deletion_aopc", "insertion_aopc", "comprehensiveness", "faithfulness_corr"]
    labels = ["Deletion", "Insertion", "Compreh.", "Faith-corr"]
    M = np.array([[f["per_method"][m][k] for k in metrics] for m in methods])
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(methods)); w = 0.2
    for j, lab in enumerate(labels):
        ax.bar(x + (j - 1.5) * w, M[:, j], w, label=lab, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score (higher = more faithful)")
    ax.set_title("Faithfulness of attribution methods (GRU-D)")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.savefig(f"{FIG}/fig5_faithfulness.png"); plt.close(fig)
    print("  [fig5] ✓ faithfulness")


# ── Fig 6: agreement↔uncertainty + per-seed distribution ──────
def fig6_hypothesis():
    px = _load(f"{XAI}/per_patient_xai.json"); ms = _load(f"{RES}/multiseed_summary.json")
    if not (px and ms):
        print("  [fig6] رد شد"); return
    agr = np.array([p["xai_agreement"] for p in px["patients"] if p["xai_agreement"] is not None])
    unc = np.array([p["uncertainty_std"] for p in px["patients"] if p["xai_agreement"] is not None])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].scatter(unc, agr, s=14, alpha=0.5, color=C["grud"])
    axes[0].set_xlabel("MC-Dropout uncertainty (std)")
    axes[0].set_ylabel("Cross-method agreement")
    axes[0].set_title("Agreement vs uncertainty (single model)")
    # per-seed distribution
    ru = ms["aggregate"]["rho_agr_unc"]["values"]
    rc = ms["aggregate"]["r_agr_correct"]["values"]
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].boxplot([ru, rc])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["ρ: agr↔uncert.", "r: agr↔correct"])
    for i, vals in enumerate([ru, rc], 1):
        axes[1].scatter(np.full(len(vals), i) + np.random.uniform(-0.05, 0.05, len(vals)),
                        vals, s=18, color=C["grud"], alpha=0.7, zorder=3)
    K = ms.get("K", len(ru))
    axes[1].set_ylabel(f"Correlation over {K} seeds")
    axes[1].set_title("Stability across seeds")
    fig.savefig(f"{FIG}/fig6_hypothesis.png"); plt.close(fig)
    print("  [fig6] ✓ فرضیه")


# ── Fig 7: TreeSHAP importance + cross-model ──────────────────
def fig7_treeshap():
    p = f"{XAI}/treeshap_feature_importance.csv"; tj = _load(f"{XAI}/treeshap_lgbm.json")
    if not os.path.exists(p):
        print("  [fig7] رد شد"); return
    df = pd.read_csv(p).head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].barh(df["feature"], df["treeshap_importance"], color=C["lgbm"])
    axes[0].set_xlabel("mean |SHAP|"); axes[0].set_title("LightGBM TreeSHAP — top features")
    if tj and tj.get("cross_model_agreement_with_grud"):
        cm = tj["cross_model_agreement_with_grud"]
        ms_ = list(cm.keys()); rs = [cm[m]["rho"] for m in ms_]
        axes[1].bar(range(len(ms_)), rs, color=C["acc"])
        axes[1].set_xticks(range(len(ms_)))
        axes[1].set_xticklabels(ms_, rotation=25, ha="right", fontsize=8)
        axes[1].set_ylabel("Spearman ρ"); axes[1].set_ylim(0, 1)
        axes[1].set_title("Cross-model agreement\nTreeSHAP ↔ GRU-D methods")
    fig.savefig(f"{FIG}/fig7_treeshap.png"); plt.close(fig)
    print("  [fig7] ✓ TreeSHAP")


def run():
    os.makedirs(FIG, exist_ok=True)
    print("Generate Figures...")
    fig1_models()
    fig3_agreement()
    fig4_sanity()
    fig5_faithfulness()
    fig6_hypothesis()
    fig7_treeshap()
    try:
        yp = _grud_probs()
        if yp is not None:
            fig2_curves(*yp)
            fig8_calibration(*yp)
    except Exception as e:
        print(f"  [fig2/8] رد شد: {e}")
    print(f"\n  ✅ نمودارها در {FIG}/")


if __name__ == "__main__":
    run()