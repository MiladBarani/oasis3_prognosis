"""
make_figures_extended.py — File 11/N
==================================================================
نمودارهای تکمیلیِ پُرارزش برای مقاله (بدونِ تصویرِ مغز).
خروجی: results/figures/ext_*.png

شامل:
  S1  CONSORT-style cohort flow
  S2  Missingness per feature (motivates masking / GRU-D)
  S3  Swimmer plot — longitudinal visit structure & conversion
  S4  Decision Curve Analysis (clinical net benefit)
  S5  LightGBM SHAP beeswarm
  S6  Temporal attribution profile (importance over visits)

پیش‌نیاز: همان فایل‌های پروژه + results/artifacts/ (فایل ۳).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join("results", "figures")
ART = os.path.join("results", "artifacts")

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
CLR = {0: "#2e8b57", 1: "#e08e0b", 2: "#c0392b"}   # CDR/Label colors


# ── S1: CONSORT flow ──────────────────────────────────────────
def s1_consort(csv):
    from oasis_prognosis_data import (
        load_raw, _label_and_window, patient_level_split,
        build_prognosis_dataset, MIN_INPUT_VISITS, ID_COL, LABEL_COL, DEMENTIA_LABEL)
    df = load_raw(csv)
    total = df[ID_COL].nunique()
    n_short = n_dem = n_other = n_final = 0
    for _, pdf in df.groupby(ID_COL):
        pdf = pdf.reset_index(drop=True)
        if len(pdf) < MIN_INPUT_VISITS:
            n_short += 1; continue
        if _label_and_window(pdf) is None:
            lab0 = pdf[LABEL_COL].values[0]
            if not np.isnan(lab0) and lab0 >= DEMENTIA_LABEL:
                n_dem += 1
            else:
                n_other += 1
            continue
        n_final += 1
    ds = build_prognosis_dataset(csv, verbose=False)
    tr, te = patient_level_split(ds, 0.3, 42)
    ntr, nte = len(tr), len(te)

    fig, ax = plt.subplots(figsize=(7.6, 5.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(2.8, 9.4)
    def box(x, y, w, h, text, fc="#eef2f7", fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#34495e", lw=1.2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#34495e", lw=1.3))
    # main column centred at x=3.6
    box(1.4, 8.0, 4.4, 1.1, f"OASIS-3 participants\nN = {total}")
    arrow(3.6, 8.0, 3.6, 7.1)                       # down to eligible
    box(1.4, 6.0, 4.4, 1.1, f"Eligible for prognosis\nN = {n_final}")
    # exclusion branch (off the vertical arrow, to the right)
    ax.plot([3.6, 6.2], [7.55, 7.55], color="#34495e", lw=1.1)
    arrow(6.2, 7.55, 6.4, 7.55)
    box(6.4, 6.8, 3.4, 1.5,
        f"Excluded (N = {total - n_final}):\n"
        f"• demented at baseline: {n_dem}\n"
        f"• < {MIN_INPUT_VISITS} visits: {n_short}\n"
        f"• short window / follow-up: {n_other}",
        fc="#fbf6e7", fs=8.5)
    # split to train / test
    arrow(3.6, 6.0, 2.2, 5.0)
    arrow(3.6, 6.0, 5.0, 5.0)
    box(0.4, 3.8, 3.6, 1.1, f"Train\nN = {ntr}", fc="#dce8f5")
    box(4.2, 3.8, 3.6, 1.1, f"Test\nN = {nte}", fc="#f5e3dc")
    ax.set_title("Cohort selection (CONSORT-style)", fontsize=12, y=0.98)
    fig.tight_layout(); fig.savefig(f"{FIG}/ext_s1_consort.png"); plt.close(fig)
    print("  [S1] ✓ CONSORT flow")


# ── S2: missingness per feature ───────────────────────────────
def s2_missingness(csv):
    from oasis_prognosis_data import FEATURE_COLS
    df = pd.read_csv(csv)
    miss = (df[FEATURE_COLS].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = ["#c0392b" if v > 80 else "#e08e0b" if v > 40 else "#2e8b57" for v in miss]
    ax.barh(miss.index, miss.values, color=colors)
    ax.set_xlabel("Missing (%)"); ax.set_xlim(0, 100)
    ax.axvline(80, ls="--", color="gray", lw=1)
    ax.set_title("Per-feature missingness (motivates masking / GRU-D)")
    for i, v in enumerate(miss.values):
        ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/ext_s2_missingness.png"); plt.close(fig)
    print("  [S2] ✓ missingness")


# ── S3: swimmer plot ──────────────────────────────────────────
def s3_swimmer(csv, n_each=14):
    from oasis_prognosis_data import (load_raw, ID_COL, TIME_COL, LABEL_COL,
                                      DEMENTIA_LABEL, MIN_INPUT_VISITS)
    from matplotlib.lines import Line2D

    df = load_raw(csv)
    prog, stable = [], []
    for pid, pdf in df.groupby(ID_COL):
        pdf = pdf.reset_index(drop=True)
        if len(pdf) < 3:
            continue
        labels = pdf[LABEL_COL].values
        if np.isnan(labels[0]) or labels[0] >= DEMENTIA_LABEL:
            continue
        (prog if (labels >= DEMENTIA_LABEL).any() else stable).append(pid)

    rng = np.random.default_rng(1)
    prog_sel = list(rng.choice(prog, min(n_each, len(prog)), replace=False))
    stab_sel = list(rng.choice(stable, min(n_each, len(stable)), replace=False))

    def conv_year(pid):
        pdf = df[df[ID_COL] == pid].sort_values(TIME_COL)
        yrs = pdf[TIME_COL].values / 365.25
        labs = pdf[LABEL_COL].values
        idx = np.where(labs >= DEMENTIA_LABEL)[0]
        return yrs[idx[0]] if len(idx) else np.inf

    def follow_len(pid):
        pdf = df[df[ID_COL] == pid]
        return pdf[TIME_COL].max() / 365.25

    prog_sel.sort(key=conv_year)
    stab_sel.sort(key=follow_len)

    ordered = stab_sel + [None] + prog_sel
    n_rows = len(ordered)
    sep_row = len(stab_sel)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    ax.axhspan(-0.5, sep_row - 0.5, color="#eaf3ee", zorder=0)
    ax.axhspan(sep_row + 0.5, n_rows - 0.5, color="#fbeceb", zorder=0)

    for row, pid in enumerate(ordered):
        if pid is None:
            continue
        pdf = df[df[ID_COL] == pid].sort_values(TIME_COL)
        yrs = pdf[TIME_COL].values / 365.25
        labs = pdf[LABEL_COL].values
        ax.plot(yrs, [row] * len(yrs), "-", color="#c9c9c9", lw=1.1, zorder=1)
        for t, lab in zip(yrs, labs):
            c = CLR.get(int(lab) if not np.isnan(lab) else 0, "#999")
            ax.scatter(t, row, color=c, s=30, zorder=3, edgecolor="white", lw=0.5)
        idx = np.where(labs >= DEMENTIA_LABEL)[0]
        if len(idx):
            ax.scatter(yrs[idx[0]], row, marker="x", s=70,
                       color="#7b241c", lw=1.8, zorder=4)

    xmin = -0.6
    ax.plot([xmin, xmin], [-0.5, sep_row - 0.5], color="#2e8b57",
            lw=5, solid_capstyle="butt", zorder=5, clip_on=False)
    ax.plot([xmin, xmin], [sep_row + 0.5, n_rows - 0.5], color="#c0392b",
            lw=5, solid_capstyle="butt", zorder=5, clip_on=False)
    ax.text(xmin - 0.35, (sep_row - 1) / 2, "Stable", rotation=90,
            va="center", ha="center", fontsize=9, color="#1e6b3a", fontweight="bold")
    ax.text(xmin - 0.35, (sep_row + n_rows) / 2, "Progressors", rotation=90,
            va="center", ha="center", fontsize=9, color="#8b2e22", fontweight="bold")

    ax.set_xlabel("Years from entry", fontsize=10.5)
    ax.set_ylabel("Patients (representative sample)", fontsize=10.5)
    ax.set_yticks([])
    ax.set_ylim(-0.5, n_rows - 0.5)

    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR[i],
                  label=l, markersize=8, markeredgecolor="white") for i, l in
           [(0, "Normal (CDR 0)"), (1, "MCI (CDR 0.5)"), (2, "Dementia (CDR\u22651)")]]
    leg.append(Line2D([0], [0], marker="x", color="#7b241c", lw=0,
                      markersize=8, markeredgewidth=1.8, label="Conversion visit"))
    ax.legend(handles=leg, loc="lower right", frameon=True, fontsize=8,
              facecolor="white", edgecolor="#cccccc")

    ax.set_title("Longitudinal visit structure & conversion to dementia\n"
                 "(sample; progressors ordered by time of conversion)", fontsize=11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{FIG}/ext_s3_swimmer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [S3] swimmer plot (improved)")


# ── helper: GRU-D probs + test ────────────────────────────────
def _grud():
    import keras
    from grud_model import GRUDCell  # noqa
    from xai_grud import mc_dropout
    model = keras.saving.load_model(f"{ART}/grud_model.keras")
    d = np.load(f"{ART}/grud_test_data.npz", allow_pickle=True)
    X, y = d["X_test"].astype("float32"), d["y_test"].astype(int)
    prob = mc_dropout(model, X, n=50)["mean"]
    return model, X, y, prob


# ── S4: decision curve analysis ───────────────────────────────
def s4_dca(y, prob):
    n = len(y); prev = y.mean()
    pts = np.linspace(0.01, 0.6, 80)
    nb_model, nb_all = [], []
    for pt in pts:
        pred = (prob >= pt).astype(int)
        tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
        nb_model.append(tp/n - fp/n * (pt/(1-pt)))
        nb_all.append(prev - (1-prev) * (pt/(1-pt)))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(pts, nb_model, color="#c0392b", lw=2, label="GRU-D model")
    ax.plot(pts, nb_all, color="#888", lw=1.3, ls="--", label="Treat all")
    ax.axhline(0, color="#444", lw=1.2, label="Treat none")
    ax.set_ylim(min(-0.05, min(nb_model) - 0.02), prev + 0.03)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis — clinical utility")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/ext_s4_dca.png"); plt.close(fig)
    print("  [S4] ✓ decision curve")


# ── S5: SHAP beeswarm (LightGBM) ──────────────────────────────
def s5_shap(csv):
    import shap
    from oasis_prognosis_data import build_prognosis_dataset, patient_level_split
    from lgbm_baseline import build_tabular_features, make_lgbm
    ds = build_prognosis_dataset(csv, verbose=False)
    tr, te = patient_level_split(ds, 0.3, 42)
    Xtr, cols = build_tabular_features(tr); Xte, _ = build_tabular_features(te)
    Xtr = pd.DataFrame(Xtr, columns=cols); Xte = pd.DataFrame(Xte, columns=cols)
    spw = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))
    clf = make_lgbm(spw); clf.fit(Xtr, tr.y)
    sv = shap.TreeExplainer(clf).shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    plt.figure(figsize=(7, 5.5))
    shap.summary_plot(sv, Xte, max_display=12, show=False)
    plt.title("LightGBM — SHAP summary (beeswarm)", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{FIG}/ext_s5_shap_beeswarm.png"); plt.close()
    print("  [S5] ✓ SHAP beeswarm")


# ── S6: temporal attribution profile ──────────────────────────
def s6_temporal(model, X):
    import tensorflow as tf
    F = (X.shape[2]) // 3
    x = tf.convert_to_tensor(X, tf.float32)
    with tf.GradientTape() as t:
        t.watch(x); p = model(x, training=False)
    g = np.abs(t.gradient(p, x).numpy()[:, :, :F])     # (N,T,F)
    mask = X[:, :, F:2*F]                               # observed indicator
    per_step = (g * mask).sum(axis=2)                   # (N,T) importance per visit
    valid = mask.sum(axis=2) > 0                        # real (non-pad) timesteps
    T = X.shape[1]
    # align by position from last visit (recency): index -1 = most recent
    prof_mean, prof_se, xs = [], [], []
    for back in range(0, 8):                            # last 8 visits
        col = T - 1 - back
        v = per_step[valid[:, col], col]
        if len(v) >= 10:
            prof_mean.append(v.mean()); prof_se.append(v.std()/np.sqrt(len(v)))
            xs.append(-back)
    prof_mean, prof_se, xs = map(np.array, (prof_mean, prof_se, xs))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.errorbar(xs, prof_mean, yerr=1.96*prof_se, fmt="o-", color="#6a4ca0",
                capsize=3, lw=2)
    ax.set_xlabel("Visit position (0 = most recent, −k = k visits earlier)")
    ax.set_ylabel("Mean |attribution| per visit")
    ax.set_title("Temporal attribution profile (GRU-D, saliency)")
    fig.tight_layout(); fig.savefig(f"{FIG}/ext_s6_temporal.png"); plt.close(fig)
    print("  [S6] ✓ temporal attribution")


def run(csv):
    os.makedirs(FIG, exist_ok=True)
    print("نمودارهای تکمیلی...")
    s1_consort(csv)
    s2_missingness(csv)
    s3_swimmer(csv)
    s5_shap(csv)
    try:
        model, X, y, prob = _grud()
        s4_dca(y, prob)
        s6_temporal(model, X)
    except Exception as e:
        print(f"  [S4/S6] رد شد: {e}")
    print(f"\n  ✅ در {FIG}/ (ext_s1..s6)")


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    run(csv)