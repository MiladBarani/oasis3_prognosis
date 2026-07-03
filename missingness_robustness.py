"""
missingness_robustness.py — File 12/N
==================================================================
آزمونِ مقاومت در برابر «informative missingness».

SHAP نشان داد ویژگی‌های __missrate جزو مهم‌ترین پیش‌بینی‌کننده‌ها هستند.
این یعنی مدل ممکن است الگوی گمشدگی (که می‌تواند آرتیفکتِ سایت/پروتکل
باشد) را یاد بگیرد، نه فقط مقادیر بالینی. این اسکریپت LightGBM را در
سه پیکربندی روی K seed آموزش می‌دهد و عملکرد را مقایسه می‌کند:

  full         : همهٔ ویژگی‌ها (last/mean/slope/missrate + seqlen)
  no_missrate  : بدون missrate (کنترلِ گمشدگیِ صریح)
  values_only  : فقط last و mean (مقادیر بالینیِ خام؛ بدون slope/missrate/seqlen)

تفسیر:
  اگر no_missrate ≈ full → سیگنال عمدتاً بالینی است (خیال راحت).
  اگر no_missrate ≪ full → مدل به الگوی گمشدگی تکیه دارد (باید بحث شود).

اجرا:
    python missingness_robustness.py data\\integrated_data.csv 10

خروجی: results/missingness_robustness.json , results/figures/ext_s7_missingness_robustness.png
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding="utf-8")
from oasis_prognosis_data import build_prognosis_dataset, patient_level_split
from lgbm_baseline import build_tabular_features, make_lgbm, binary_metrics

RESULTS = "results"
FIG = os.path.join(RESULTS, "figures")
BASE_SEED = 42

CONFIGS = {
    "full":        lambda c: True,
    "no_missrate": lambda c: not c.endswith("__missrate"),
    # کنترلِ نشتیِ طراحی: seqlen با برچسب همبسته است (مثبت‌ها فقط ویزیت‌های
    # قبل از دمانس را دارند). این config هم missrate و هم seqlen را حذف می‌کند
    # تا ببینیم مدل به آرتیفکتِ طولِ توالی تکیه دارد یا نه.
    "no_leak":     lambda c: (not c.endswith("__missrate")) and c != "__seqlen",
    "values_only": lambda c: c.endswith("__last") or c.endswith("__mean"),
}


def select(cols, keep):
    return [c for c in cols if keep(c)]


def run(csv_path: str, K: int = 10):
    os.makedirs(FIG, exist_ok=True)
    ds = build_prognosis_dataset(csv_path, verbose=False)
    print(f"  دیتاست: N={len(ds)} | {K} seed | سه پیکربندی\n")

    per = {name: {"roc": [], "pr": []} for name in CONFIGS}
    ncols = {}
    for s in range(K):
        seed = BASE_SEED + s
        tr, te = patient_level_split(ds, test_size=0.3, random_state=seed)
        Xtr, cols = build_tabular_features(tr)
        Xte, _ = build_tabular_features(te)
        Xtr = pd.DataFrame(Xtr, columns=cols)
        Xte = pd.DataFrame(Xte, columns=cols)
        spw = float((tr.y == 0).sum() / max((tr.y == 1).sum(), 1))

        line = f"  seed {s+1}/{K}: "
        for name, keep in CONFIGS.items():
            use = select(cols, keep)
            ncols[name] = len(use)
            clf = make_lgbm(spw)
            clf.fit(Xtr[use], tr.y)
            m = binary_metrics(te.y, clf.predict_proba(Xte[use])[:, 1])
            per[name]["roc"].append(m["roc_auc"])
            per[name]["pr"].append(m["pr_auc"])
            line += f"{name}={m['roc_auc']:.3f}  "
        print(line)

    def agg(vals):
        v = np.array(vals); mean = v.mean()
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        half = student_t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return {"mean": float(mean), "std": float(sd),
                "ci95": [float(mean - half), float(mean + half)]}

    summary = {name: {"n_features": ncols[name],
                      "roc_auc": agg(per[name]["roc"]),
                      "pr_auc": agg(per[name]["pr"])}
               for name in CONFIGS}

    print("\n  " + "=" * 56)
    print(f"  {'پیکربندی':14s} {'#ویژگی':>7s} {'ROC-AUC (95% CI)':>26s}")
    for name in CONFIGS:
        r = summary[name]["roc_auc"]
        print(f"  {name:14s} {summary[name]['n_features']:>7d}   "
              f"{r['mean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")

    drop = summary["full"]["roc_auc"]["mean"] - summary["no_missrate"]["roc_auc"]["mean"]
    print(f"\n  افتِ AUC با حذفِ missrate: {drop:+.3f}")
    if drop < 0.01:
        print("  → حذفِ missrate تقریباً بی‌اثر است: سیگنال عمدتاً بالینی. ✅")
    elif drop < 0.03:
        print("  → افتِ کوچک: missingness نقشِ ثانوی دارد.")
    else:
        print("  → افتِ محسوس: مدل به الگوی گمشدگی تکیه دارد (در Limitations بحث شود).")

    # افتِ ناشی از حذفِ seqlen (کنترلِ نشتیِ طراحی)
    drop_leak = (summary["no_missrate"]["roc_auc"]["mean"]
                 - summary["no_leak"]["roc_auc"]["mean"])
    print(f"  افتِ اضافیِ AUC با حذفِ seqlen (روی no_missrate): {drop_leak:+.3f}")
    if drop_leak < 0.01:
        print("  → seqlen به‌تنهایی shortcut نیست: خیال راحت. ✅")
    elif drop_leak < 0.03:
        print("  → seqlen نقشِ کوچکی دارد: در Limitations اشاره شود.")
    else:
        print("  → seqlen یک shortcutِ محسوس است: مدل تا حدی از آرتیفکتِ طراحی "
              "استفاده می‌کند (باید در Limitations صریح بحث شود).")

    with open(os.path.join(RESULTS, "missingness_robustness.json"), "w",
              encoding="utf-8") as f:
        json.dump({"K": K, "summary": summary,
                   "roc_drop_full_minus_nomissrate": float(drop),
                   "roc_drop_nomissrate_minus_noleak": float(drop_leak)}, f,
                  indent=2, ensure_ascii=False)

    # ── نمودار ──
    names = list(CONFIGS.keys())
    nconf = len(names)
    means = [summary[n]["roc_auc"]["mean"] for n in names]
    errs = [[means[i] - summary[n]["roc_auc"]["ci95"][0] for i, n in enumerate(names)],
            [summary[n]["roc_auc"]["ci95"][1] - means[i] for i, n in enumerate(names)]]
    labels = [f"{n}\n({summary[n]['n_features']} feat.)" for n in names]
    palette = ["#2c6fbb", "#7a8450", "#e0913a", "#d1495b", "#6a4ca0", "#3a3a3a"]
    colors = [palette[i % len(palette)] for i in range(nconf)]
    fig, ax = plt.subplots(figsize=(max(6.4, 1.7 * nconf), 4.6))
    bars = ax.bar(range(nconf), means, yerr=errs, capsize=5,
                  color=colors, edgecolor="white")
    for i, (m, n) in enumerate(zip(means, names)):
        hi_err = summary[n]["roc_auc"]["ci95"][1] - m
        ax.text(i, m + hi_err + 0.006, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(nconf)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ROC-AUC (mean ± 95% CI)")
    ax.set_ylim(min(means) - 0.05, max(means) + 0.06)
    ax.set_title(f"Missingness robustness — LightGBM ({K} seeds)")
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{FIG}/ext_s7_missingness_robustness.png"); plt.close(fig)
    print(f"\n  ✅ ذخیره شد: {RESULTS}/missingness_robustness.json , "
          f"{FIG}/ext_s7_missingness_robustness.png")
    return summary


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run(csv, K)