"""
make_paradox_figure.py — File 13/N
==================================================================
شکلِ امضای مقاله: «نقشهٔ sanity ↔ faithfulness».

هدف: نشان‌دادنِ «IG Paradox» در یک نگاه — روش‌هایی که وفادارترین‌اند
(faithfulness بالا) لزوماً معتبرترین از نظرِ sanity نیستند. هر روش یک
نقطه در صفحهٔ دوبعدی است:

  محور X = اعتبارِ sanity   = 1 − شباهت(اصلی↔تصادفی)   (بزرگ‌تر = بهتر)
  محور Y = وفاداری (نرمال‌شده، ترکیبِ Del/Ins/Compr)      (بزرگ‌تر = بهتر)

چهار ربع:
  بالا-راست : هم وفادار، هم معتبر  → روشِ ایده‌آل
  بالا-چپ   : وفادار ولی نامعتبر   → «پارادوکس» (IG اینجاست)
  پایین-راست: معتبر ولی کم‌وفادار
  پایین-چپ  : نه این نه آن

ورودی: results/xai/sanity.json , faithfulness.json
خروجی: results/figures/fig9_sanity_faithfulness_map.png
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RES = "results"
XAI = os.path.join(RES, "xai")
FIG = os.path.join(RES, "figures")

# پالتِ عمدی: هر خانوادهٔ روش یک ته‌رنگ. جدا از پیش‌فرض‌های کلیشه‌ای.
METHOD_STYLE = {
    "saliency":             {"c": "#3f7cac", "label": "Saliency"},
    "smoothgrad":           {"c": "#5fa8d3", "label": "SmoothGrad"},
    "grad_input":           {"c": "#c1666b", "label": "Gradient×Input"},
    "integrated_gradients": {"c": "#8b2635", "label": "Integrated Gradients"},
    "occlusion":            {"c": "#6a8532", "label": "Occlusion"},
}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run():
    os.makedirs(FIG, exist_ok=True)
    sanity = _load(f"{XAI}/sanity.json")
    faith = _load(f"{XAI}/faithfulness.json")

    methods = list(METHOD_STYLE.keys())

    # محورِ X: اعتبارِ sanity = 1 − شباهت (میانگینِ دو تست: full + data)
    full_sim = sanity["full_randomization_similarity"]
    data_sim = sanity["data_randomization_similarity"]
    sanity_validity = {m: 1.0 - 0.5 * (full_sim[m] + data_sim[m]) for m in methods}

    # محورِ Y: وفاداریِ ترکیبی از سه معیارِ غیر-circular (Del/Ins/Compr)
    # (F-corr و occlusion را برای انصاف کنار می‌گذاریم چون circular بودند)
    pm = faith["per_method"]
    raw = {m: pm[m]["deletion_aopc"] + pm[m]["insertion_aopc"]
              + pm[m]["comprehensiveness"] for m in methods}
    lo, hi = min(raw.values()), max(raw.values())
    faith_norm = {m: (raw[m] - lo) / (hi - lo + 1e-12) for m in methods}

    # آستانهٔ sanity: verdict قوی در کد اصلی sim<0.30 → validity>0.70
    x_thresh = 0.70
    y_thresh = 0.50

    # کرانِ محورها با headroom کافی برای برچسب‌ها
    xmin, xmax = 0.10, 0.62
    ymin, ymax = -0.12, 1.22

    fig, ax = plt.subplots(figsize=(8.2, 6.6))

    # ── پس‌زمینهٔ چهار ربع ──
    ax.axvspan(x_thresh, xmax, color="#edf3ec", zorder=0)          # ستونِ معتبر
    ax.axhspan(y_thresh, ymax, color="#f4eef1", alpha=0.45, zorder=0)
    ax.axvline(x_thresh, color="#2e8b57", lw=1.2, ls="--", zorder=1)
    ax.axhline(y_thresh, color="#8b2635", lw=1.2, ls="--", zorder=1)

    # ── عنوانِ ربع‌ها (در گوشه‌ها، خارج از مسیرِ نقاط) ──
    ax.text(0.115, 1.17, "PARADOX", fontsize=10, color="#8b2635",
            va="top", ha="left", fontweight="bold")
    ax.text(0.115, 1.12, "faithful, not model-dependent", fontsize=8.2,
            color="#8b2635", va="top", ha="left", style="italic")
    ax.text(0.615, 1.17, "IDEAL", fontsize=10, color="#2e6b3a",
            va="top", ha="right", fontweight="bold")
    ax.text(0.615, 1.12, "faithful & model-dependent", fontsize=8.2,
            color="#2e6b3a", va="top", ha="right")
    ax.text(0.115, -0.08, "neither", fontsize=8.5, color="#999",
            va="bottom", ha="left", style="italic")
    ax.text(0.615, -0.08, "model-dependent, less faithful", fontsize=8.5,
            color="#667", va="bottom", ha="right", style="italic")

    # ── فلشِ روایت: از پارادوکس (IG) به سمتِ آستانهٔ اعتبار ──
    ig_x, ig_y = sanity_validity["integrated_gradients"], faith_norm["integrated_gradients"]
    sal_x, sal_y = sanity_validity["saliency"], faith_norm["saliency"]
    arrow = FancyArrowPatch((ig_x + 0.006, ig_y - 0.02), (sal_x - 0.006, sal_y + 0.02),
                            connectionstyle="arc3,rad=-0.28",
                            arrowstyle="-|>", mutation_scale=13,
                            color="#b0b0b0", lw=1.2, ls=(0, (2, 2)), zorder=2)
    ax.add_patch(arrow)

    # ── نقاطِ روش‌ها + برچسب با آفستِ ضدِتداخل ──
    # آفستِ دستیِ هر برچسب تا هیچ‌کدام روی هم/روی عنوان‌ها نیفتد
    label_off = {
        "saliency":             (0.0, -0.075, "center", "top"),
        "smoothgrad":           (0.0,  0.055, "center", "bottom"),
        "grad_input":           (-0.014, 0.02, "right", "center"),
        "integrated_gradients": (0.012, 0.045, "left", "bottom"),
        "occlusion":            (0.014, 0.0, "left", "center"),
    }
    for m in methods:
        st = METHOD_STYLE[m]
        x, y = sanity_validity[m], faith_norm[m]
        ax.scatter(x, y, s=240, color=st["c"], edgecolor="white", lw=1.8,
                   zorder=4)
        dx, dy, ha, va = label_off[m]
        ax.annotate(st["label"], (x, y), (x + dx, y + dy),
                    fontsize=9.5, ha=ha, va=va, color="#1a1a1a",
                    fontweight="medium", zorder=5)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Sanity validity  (1 − similarity to randomized model)  →",
                  fontsize=10.5)
    ax.set_ylabel("Faithfulness  (normalized Deletion + Insertion + Comprehensiveness)  →",
                  fontsize=10.5)
    ax.set_title("The IG paradox — faithfulness and sanity validity are not aligned",
                 fontsize=13, pad=14)
    ax.grid(alpha=0.16, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{FIG}/fig9_sanity_faithfulness_map.png", dpi=300)
    plt.close(fig)
    print(f"  [fig9] ✓ نقشهٔ sanity↔faithfulness ذخیره شد: "
          f"{FIG}/fig9_sanity_faithfulness_map.png")


if __name__ == "__main__":
    run()