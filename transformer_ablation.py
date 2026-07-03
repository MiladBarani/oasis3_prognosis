"""
transformer_ablation.py — File 4/N
==================================================================
مدل ablation: یک Transformer کوچک برای «پیشرفت به دمانس».

نقش این فایل در مقاله:
  این مدل عمداً «استاندارد» است — برخلاف GRU-D، زمانِ نامنظم را مدل
  نمی‌کند و گمشدگی را با imputation پر می‌کند. هدف نشان دادن این است
  که یک مدل attentionِ بزرگ‌تر هم سقف baseline را نمی‌شکند. اگر این
  هم نبازد، روایت «پیچیدگیِ بیشتر روی این داده کمک نکرد» کامل می‌شود.

تفاوت کلیدی با GRU-D:
  - ورودی = مقادیرِ impute‌شدهٔ بدون‌نشتی (LeakFreeImputerScaler از فایل ۱)
  - positional embedding یادگرفته‌شده روی گام‌های زمانی (نه فاصلهٔ واقعی)
  - attention با ماسکِ گام‌های padding
  - pooling میانگینِ ماسک‌دار

بدون‌نشتی: imputation و scaling فقط روی train (داخل LeakFreeImputerScaler).
ارزیابی: همان CV + test + bootstrap که برای GRU-D استفاده شد.

این فایل از فایل‌های ۱ و ۲ استفاده می‌کند و باید کنارشان باشد.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split

from oasis_prognosis_data import (
    build_prognosis_dataset, patient_level_split, LeakFreeImputerScaler,
)
from lgbm_baseline import binary_metrics, bootstrap_ci

RANDOM_STATE = 42
RESULTS_DIR = "results"


def set_seeds(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ==================================================================
# لایه‌های کمکیِ ماسک‌دار (به‌جای Lambda، برای پایداری)
# ==================================================================

@keras.saving.register_keras_serializable(package="tfm")
class MaskedAveragePooling(layers.Layer):
    """میانگین روی گام‌های معتبر (padding نادیده)."""
    def call(self, inputs):
        x, step_mask = inputs
        m = keras.ops.cast(step_mask, x.dtype)
        m = keras.ops.expand_dims(m, axis=-1)            # (N,T,1)
        s = keras.ops.sum(x * m, axis=1)
        c = keras.ops.sum(m, axis=1) + 1e-8
        return s / c


@keras.saving.register_keras_serializable(package="tfm")
class AddPositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model, **kw):
        super().__init__(**kw)
        self.seq_len = int(seq_len); self.d_model = int(d_model)
        self.pos = layers.Embedding(self.seq_len, self.d_model)

    def call(self, x):
        positions = keras.ops.arange(self.seq_len)
        return x + self.pos(positions)[None, :, :]

    def get_config(self):
        c = super().get_config()
        c.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return c


# ==================================================================
# مدل
# ==================================================================

def build_model(T: int, F: int, d_model: int = 32, heads: int = 2,
                n_blocks: int = 1, dropout: float = 0.3) -> keras.Model:
    x_in = keras.Input(shape=(T, F), name="values")
    m_in = keras.Input(shape=(T,), name="step_mask")

    h = layers.Dense(d_model)(x_in)
    h = AddPositionalEmbedding(T, d_model)(h)

    # ماسک attention: کلیدهای padding حذف شوند → (N,1,T) با broadcast
    attn_mask = keras.ops.expand_dims(keras.ops.cast(m_in, "bool"), axis=1)

    for _ in range(n_blocks):
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads,
                                         dropout=dropout)(
            h, h, attention_mask=attn_mask)
        h = layers.LayerNormalization()(h + attn)
        ff = layers.Dense(d_model * 2, activation="relu")(h)
        ff = layers.Dense(d_model)(ff)
        h = layers.LayerNormalization()(h + ff)

    pooled = MaskedAveragePooling()([h, m_in])
    z = layers.Dropout(dropout)(pooled)
    z = layers.Dense(16, activation="relu")(z)
    z = layers.Dropout(dropout)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    model = keras.Model([x_in, m_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"),
                           keras.metrics.AUC(name="prauc", curve="PR")])
    return model


def class_weights(y: np.ndarray) -> Dict[int, float]:
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    return {0: n / (2 * n0), 1: n / (2 * n1)}


def train_one(Xtr, Mtr, ytr, T, F, seed=RANDOM_STATE, max_epochs=120) -> keras.Model:
    set_seeds(seed)
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                stratify=ytr, random_state=seed)
    model = build_model(T, F)
    es = keras.callbacks.EarlyStopping(monitor="val_prauc", mode="max",
                                       patience=15, restore_best_weights=True)
    model.fit([Xtr[itr], Mtr[itr]], ytr[itr],
              validation_data=([Xtr[iva], Mtr[iva]], ytr[iva]),
              epochs=max_epochs, batch_size=32,
              class_weight=class_weights(ytr[itr]),
              callbacks=[es], verbose=0)
    return model


# ==================================================================
# اجرا
# ==================================================================

def run(csv_path: str, n_folds: int = 5):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seeds()

    ds = build_prognosis_dataset(csv_path)
    tr, te = patient_level_split(ds, test_size=0.3, random_state=RANDOM_STATE)
    T, F = tr.values.shape[1], tr.values.shape[2]

    # imputation + scaling بدون‌نشتی (فقط روی train)
    imp = LeakFreeImputerScaler().fit(tr.values, tr.mask)
    Xtr = imp.transform(tr.values, tr.mask)
    Xte = imp.transform(te.values, te.mask)
    # ماسک گام (هر گامی که حداقل یک ویژگی مشاهده‌شده دارد)
    Mtr = (tr.mask.sum(axis=2) > 0).astype(np.float32)
    Mte = (te.mask.sum(axis=2) > 0).astype(np.float32)
    ytr, yte = tr.y, te.y

    print(f"\n  ورودی Transformer: X={Xtr.shape} (impute‌شدهٔ بدون‌نشتی), "
          f"step_mask={Mtr.shape}")

    # ── CV ──
    print(f"\n── Stratified {n_folds}-Fold روی train (Transformer) ──")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for k, (tri, vai) in enumerate(skf.split(Xtr, ytr), 1):
        model = train_one(Xtr[tri], Mtr[tri], ytr[tri], T, F, seed=RANDOM_STATE + k)
        prob = model.predict([Xtr[vai], Mtr[vai]], verbose=0).ravel()
        m = binary_metrics(ytr[vai], prob)
        fold_metrics.append(m)
        print(f"    fold {k}: ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"sens={m['sensitivity']:.3f}")

    import pandas as pd
    dfm = pd.DataFrame(fold_metrics)
    cv = {f"{c}_mean": float(dfm[c].mean()) for c in dfm.columns}
    cv.update({f"{c}_std": float(dfm[c].std()) for c in dfm.columns})
    print("  میانگین CV:")
    for c in ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]:
        print(f"    {c:18s}: {cv[c+'_mean']:.3f} ± {cv[c+'_std']:.3f}")

    # ── test ──
    print("\n── ارزیابی نهایی روی test ──")
    final = train_one(Xtr, Mtr, ytr, T, F, seed=RANDOM_STATE)
    prob_te = final.predict([Xte, Mte], verbose=0).ravel()
    test_m = binary_metrics(yte, prob_te)
    _, auc_lo, auc_hi = bootstrap_ci(yte, prob_te, "roc_auc")
    _, ap_lo, ap_hi = bootstrap_ci(yte, prob_te, "pr_auc")
    print(f"    ROC-AUC : {test_m['roc_auc']:.3f}  (95% CI: {auc_lo:.3f}–{auc_hi:.3f})")
    print(f"    PR-AUC  : {test_m['pr_auc']:.3f}  (95% CI: {ap_lo:.3f}–{ap_hi:.3f})")
    for c in ["balanced_accuracy", "f1", "mcc", "sensitivity", "specificity"]:
        print(f"    {c:18s}: {test_m[c]:.3f}")

    out = {"cv": cv, "test": test_m,
           "test_ci": {"roc_auc": [auc_lo, auc_hi], "pr_auc": [ap_lo, ap_hi]}}
    with open(os.path.join(RESULTS_DIR, "transformer_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ ذخیره شد: {RESULTS_DIR}/transformer_results.json")
    return out


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    run(csv)
