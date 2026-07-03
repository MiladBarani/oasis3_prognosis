"""
grud_model.py — File 3/N
==================================================================
مدل اصلیِ عمیق: GRU-D (Che et al., 2018) برای «پیشرفت به دمانس».

چرا GRU-D و نه LSTM ساده:
  GRU-D سه چیزِ این داده را مستقیم در معماری مدل می‌کند:
    - گمشدگی (mask)               → بدون imputationِ ساختگی
    - فاصلهٔ زمانیِ نامنظم (delta)  → decay زمانی به سمت میانگین
    - حافظهٔ زمانی                  → recurrence
  هیچ‌کدام در BiLSTM استاندارد وجود ندارد.

مکانیک هر گام (به‌صورت خلاصه):
  γ_x = exp(-relu(w_dx·δ + b_dx))         decay ورودی (per-feature)
  γ_h = exp(-relu(δ·W_dh + b_dh))         decay حالت پنهان
  x_last = m·x + (1-m)·x_last_prev        آخرین مقدار مشاهده‌شده
  x_hat  = m·x + (1-m)·(γ_x·x_last + (1-γ_x)·x_mean)   ورودیِ بازسازی‌شده
  h_dec  = γ_h · h_prev                   حالت پنهانِ decay‌شده
  سپس یک به‌روزرسانی GRU استاندارد با ورودی [x_hat, m].

نکات بدون‌نشتی:
  - استانداردسازی (mean/std) فقط روی مقادیر *مشاهده‌شدهٔ train*.
  - چون داده استاندارد می‌شود، x_mean (هدفِ decay) صفر است.
  - δ به سال تبدیل می‌شود تا decay پایدار آموزش ببیند.

این فایل از:
  - oasis_prognosis_data.py (فایل ۱) برای داده
  - lgbm_baseline.py (فایل ۲) برای متریک‌ها و bootstrap
استفاده می‌کند. هر سه باید کنار هم در ریشهٔ پروژه باشند.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import StratifiedKFold

from oasis_prognosis_data import build_prognosis_dataset, patient_level_split
from lgbm_baseline import binary_metrics, bootstrap_ci

RANDOM_STATE = 42
RESULTS_DIR = "results"
ARTIFACT_DIR = os.path.join(RESULTS_DIR, "artifacts")
DAYS_PER_YEAR = 365.25


def set_seeds(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ==================================================================
# استانداردسازیِ بدون‌نشتی روی مقادیر مشاهده‌شدهٔ train
# ==================================================================

def fit_obs_standardizer(values: np.ndarray, mask: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """mean/std هر ویژگی فقط از سلول‌های mask==1 در train."""
    F = values.shape[2]
    flat_v = values.reshape(-1, F)
    flat_m = mask.reshape(-1, F).astype(bool)
    mean = np.zeros(F); std = np.ones(F)
    for f in range(F):
        obs = flat_v[flat_m[:, f], f]
        obs = obs[~np.isnan(obs)]
        if obs.size > 0:
            mean[f] = obs.mean()
            std[f] = obs.std() + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def make_grud_inputs(values, mask, delta, mean, std) -> np.ndarray:
    """ساخت تنسور (N,T,3F): [x_std_filled, mask, delta_years]."""
    x = (np.nan_to_num(values, nan=0.0) - mean) / std
    x = x * mask                              # سلول‌های گمشده → 0
    d = delta / DAYS_PER_YEAR                 # روز → سال
    return np.concatenate([x.astype(np.float32),
                           mask.astype(np.float32),
                           d.astype(np.float32)], axis=-1)


# ==================================================================
# سلول GRU-D
# ==================================================================

@keras.saving.register_keras_serializable(package="grud")
class GRUDCell(layers.Layer):
    def __init__(self, units, n_features, **kw):
        super().__init__(**kw)
        self.units = int(units)
        self.F = int(n_features)
        self.state_size = [self.units, self.F]   # [h, x_last]
        self.output_size = self.units

    def build(self, input_shape):
        F, U = self.F, self.units
        in_dim = 2 * F  # [x_hat, m]
        gi = "glorot_uniform"
        self.Wz = self.add_weight((in_dim, U), gi, name="Wz")
        self.Wr = self.add_weight((in_dim, U), gi, name="Wr")
        self.Wh = self.add_weight((in_dim, U), gi, name="Wh")
        self.Uz = self.add_weight((U, U), "orthogonal", name="Uz")
        self.Ur = self.add_weight((U, U), "orthogonal", name="Ur")
        self.Uh = self.add_weight((U, U), "orthogonal", name="Uh")
        self.bz = self.add_weight((U,), "zeros", name="bz")
        self.br = self.add_weight((U,), "zeros", name="br")
        self.bh = self.add_weight((U,), "zeros", name="bh")
        # پارامترهای decay
        self.w_dx = self.add_weight((F,), "zeros", name="w_dx")
        self.b_dx = self.add_weight((F,), "zeros", name="b_dx")
        self.W_dh = self.add_weight((F, U), "zeros", name="W_dh")
        self.b_dh = self.add_weight((U,), "zeros", name="b_dh")
        self.built = True

    def call(self, inputs, states, training=None):
        F = self.F
        h_prev, x_last_prev = states[0], states[1]
        x = inputs[:, :F]
        m = inputs[:, F:2 * F]
        d = inputs[:, 2 * F:3 * F]

        gamma_x = tf.exp(-tf.nn.relu(self.w_dx * d + self.b_dx))            # (b,F)
        gamma_h = tf.exp(-tf.nn.relu(tf.matmul(d, self.W_dh) + self.b_dh))  # (b,U)

        x_last = m * x + (1.0 - m) * x_last_prev
        # x_mean = 0 (داده استاندارد است) → decay به سمت صفر
        x_hat = m * x + (1.0 - m) * (gamma_x * x_last)

        h_dec = gamma_h * h_prev
        x_in = tf.concat([x_hat, m], axis=-1)
        z = tf.sigmoid(tf.matmul(x_in, self.Wz) + tf.matmul(h_dec, self.Uz) + self.bz)
        r = tf.sigmoid(tf.matmul(x_in, self.Wr) + tf.matmul(h_dec, self.Ur) + self.br)
        hh = tf.tanh(tf.matmul(x_in, self.Wh) + tf.matmul(r * h_dec, self.Uh) + self.bh)
        h = (1.0 - z) * h_dec + z * hh
        return h, [h, x_last]

    def get_config(self):
        c = super().get_config()
        c.update({"units": self.units, "n_features": self.F})
        return c


# ==================================================================
# ساخت مدل
# ==================================================================

def build_model(T: int, F: int, units: int = 32, dropout: float = 0.3) -> keras.Model:
    inp = keras.Input(shape=(T, 3 * F))
    x = layers.Masking(mask_value=0.0)(inp)        # گام‌های padding (تمام‌صفر) نادیده
    h = layers.RNN(GRUDCell(units, F))(x)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(16, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(1, activation="sigmoid")(h)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"),
                           keras.metrics.AUC(name="prauc", curve="PR")])
    return model


def class_weights(y: np.ndarray) -> Dict[int, float]:
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    return {0: n / (2 * n0), 1: n / (2 * n1)}


def train_one(Xtr, ytr, T, F, units=32, dropout=0.3,
              max_epochs=120, seed=RANDOM_STATE) -> keras.Model:
    """آموزش با early stopping روی یک inner-val کوچک."""
    set_seeds(seed)
    from sklearn.model_selection import train_test_split
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                stratify=ytr, random_state=seed)
    model = build_model(T, F, units, dropout)
    es = keras.callbacks.EarlyStopping(monitor="val_prauc", mode="max",
                                       patience=15, restore_best_weights=True)
    model.fit(Xtr[itr], ytr[itr],
              validation_data=(Xtr[iva], ytr[iva]),
              epochs=max_epochs, batch_size=32,
              class_weight=class_weights(ytr[itr]),
              callbacks=[es], verbose=0)
    return model


# ==================================================================
# اجرا: CV روی train + ارزیابی نهایی روی test
# ==================================================================

def run(csv_path: str, units: int = 32, dropout: float = 0.3, n_folds: int = 5):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    set_seeds()

    ds = build_prognosis_dataset(csv_path)
    tr, te = patient_level_split(ds, test_size=0.3, random_state=RANDOM_STATE)
    T, F = tr.values.shape[1], tr.values.shape[2]

    mean, std = fit_obs_standardizer(tr.values, tr.mask)
    Xtr = make_grud_inputs(tr.values, tr.mask, tr.delta, mean, std)
    Xte = make_grud_inputs(te.values, te.mask, te.delta, mean, std)
    ytr, yte = tr.y, te.y

    print(f"\n  ورودی GRU-D: {Xtr.shape}  (T={T}, F={F}, کانال‌ها=3F={3*F})")

    # ── CV روی train ──
    print(f"\n── Stratified {n_folds}-Fold روی train (GRU-D) ──")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for k, (tri, vai) in enumerate(skf.split(Xtr, ytr), 1):
        model = train_one(Xtr[tri], ytr[tri], T, F, units, dropout, seed=RANDOM_STATE + k)
        prob = model.predict(Xtr[vai], verbose=0).ravel()
        m = binary_metrics(ytr[vai], prob)
        fold_metrics.append(m)
        print(f"    fold {k}: ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
              f"sens={m['sensitivity']:.3f}")

    import pandas as pd
    dfm = pd.DataFrame(fold_metrics)
    cv_summary = {f"{c}_mean": float(dfm[c].mean()) for c in dfm.columns}
    cv_summary.update({f"{c}_std": float(dfm[c].std()) for c in dfm.columns})
    print("  میانگین CV:")
    for c in ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"]:
        print(f"    {c:18s}: {cv_summary[c+'_mean']:.3f} ± {cv_summary[c+'_std']:.3f}")

    # ── مدل نهایی روی کل train، ارزیابی روی test ──
    print("\n── ارزیابی نهایی روی test ──")
    final = train_one(Xtr, ytr, T, F, units, dropout, seed=RANDOM_STATE)
    prob_te = final.predict(Xte, verbose=0).ravel()
    test_m = binary_metrics(yte, prob_te)
    auc_m, auc_lo, auc_hi = bootstrap_ci(yte, prob_te, "roc_auc")
    ap_m, ap_lo, ap_hi = bootstrap_ci(yte, prob_te, "pr_auc")
    print(f"    ROC-AUC : {test_m['roc_auc']:.3f}  (95% CI: {auc_lo:.3f}–{auc_hi:.3f})")
    print(f"    PR-AUC  : {test_m['pr_auc']:.3f}  (95% CI: {ap_lo:.3f}–{ap_hi:.3f})")
    for c in ["balanced_accuracy", "f1", "mcc", "sensitivity", "specificity"]:
        print(f"    {c:18s}: {test_m[c]:.3f}")

    # ── ذخیره ──
    final.save(os.path.join(ARTIFACT_DIR, "grud_model.keras"))
    np.savez(os.path.join(ARTIFACT_DIR, "grud_test_data.npz"),
             X_test=Xte, y_test=yte, pid_test=te.pid.astype(str),
             feat_mean=mean, feat_std=std,
             feature_names=np.array(ds.feature_names))
    out = {"cv": cv_summary, "test": test_m,
           "test_ci": {"roc_auc": [auc_lo, auc_hi], "pr_auc": [ap_lo, ap_hi]},
           "units": units, "dropout": dropout, "n_folds": n_folds}
    with open(os.path.join(RESULTS_DIR, "grud_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ ذخیره شد: {ARTIFACT_DIR}/grud_model.keras , grud_test_data.npz")
    return out


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "integrated_data.csv")
    run(csv)
