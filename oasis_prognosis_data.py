"""
oasis_prognosis_data.py — File 1/N
==================================================================
ماژول دادهٔ بدون‌نشتی برای مسئلهٔ «پیش‌بینی پیشرفت به دمانس» روی OASIS-3.

این ماژول جایگزین oasis3_loader.py می‌شود و سه مشکل را هم‌زمان حل می‌کند:
  - نشتی imputation  → هیچ آماره‌ای روی کل داده حساب نمی‌شود.
  - گمشدگی شدید      → mask و time-delta برای GRU-D تولید می‌شود؛
                       imputation فقط داخل fold و فقط با آماره‌های train.
  - چارچوب مفهومی    → برچسب prognosis (آینده)، نه classification هم‌زمان.

تعریف outcome (دودویی) — همان طرح تأییدشده:
  جامعه : بیمارانی که در ابتدا دمانس نیستند (Label != 2 در شروع).
  مثبت  : در طول follow-up به دمانس می‌رسند (Label == 2).
          ورودی = فقط ویزیت‌های «قبل از» اولین ویزیت دمانس.
  منفی  : هرگز به دمانس نمی‌رسند، مشروط به حداقل NEG_MIN_FU_YEARS سال پیگیری.

خروجی اصلی: یک شیء PrognosisDataset که شامل تنسورهای زیر است
  values : (N, T, F)  مقادیر خام — np.nan در جاهای گمشده، 0 در padding
  mask   : (N, T, F)  1 اگر مقدار مشاهده‌شده، 0 اگر گمشده/padding   (برای GRU-D)
  delta  : (N, T, F)  زمان (روز) از آخرین مشاهدهٔ همان ویژگی          (برای GRU-D)
  seqlen : (N,)       تعداد ویزیت‌های معتبر هر بیمار
  y      : (N,)       0 = پایدار، 1 = پیشرفت به دمانس
  pid    : (N,)       شناسهٔ بیمار

نکتهٔ مهم دربارهٔ leakage:
  این ماژول هیچ imputation/scaling سراسری انجام نمی‌دهد.
  برای مدل‌هایی که ورودی عددیِ کامل می‌خواهند (BiLSTM/Transformer)،
  از کلاس LeakFreeImputerScaler استفاده کنید که فقط روی train fit می‌شود.
  GRU-D اصلاً imputation نمی‌خواهد (از mask+delta استفاده می‌کند).
  LightGBM هم NaN را بومی هندل می‌کند.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ==================================================================
# پیکربندی — همه‌چیز اینجا قابل تغییر است
# ==================================================================

# ستون‌های ویژگی (۲۲ ویژگی؛ CDRTOT/CDRSUM/Label/ID حذف شده‌اند).
# PARKSIGN (۹۸٪ گمشده) از ابتدا حذف است.
FEATURE_COLS: List[str] = [
    "Memory_Score", "LOGIMEM", "ANIMALS", "VEG", "digfor", "digback",
    "GDS_Total", "NPI_Apathy", "BEAPATHY", "GAITDIS", "FAQ_Score",
    "Gender", "Education", "APOE_risk", "Age", "SES",
    "HEIGHT", "WEIGHT", "BPSYS", "BPDIAS", "MOMDEM", "DADDEM",
]

ID_COL = "OASISID"
TIME_COL = "days_to_visit"
LABEL_COL = "Label"            # 0=Normal, 1=MCI, 2=Dementia (مشتق از CDRTOT)
DEMENTIA_LABEL = 2

SENTINEL_VALUES = [999, 999.0]

# پارامترهای تعریف مسئله
MAX_VISITS = 20               # حداکثر طول توالی (آخرین ویزیت‌ها نگه داشته می‌شوند)
MIN_INPUT_VISITS = 2          # حداقل ویزیت ورودی لازم
NEG_MIN_FU_YEARS = 3.0        # منفی‌ها باید حداقل این مدت بدون پیشرفت دنبال شده باشند
DAYS_PER_YEAR = 365.25


# ==================================================================
# ساختار خروجی
# ==================================================================

@dataclass
class PrognosisDataset:
    values: np.ndarray          # (N, T, F) خام، nan در گمشده، 0 در padding
    mask: np.ndarray            # (N, T, F) 1=مشاهده‌شده
    delta: np.ndarray           # (N, T, F) روز از آخرین مشاهده
    seqlen: np.ndarray          # (N,)
    y: np.ndarray               # (N,)
    pid: np.ndarray             # (N,)
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLS))

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "PrognosisDataset":
        return PrognosisDataset(
            values=self.values[idx], mask=self.mask[idx], delta=self.delta[idx],
            seqlen=self.seqlen[idx], y=self.y[idx], pid=self.pid[idx],
            feature_names=list(self.feature_names),
        )

    def summary(self) -> str:
        n = len(self.y)
        pos = int(self.y.sum())
        return (f"PrognosisDataset: N={n}  pos={pos} ({100*pos/n:.1f}%)  "
                f"neg={n-pos}  shape={self.values.shape}  "
                f"median_seqlen={int(np.median(self.seqlen))}")


# ==================================================================
# بارگذاری و پاک‌سازی اولیه
# ==================================================================

def _encode_apoe(val) -> float:
    if pd.isna(val):
        return -1.0
    code = int(val)
    if code in (22, 23, 33):
        return 0.0
    if code in (24, 34):
        return 1.0
    if code == 44:
        return 2.0
    return -1.0


def load_raw(csv_path: str) -> pd.DataFrame:
    """خواندن CSV و پاک‌سازی اولیه — بدون هیچ imputation."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # sentinel 999 → NaN
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace(SENTINEL_VALUES, np.nan)

    # encode دموگرافیک‌ها
    if "APOE_risk" in df.columns:
        df["APOE_risk"] = df["APOE_risk"].apply(_encode_apoe)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({1: 0, 2: 1})

    df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    return df


# ==================================================================
# ساخت برچسب prognosis و انتخاب ویزیت‌های ورودی
# ==================================================================

def _label_and_window(patient_df: pd.DataFrame) -> Optional[Tuple[int, pd.DataFrame]]:
    """
    خروجی: (y, input_visits_df) یا None اگر بیمار واجد شرایط نباشد.

    منطق:
      - اگر بیمار از ابتدا دمانس است → خارج (None).
      - اگر به دمانس می‌رسد → y=1، ورودی = ویزیت‌های قبل از اولین دمانس.
      - اگر هرگز نمی‌رسد و follow-up کافی دارد → y=0، ورودی = همهٔ ویزیت‌ها.
      - در غیر این صورت → None.
    """
    labels = patient_df[LABEL_COL].values
    days = patient_df[TIME_COL].values

    if np.isnan(labels[0]) or labels[0] >= DEMENTIA_LABEL:
        return None

    dem_idx = np.where(labels >= DEMENTIA_LABEL)[0]

    if len(dem_idx) > 0:
        first_dem = int(dem_idx[0])
        window = patient_df.iloc[:first_dem]          # قبل از اولین دمانس
        if len(window) >= MIN_INPUT_VISITS:
            return 1, window
        return None
    else:
        fu_years = (days[-1] - days[0]) / DAYS_PER_YEAR
        if fu_years >= NEG_MIN_FU_YEARS and len(patient_df) >= MIN_INPUT_VISITS:
            return 0, patient_df
        return None


# ==================================================================
# ساخت تنسورهای values / mask / delta برای یک بیمار
# ==================================================================

def _build_patient_tensors(window: pd.DataFrame
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    ساخت (values, mask, delta) برای ویزیت‌های یک بیمار، سپس pre-padding تا MAX_VISITS.

    delta طبق Che et al. 2018 (GRU-D):
      delta[0]   = 0
      delta[t]   = (s_t - s_{t-1})                        اگر ویژگی در t-1 مشاهده شده
                 = (s_t - s_{t-1}) + delta[t-1]           اگر در t-1 گمشده بوده
    """
    vals = window[FEATURE_COLS].values.astype(np.float64)   # (v, F) با nan
    days = window[TIME_COL].values.astype(np.float64)       # (v,)
    v, F = vals.shape

    mask = (~np.isnan(vals)).astype(np.float64)             # 1=مشاهده‌شده

    # محاسبهٔ delta روی ویزیت‌های واقعی
    delta = np.zeros((v, F), dtype=np.float64)
    for t in range(1, v):
        gap = days[t] - days[t - 1]
        # اگر در گام قبلی مشاهده شده بود → فقط gap؛ وگرنه gap + delta قبلی
        delta[t] = gap + np.where(mask[t - 1] == 0, delta[t - 1], 0.0)

    # نگه‌داشتن آخرین MAX_VISITS ویزیت (جدیدترها مهم‌تر)
    if v > MAX_VISITS:
        vals, mask, delta = vals[-MAX_VISITS:], mask[-MAX_VISITS:], delta[-MAX_VISITS:]
        v = MAX_VISITS

    seqlen = v

    # pre-padding در ابتدا تا طول ثابت MAX_VISITS
    if v < MAX_VISITS:
        pad = MAX_VISITS - v
        vals = np.vstack([np.zeros((pad, F)), vals])
        mask = np.vstack([np.zeros((pad, F)), mask])
        delta = np.vstack([np.zeros((pad, F)), delta])

    return vals, mask, delta, seqlen


# ==================================================================
# تابع اصلی ساخت دیتاست
# ==================================================================

def build_prognosis_dataset(csv_path: str, verbose: bool = True) -> PrognosisDataset:
    df = load_raw(csv_path)

    values_list, mask_list, delta_list = [], [], []
    seqlen_list, y_list, pid_list = [], [], []

    n_total = df[ID_COL].nunique()
    n_baseline_dem = 0
    n_short = 0

    for pid, pdf in df.groupby(ID_COL, sort=False):
        pdf = pdf.reset_index(drop=True)
        if len(pdf) < MIN_INPUT_VISITS:
            n_short += 1
            continue

        res = _label_and_window(pdf)
        if res is None:
            # تفکیک دلیل رد شدن (فقط برای گزارش)
            if not np.isnan(pdf[LABEL_COL].values[0]) and pdf[LABEL_COL].values[0] >= DEMENTIA_LABEL:
                n_baseline_dem += 1
            continue

        y, window = res
        vals, mask, delta, seqlen = _build_patient_tensors(window)

        values_list.append(vals)
        mask_list.append(mask)
        delta_list.append(delta)
        seqlen_list.append(seqlen)
        y_list.append(y)
        pid_list.append(pid)

    # nan در padding را به 0 تبدیل می‌کنیم (mask=0 آنجا)؛ nan در ویزیت‌های واقعی باقی می‌ماند
    values = np.array(values_list)
    mask = np.array(mask_list)
    # در جای padding، values ممکن است 0 باشد و mask=0 — اوکی.
    # در جای گمشدهٔ واقعی، values=nan و mask=0.

    ds = PrognosisDataset(
        values=values,
        mask=mask,
        delta=np.array(delta_list),
        seqlen=np.array(seqlen_list, dtype=np.int64),
        y=np.array(y_list, dtype=np.int64),
        pid=np.array(pid_list, dtype=object),
        feature_names=list(FEATURE_COLS),
    )

    if verbose:
        print("=" * 64)
        print("ساخت دیتاست prognosis (پیشرفت به دمانس)")
        print("=" * 64)
        print(f"  کل بیماران در CSV          : {n_total}")
        print(f"  حذف‌شده (در ابتدا دمانس)    : {n_baseline_dem}")
        print(f"  حذف‌شده (ویزیت کم)          : {n_short}")
        print(f"  {ds.summary()}")
        print("=" * 64)

    return ds


# ==================================================================
# split در سطح بیمار (stratified) — بدون هم‌پوشانی بیمار
# ==================================================================

def patient_level_split(ds: PrognosisDataset, test_size: float = 0.3,
                        random_state: int = 42
                        ) -> Tuple[PrognosisDataset, PrognosisDataset]:
    """چون هر نمونه یک بیمار است، split معمولیِ stratified کافی است (بدون نشتی بیمار)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(ds))
    tr, te = train_test_split(idx, test_size=test_size, stratify=ds.y,
                              random_state=random_state)
    return ds.subset(tr), ds.subset(te)


# ==================================================================
# پیش‌پردازشِ بدون‌نشتی برای مدل‌هایی که ورودی عددیِ کامل می‌خواهند
# (BiLSTM, Transformer). GRU-D و LightGBM به این نیاز ندارند.
# ==================================================================

class LeakFreeImputerScaler:
    """
    imputation + scaling که فقط روی train fit می‌شود.
      fit(train_values, train_mask) → میانگین هر ویژگی فقط از مقادیر مشاهده‌شدهٔ train
      transform():
        ۱. forward-fill درون هر بیمار (فقط گذشته→حال، بدون نشتی زمانی)
        ۲. پر کردن باقیمانده با میانگین train
        ۳. استانداردسازی با (mean, std) از train
      padding (mask=0 در ابتدای توالی) همیشه 0 می‌ماند.
    """

    def __init__(self):
        self.feat_mean_: Optional[np.ndarray] = None
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_std_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "LeakFreeImputerScaler":
        N, T, F = values.shape
        flat_v = values.reshape(-1, F)
        flat_m = mask.reshape(-1, F).astype(bool)
        # میانگین فقط روی مقادیر واقعاً مشاهده‌شده
        self.feat_mean_ = np.array([
            np.nanmean(flat_v[flat_m[:, f], f]) if flat_m[:, f].any() else 0.0
            for f in range(F)
        ])
        # برای scaler: ابتدا impute موقت با همین میانگین، بعد mean/std
        imputed = self._impute(values, mask)
        valid_rows = mask.reshape(-1, F).any(axis=1)  # ردیف‌های غیر-padding
        flat_imp = imputed.reshape(-1, F)[valid_rows]
        self.scaler_mean_ = flat_imp.mean(axis=0)
        self.scaler_std_ = flat_imp.std(axis=0) + 1e-8
        return self

    def _impute(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ffill درون‌بیمار + پر کردن با میانگین train. padding صفر می‌ماند."""
        N, T, F = values.shape
        out = values.copy()
        for i in range(N):
            last = np.full(F, np.nan)
            for t in range(T):
                if mask[i, t].sum() == 0 and np.all(np.nan_to_num(values[i, t]) == 0):
                    continue  # احتمالاً padding — دست نزن
                row = out[i, t]
                observed = mask[i, t].astype(bool)
                # ffill: جاهای گمشده را از آخرین مقدار معتبر پر کن
                row = np.where(observed, row, last)
                out[i, t] = row
                # به‌روزرسانی last فقط با مقادیر مشاهده‌شده
                last = np.where(observed, values[i, t], last)
        # هر nan باقیمانده → میانگین train
        nan_mask = np.isnan(out)
        if nan_mask.any():
            fill = np.broadcast_to(self.feat_mean_, out.shape)
            out = np.where(nan_mask, fill, out)
        return out

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        assert self.feat_mean_ is not None, "ابتدا fit کنید."
        imputed = self._impute(values, mask)
        scaled = (imputed - self.scaler_mean_) / self.scaler_std_
        # padding را صفر نگه دار (جاهایی که کل timestep mask=0 است)
        pad = (mask.sum(axis=2, keepdims=True) == 0)
        scaled = np.where(pad, 0.0, scaled)
        return scaled.astype(np.float32)

    def fit_transform(self, values, mask):
        return self.fit(values, mask).transform(values, mask)


# ==================================================================
# تست سریع روی داده واقعی
# ==================================================================

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "integrated_data.csv"

    ds = build_prognosis_dataset(csv)
    tr, te = patient_level_split(ds, test_size=0.3, random_state=42)
    print(f"\nTrain: {tr.summary()}")
    print(f"Test : {te.summary()}")
    print(f"هم‌پوشانی بیمار train/test: "
          f"{len(set(tr.pid) & set(te.pid))}")

    # تست LeakFreeImputerScaler (فقط روی train fit)
    imp = LeakFreeImputerScaler().fit(tr.values, tr.mask)
    Xtr = imp.transform(tr.values, tr.mask)
    Xte = imp.transform(te.values, te.mask)
    print(f"\nبعد از impute+scale بدون‌نشتی:")
    print(f"  X_train: {Xtr.shape}  nan={np.isnan(Xtr).sum()}  "
          f"mean≈{Xtr[tr.mask.sum(2) > 0].mean():.3f}")
    print(f"  X_test : {Xte.shape}  nan={np.isnan(Xte).sum()}")
    print(f"  میانگین/انحراف از train حساب شده (نه test) → بدون نشتی ✅")
