# Leakage-Free Dementia-Progression Prognosis and Multi-Dimensional XAI Validation on OASIS-3

A reproducible pipeline for predicting **progression to dementia** in longitudinal clinical data, paired with a rigorous, multi-axis framework for **validating** the explanations rather than merely computing them. Built on the OASIS-3 cohort.

The project has two tightly linked goals:

1. **A leakage-controlled prognostic benchmark** — predict whether a baseline non-demented participant will progress to dementia, using patient-level splitting, fold-internal preprocessing, and explicit removal of the label-defining CDR variables from the inputs.
2. **An explanation-audit framework** — for the deep model (GRU-D), jointly quantify cross-method *agreement*, model dependence (*sanity*), and prediction impact (*faithfulness*), and surface the tension between them.

---

## Key findings

- **A strong tabular model matches time-aware deep models.** On the held-out test set, gradient-boosted trees (LightGBM) reach ROC-AUC ≈ 0.95, on par with or ahead of GRU-D (≈ 0.91–0.92 over 20 seeds) and a Transformer (≈ 0.85), on this small, imbalanced cohort.
- **Sanity and faithfulness rankings are (near-)reversed.** The attribution methods that pass Adebayo-style sanity checks most convincingly (saliency, SmoothGrad) are the *least* faithful, while the most faithful methods (Integrated Gradients, gradient×input) *fail* sanity. High cross-method agreement therefore does **not** certify trustworthiness.
- **Occlusion faithfulness is circular.** The occlusion attribution equals the single-feature deletion effect used as ground truth, forcing its faithfulness correlation to 1 by construction; it is excluded from that metric.
- **Missingness and sequence length are secondary.** A nested-feature ablation shows removing missingness-rate features costs only ≈ 0.013 ROC-AUC, and removing the (label-correlated) sequence-length feature has essentially no effect — the model relies on clinical signal, not design artifacts.
- **Operating-point trade-off.** At a fixed 0.5 threshold, LightGBM is conservative (high specificity/precision) while GRU-D is more sensitive (fewer missed progressors), a clinically meaningful difference invisible to ranking metrics alone.

---

## Repository structure

The scripts are numbered by execution stage. Each writes its outputs to `results/` (git-ignored) and is designed to run standalone from the project root.

| Stage | Script | Role |
|------|--------|------|
| 1 | `oasis_prognosis_data.py` | Leakage-free data module: builds the longitudinal tensor (values, mask, time deltas), defines the progression label, and provides patient-level splitting and a leak-free imputer/scaler. **All other scripts import from this.** |
| 2 | `lgbm_baseline.py` | Strong LightGBM baseline on per-patient aggregated features (last / mean / slope / missing-rate). Native NaN handling — no imputation, no leakage. Repeated stratified CV + bootstrap CIs. |
| 3 | `grud_model.py` | The main deep model: a GRU-D cell (Che et al., 2018) modelling missingness, irregular time, and recurrence directly in the architecture. Observed-only standardization; CV + test evaluation. |
| 4 | `transformer_ablation.py` | A small Transformer ablation using leak-free imputed inputs, to test whether extra attention-based capacity helps. |
| 5 | `xai_grud.py` | **Scientific core.** Five attribution methods on GRU-D (saliency, SmoothGrad, gradient×input, Integrated Gradients, occlusion); cross-method agreement; MC-dropout uncertainty; agreement-vs-uncertainty/correctness hypothesis test. |
| 6 | `sanity_grud.py` | Adebayo-style sanity checks: weight randomization (averaged over 5 re-inits), cascading randomization, and label-permutation. Lower similarity = more model-dependent = more trustworthy. |
| 7 | `faithfulness_grud.py` | Faithfulness metrics: deletion/insertion AOPC, comprehensiveness, sufficiency, and a faithfulness correlation (occlusion excluded as circular). |
| 8 | `multi_seed_experiment.py` | Repeats the full GRU-D experiment over K seeds (default 20) with means ± 95% CI, separating robust findings from single-split effects. |
| 9 | `treeshap_lgbm.py` | Exact TreeSHAP on LightGBM, aggregated to feature level, plus cross-model agreement with the GRU-D attributions. |
| 10 | `make_figures.py` | Main publication figures (300 DPI): model comparison, ROC/PR, agreement matrix, sanity, faithfulness, hypothesis, TreeSHAP, calibration. |
| 11 | `make_figures_extended.py` | Supplementary figures: CONSORT flow, per-feature missingness, swimmer plot, decision-curve analysis, SHAP beeswarm, temporal attribution profile. |
| 12 | `missingness_robustness.py` | Nested-feature ablation (full / no-missrate / no-leak / values-only) over K seeds, isolating the marginal contribution of missingness and the sequence-length control. |

Two additional figure scripts accompany the paper:

- `make_paradox_figure.py` — the signature *sanity ↔ faithfulness* quadrant map.
- `confusion_matrices.py` — confusion matrices for all three models on the shared test set.

---

## Data

This code targets the **OASIS-3** longitudinal cohort. The integrated CSV is expected at `data/integrated_data.csv` with one row per visit and columns including `OASISID`, `days_to_visit`, cognitive/clinical measures, and the `CDRTOT` / `Label` fields used to derive progression.

> **The dataset is not included in this repository.** OASIS-3 is released under a data-use agreement and cannot be redistributed. Request access at [OASIS-3](https://www.oasis-brains.org/) and place your integrated CSV at `data/integrated_data.csv`. The `data/` folder and all `*.csv` files are git-ignored.

The task is defined only over participants who are **non-demented at baseline**, have a minimum number of input visits, and a valid prediction window; the label is progression to dementia within the follow-up window. CDR-derived variables are deliberately excluded from the model inputs to prevent target leakage.

---

## Installation

```bash
# clone
git clone https://github.com/MiladBarani/oasis3_prognosis.git
cd <repo-name>

# create an environment (example: venv)
python -m venv oasis3
# Windows:
.\oasis3\Scripts\Activate.ps1
# macOS/Linux:
source oasis3/bin/activate

# install
pip install -r requirements.txt
```

For an exact, pinned environment (recommended for reproduction) use `requirements_lock.txt` instead. Core dependencies: NumPy, pandas, scikit-learn, SciPy, TensorFlow/Keras 3, LightGBM, SHAP, matplotlib. Developed on Python 3.13.

---

## Reproducing the results

Run from the project root, passing the dataset path. The models must be trained (stages 2–4) before the explanation scripts (5–9) can load them.

```bash
# 1. sanity-check the data module
python oasis_prognosis_data.py data/integrated_data.csv

# 2–4. models
python lgbm_baseline.py        data/integrated_data.csv
python grud_model.py           data/integrated_data.csv
python transformer_ablation.py data/integrated_data.csv

# 5–9. explanation audit (require the trained GRU-D artifacts from stage 3)
python xai_grud.py
python sanity_grud.py          data/integrated_data.csv
python faithfulness_grud.py
python treeshap_lgbm.py        data/integrated_data.csv
python multi_seed_experiment.py data/integrated_data.csv 20

# 12. robustness
python missingness_robustness.py data/integrated_data.csv 20

# 10–11 + extras. figures
python make_figures.py
python make_figures_extended.py data/integrated_data.csv
python make_paradox_figure.py
python confusion_matrices.py   data/integrated_data.csv
```

All numeric outputs are written as JSON/CSV under `results/`, and all figures as 300-DPI PNGs under `results/figures/`. Both folders are git-ignored and regenerated by the commands above.

> **Windows note:** if you hit a `UnicodeEncodeError` on the ✓ characters in console output, set `PYTHONIOENCODING=utf-8` in your shell before running.

---

## Reproducibility notes

- **Leakage control is the defining constraint.** Splitting is at the patient level; standardization, imputation, and feature statistics are fit on training data only; CDR-derived variables are excluded from inputs.
- **Multi-seed by default.** Headline GRU-D numbers are reported as mean ± 95% CI over 20 seeds; correlation findings report the number of seeds in which they remain significant.
- **Deterministic vs. stochastic models.** LightGBM is deterministic; the neural models (GRU-D, Transformer) show small run-to-run variation from CPU non-determinism in TensorFlow, so exact decimals may differ slightly between runs while the qualitative pattern holds.

---

## Citation

If this code is useful in your work, please cite the accompanying paper (details to be added upon publication).

```bibtex
@article{placeholder,
  title   = {A Leakage-Free Benchmark and Multi-Dimensional Evaluation of Explainable AI for Trustworthy Dementia-Progression Prediction},
  author  = {Jafari Barani M, Garcia-Diaz V, Nunez-Valdez ER, Espana Lopera JC},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

No license is specified yet. Until a license file is added, the code is under exclusive copyright by default; contact the author before reuse. (Consider adding an OSI-approved license such as MIT or Apache-2.0 before public release.)

## Acknowledgements

Data provided by **OASIS-3** (Open Access Series of Imaging Studies). Attribution methods and validation protocols build on the work of Adebayo et al. (sanity checks) and Che et al. (GRU-D), among others cited in the manuscript.
