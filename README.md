# Predicting 30-Day Hospital Readmission in Patients with Diabetes

**EXSU 500: Fundamentals of Artificial Intelligence in Medicine, Fall 2026, McGill University**
Team project. Team: 9

> **Course prototype for research and educational purposes only. Not a medical device. Not for clinical use.**

## Team and roles

| Member | Role | GitHub |
|---|---|---|
| Imen Benadda | Data lead | @Imenbenadda |
| Salar Azadani | Modelling lead | @Salar-MTL |
| Justin Riffou | Interface lead | @Jriffou |
| Momina Ejaz | Writing lead | @Mominae-ui |

Every member contributes to code and manuscript; roles indicate who is accountable for each area. A CRediT contribution statement is in [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## 1. Dataset

- **Name:** Diabetes 130-US Hospitals for Years 1999-2008
- **Source:** UCI Machine Learning Repository (dataset ID 296, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J)), https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Provenance:** De-identified inpatient records from 130 US hospitals and integrated delivery networks (1999-2008), extracted from the Cerner Health Facts EHR database. Encounters were included by the creators only if they were (1) inpatient admissions, (2) with any diabetes diagnosis, (3) a length of stay of 1-14 days, (4) laboratory tests performed and (5) medications administered.
- **Original publication:** Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014, Article ID 781670.
- **Dataset citation:** Clore J, Cios K, DeShazo J, Strack B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
- **Licence:** CC BY 4.0 (confirmed on the UCI dataset page): reuse and adaptation are allowed for any purpose with attribution. The data are public and de-identified; no access agreement is needed. UCI flags age, gender and race as sensitive attributes; no re-identification or linkage will be attempted.
- **Access:** Downloaded programmatically, not committed to this repository:

```python
from ucimlrepo import fetch_ucirepo

ds = fetch_ucirepo(id=296)
X = ds.data.features
y = ds.data.targets
```

### Size, variables and outcome distribution

**101,766 encounters** from about 71,500 unique patients, with 2 identifiers (`encounter_id`, `patient_nbr`), **47 candidate predictors** and 1 outcome.

| Group | Variables |
|---|---|
| Demographics (4) | race, gender, age (10-year bands), weight |
| Admission context (6) | admission_type_id, admission_source_id, discharge_disposition_id, time_in_hospital, payer_code, medical_specialty |
| Utilisation (7) | num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient (prior year), number_diagnoses |
| Diagnoses (3) | diag_1, diag_2, diag_3 (ICD-9 codes, 700+ distinct values) |
| Glycaemic labs (2) | max_glu_serum, A1Cresult |
| Diabetes medications (23) | metformin, insulin, glipizide, glyburide, pioglitazone, rosiglitazone and 17 others (No / Steady / Up / Down) |
| Medication summary (2) | change, diabetesMed |

| `readmitted` | Encounters | Share | Binary target |
|---|---|---|---|
| NO (no readmission recorded) | 54,864 | 53.9% | 0 |
| >30 days | 35,545 | 34.9% | 0 |
| <30 days | 11,357 | 11.2% | **1** |

These are the standard counts of the UCI file; `src/data.py` recomputes and logs them at load.

## 2. Problem statement

**Task:** Predict, at the time of discharge, whether an adult inpatient with diabetes will be **readmitted to hospital within 30 days**, using only information available at or before discharge (demographics, admission type, length of stay, diagnoses, procedures, lab and medication information).

**Task type:** Binary classification (readmitted within 30 days vs. not), with class imbalance (~11% positive).

**Why it matters:** Unplanned 30-day readmissions are costly and are widely used as a hospital quality indicator. Patients with diabetes are readmitted often. A well-calibrated risk score could help clinical teams decide which patients to prioritise for low-cost transitional-care actions (follow-up call within 72 h, medication reconciliation, early clinic visit), so limited resources go to those most likely to benefit.

**Outcome definition:** The source variable `readmitted` has three values (`<30`, `>30`, `NO`). We define the positive class as `<30` and both `>30` and `NO` as negative. `NO` means no readmission was recorded within the same health-system data (see Limitations).

## 3. Planned approach

### Feature engineering (fit on training data only)

- `weight` dropped (~97% missing); `payer_code` and `medical_specialty` kept with "Missing" as its own level; rare levels grouped (medical_specialty: top 10 + Other + Missing).
- `diag_1`-`diag_3` mapped from ICD-9 to the 9 clinical groups of Strack et al. (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other), plus a flag for diabetes with complications.
- Admission type, admission source and discharge disposition grouped into clinically meaningful levels; "Not available" / "NULL" / "Not mapped" merged into one Unknown level per variable.
- `A1Cresult` and `max_glu_serum` "None" (not measured) kept as an informative level.
- Age band as ordinal; utilisation counts log1p-transformed; total prior visits derived.
- The 23 medications summarised as insulin status, number of diabetes drugs and any dose change; drugs used in <1% of encounters dropped.

### Models (compared under one protocol)

1. **Statistical baseline:** majority-class prediction and a two-variable rule (prior inpatient visits + length of stay) as floor references; **L2-penalised logistic regression** (`class_weight="balanced"`, C tuned on a log grid, elastic-net as a variant) as the main baseline. Logistic regression models the probability of a binary outcome and gives interpretable odds ratios. A prespecified A1Cresult x primary-diagnosis interaction replicates the HbA1c finding of Strack et al.
2. **Classical ML:** **LightGBM** with native categorical handling (raw ICD-9 3-digit codes and specialty as categoricals), `scale_pos_weight` for imbalance, tuned with Optuna (100 trials) on cross-validated AUPRC, early stopping on validation AUPRC. Random forest (500 trees, balanced weights) as a secondary comparison.
3. **Deep learning:** an **MLP with learned categorical embeddings** (diag_1-diag_3 share one ICD-9 embedding table), 3 hidden blocks (256 -> 128 -> 64, BatchNorm, ReLU, dropout), weighted binary cross-entropy (focal loss as an alternative), AdamW, early stopping on validation AUPRC; tuned with Optuna (40 trials) in PyTorch. FT-Transformer is an optional stretch model. We will report honestly if a well-tuned classical model matches or beats it on this tabular data.

An average of calibrated LightGBM and MLP probabilities may be evaluated as an optional ensemble.

### Evaluation protocol (declared before modelling)

- **Cohort filtering:** Exclude encounters ending in death or hospice discharge (`discharge_disposition_id` 11, 13, 14, 19, 20, 21), since these patients cannot be readmitted. The main analysis keeps all remaining encounters; a sensitivity analysis keeps only the first encounter per patient (~70,000 encounters, ~9% positive). All exclusions are counted and reported.
- **Splits:** **70 / 15 / 15** train / validation / held-out test, made **at the patient level** (`StratifiedGroupKFold`, grouped on `patient_nbr`, stratified on the outcome) so no patient appears in more than one partition. Proportions and the random seed are fixed in `src/splits.py` and reported in the manuscript.
- **Tuning:** 5-fold patient-grouped stratified cross-validation inside the training set. The validation set is used once per model for early stopping, calibration and threshold selection. The test set is evaluated once at the end, after all choices are frozen.
- **Sensitivity split:** a pseudo-temporal split by `encounter_id` order (earliest 80% train, latest 20% test) as a rough drift check, since true dates are not available.
- **Metrics:** **AUPRC (primary)** and **AUROC (co-primary)**; Brier score and calibration slope/intercept; sensitivity, specificity, PPV, NPV and F1 at the operating threshold; alerts per 100 discharges; decision-curve net benefit over 5-30% risk thresholds. Accuracy is not used (predicting "no readmission" for everyone is ~89% accurate).
- **Operating threshold:** chosen on calibrated validation probabilities, not defaulted to 0.5. Primary rule: the highest threshold reaching **sensitivity >= 0.70**, favouring sensitivity because the follow-up intervention is low-cost. Secondary views: top 20% highest risk (fixed team capacity) and the F2-optimal threshold. Each model gets its own threshold from the same rule.
- **Uncertainty:** 2,000 patient-level bootstrap resamples of the test set for 95% CIs; paired bootstrap (and DeLong for AUROC) for model differences; LightGBM and MLP retrained over 5 random seeds (mean +/- SD).
- **Calibration:** reliability curves (10 quantile bins), calibration slope and intercept, expected calibration error and Brier score. LightGBM and the MLP are recalibrated on validation data (isotonic regression, Platt scaling if isotonic overfits); logistic regression is recalibrated only if needed.
- **Subgroups / fairness:** metrics and calibration by sex, age band (<60, 60-79, >=80), race, payer group and admission type, with CIs; groups with fewer than ~100 positives flagged as unreliable.
- **Interpretability:** odds ratios with CIs for logistic regression; TreeSHAP (global importance, dependence and per-patient waterfall plots) for LightGBM; KernelSHAP or Integrated Gradients plus permutation importance for the MLP; error analysis of the best model. Drivers reflecting access to care (e.g. payer code) are reported and tested by refitting without them.
- **Leakage controls:** all imputation, rare-level grouping, encoding and scaling are fit inside the training data only (scikit-learn pipelines). Imbalance is handled with class weights during training only; validation and test sets are never resampled. Features not available at discharge are excluded.

## 4. Repository layout (planned)

```
.
├── README.md
├── LICENSE
├── MODEL_CARD.md
├── CONTRIBUTIONS.md          # CRediT + Use of Generative AI statement
├── environment.yml / requirements.txt   # pinned versions
├── src/
│   ├── data.py               # download + cohort definition + feature engineering
│   ├── splits.py             # patient-level splits
│   ├── train_baseline.py     # majority class + logistic regression
│   ├── train_gbm.py          # LightGBM (+ random forest comparison)
│   ├── train_mlp.py          # MLP with categorical embeddings
│   ├── calibrate.py          # recalibration + threshold selection on validation
│   └── evaluate.py           # one-time test evaluation, bootstrap CIs, subgroups
├── notebooks/                # EDA and figures
├── results/                  # tables and figures (no raw data)
└── app/                      # optional interface
```

The dataset itself and model weights larger than about 100 MB are **not** committed. No credentials or tokens are stored in this repository.

## 5. How to reproduce

> To be completed before the final submission.

```bash
git clone <REPO URL>
cd <REPO NAME>
pip install -r requirements.txt
python src/data.py          # downloads and prepares the data
python src/splits.py        # creates the fixed patient-level splits
python src/train_baseline.py
python src/train_gbm.py
python src/train_mlp.py
python src/calibrate.py     # calibration + operating threshold on validation
python src/evaluate.py      # runs the one-time test evaluation
```

## 6. Results

> To be completed. Final README will include the headline model comparison table with confidence intervals.

| Model | AUPRC (95% CI) | AUROC (95% CI) | Brier score | Sensitivity @ threshold | Specificity @ threshold | PPV @ threshold |
|---|---|---|---|---|---|---|
| Majority class | | | | | | |
| Logistic regression | | | | | | |
| LightGBM | | | | | | |
| MLP (embeddings) | | | | | | |

## 7. Interface

> Optional. If deployed, the public URL will be listed here. The interface will display the prototype notice at the top of this file at all times.

## 8. Limitations (preliminary)

- Data are from 1999-2008 US hospitals; practice patterns, ICD-9 coding (now ICD-10) and diabetes drugs (e.g. SGLT2 inhibitors, GLP-1 receptor agonists) have changed.
- Readmission is only observed within the same health-system data, so readmissions elsewhere are recorded as `NO`; the true 30-day rate is likely underestimated.
- Inclusion criteria (1-14-day stays with labs and medications) exclude very short, very long and low-acuity admissions.
- Heavy missingness in several fields: weight (~97%), medical_specialty (~49%), payer_code (~40%); HbA1c not measured in ~83% of encounters and serum glucose in ~95%.
- About 30% of encounters belong to repeat patients, which is why splits are grouped by patient.
- No exact dates, so a true temporal validation is not possible.
- Social, outpatient and detailed clinical factors (vital signs, most lab values, comorbidity scores) are not available, which caps achievable discrimination (published AUROCs on this dataset are typically ~0.65-0.70).
- The cohort is mostly Caucasian (~75%) and African American (~19%); other groups are small, so subgroup estimates are unstable. Race collection methods are not documented, and payer code may encode access to care rather than clinical risk.
- Performance on this dataset does not imply performance in any other hospital or population.
