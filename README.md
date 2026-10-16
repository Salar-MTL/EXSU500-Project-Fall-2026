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
- **Source:** UCI Machine Learning Repository (dataset ID 296), https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Original publication:** Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014.
- **Content:** About 100,000 inpatient encounters of patients with diabetes across 130 US hospitals over 10 years (1999-2008), with roughly 50 features covering demographics, admission and discharge details, diagnoses, laboratory testing, medications, and the readmission outcome.
- **Licence:** CC BY 4.0 (as listed by UCI; the team will re-confirm on the dataset page before the final release). The data are public and de-identified. No access agreement is needed.
- **Access:** Downloaded programmatically, not committed to this repository:

```python
from ucimlrepo import fetch_ucirepo

ds = fetch_ucirepo(id=296)
X = ds.data.features
y = ds.data.targets
```

## 2. Problem statement

**Task:** Predict, at the time of discharge, whether an adult inpatient with diabetes will be **readmitted to hospital within 30 days**, using only information available at or before discharge (demographics, admission type, length of stay, diagnoses, procedures, lab and medication information).

**Task type:** Binary classification (readmitted within 30 days vs. not), with class imbalance.

**Why it matters:** Unplanned 30-day readmissions are costly and are widely used as a hospital quality indicator. Patients with diabetes are readmitted often. A well-calibrated risk score could help clinical teams decide which patients to prioritise for follow-up calls, medication review or discharge planning, so limited resources go to those most likely to benefit.

**Outcome definition:** The source variable `readmitted` has three values (`<30`, `>30`, `NO`). We define the positive class as `<30` and all other values as negative.

## 3. Planned approach

### Models (compared under one protocol)

1. **Baseline:** majority-class prediction and logistic regression.
2. **Classical ML:** gradient boosting (LightGBM or XGBoost); random forest as a possible comparison.
3. **Deep learning:** an MLP with embeddings for categorical features. We will report honestly if a well-tuned classical model matches or beats it on this tabular data.

### Evaluation protocol (declared before modelling)

- **Splits:** train / validation / held-out test, made **at the patient level** (grouped on `patient_nbr`), never at the encounter level, because patients appear in multiple encounters. Split proportions and the random seed are fixed in `src/splits.py` and reported in the manuscript.
- **Cohort filtering:** Exclude encounters ending in death or hospice discharge (patients who cannot be readmitted) and keep one clearly documented rule for repeat encounters. All exclusions are counted and reported.
- **Tuning:** on validation only. The test set is evaluated once at the end.
- **Metrics:** AUROC and AUPRC; sensitivity and specificity at a stated operating threshold chosen on validation data from the clinical trade-off, not defaulted to 0.5.
- **Uncertainty:** bootstrap confidence intervals on the test set and repeated runs over several random seeds.
- **Interpretability / calibration:** reliability (calibration) curve and SHAP feature importance for the best model, plus error analysis.
- **Leakage controls:** all imputation, scaling, encoding and any resampling are fit inside the training data only (pipelines). Features not available at discharge are excluded.

## 4. Repository layout (planned)

```
.
├── README.md
├── LICENSE
├── MODEL_CARD.md
├── CONTRIBUTIONS.md          # CRediT + Use of Generative AI statement
├── environment.yml / requirements.txt   # pinned versions
├── src/
│   ├── data.py               # download + cohort definition
│   ├── splits.py             # patient-level splits
│   ├── train_baseline.py
│   ├── train_gbm.py
│   ├── train_mlp.py
│   └── evaluate.py
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
python src/train_baseline.py
python src/train_gbm.py
python src/train_mlp.py
python src/evaluate.py      # runs the one-time test evaluation
```

## 6. Results

> To be completed. Final README will include the headline model comparison table with confidence intervals.

| Model | AUROC (95% CI) | AUPRC (95% CI) | Sensitivity @ threshold | Specificity @ threshold |
|---|---|---|---|---|
| Baseline | | | | |
| Gradient boosting | | | | |
| MLP | | | | |

## 7. Interface

> Optional. If deployed, the public URL will be listed here. The interface will display the prototype notice at the top of this file at all times.

## 8. Limitations (preliminary)

- Data are from 1999-2008 US hospitals; practice patterns, coding and populations have changed.
- Readmission is only observed within the same health system data, so readmissions elsewhere are missed.
- Several features have heavy missingness (for example weight and some lab fields).
- Social and outpatient factors that strongly influence readmission are not available.
- Performance on this dataset does not imply performance in any other hospital or population.
