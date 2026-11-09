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
- **Source:** UCI Machine Learning Repository (dataset ID 296, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J)) [1]
- **Provenance:** De-identified inpatient records from 130 US hospitals (1999-2008), extracted from the Cerner Health Facts EHR database [2]. Encounters were included by the creators only if they were (1) inpatient admissions, (2) with any diabetes diagnosis, (3) a length of stay of 1-14 days, (4) laboratory tests performed and (5) medications administered [1].
- **Original publication:** Strack et al., 2014 [2].
- **Licence:** CC BY 4.0: reuse and adaptation are allowed for any purpose with attribution [1]. The data are public and de-identified; no access agreement is needed. UCI flags age, gender and race as sensitive attributes [1]; no re-identification or linkage will be attempted.
- **Access:** Downloaded programmatically, not committed to this repository:

```python
from ucimlrepo import fetch_ucirepo

ds = fetch_ucirepo(id=296)
X = ds.data.features
y = ds.data.targets
```

### Size, variables and outcome distribution

**101,766 encounters** (rows; one row = one hospital stay) from **71,518 unique patients**, with 2 identifiers (`encounter_id`, `patient_nbr`), **47 candidate predictors** and 1 outcome [1]. All counts in this README were computed by the team from the UCI file `diabetic_data.csv`.

| Group | Variables |
|---|---|
| Demographics (4) | race, gender, age (10-year bands), weight |
| Admission context (6) | admission_type_id, admission_source_id, discharge_disposition_id, time_in_hospital, payer_code, medical_specialty |
| Utilisation (7) | num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient (prior year), number_diagnoses |
| Diagnoses (3) | diag_1, diag_2, diag_3 (ICD-9 codes; 716 distinct primary codes) |
| Glycaemic labs (2) | max_glu_serum, A1Cresult |
| Diabetes medications (23) | metformin, insulin, glipizide, glyburide, pioglitazone, rosiglitazone and 17 others (No / Steady / Up / Down) |
| Medication summary (2) | change, diabetesMed |

| `readmitted` | Encounters | Share | Binary target |
|---|---|---|---|
| NO (no readmission recorded) | 54,864 | 53.9% | 0 |
| >30 days | 35,545 | 34.9% | 0 |
| <30 days | 11,357 | 11.2% | **1** |

`src/data.py` recomputes and logs these counts at load.

## 2. Problem statement

**Task:** Predict, at the time of discharge, whether an adult inpatient with diabetes will be **readmitted to hospital within 30 days**, using only information available at or before discharge (demographics, admission type, length of stay, diagnoses, procedures, lab and medication information).

**Task type:** Binary classification (readmitted within 30 days vs. not), with class imbalance (11.4% positive in the analytic cohort).

**Why it matters:** Unplanned 30-day readmission is a standard hospital quality measure, used by Medicare to target avoidable readmissions through better discharge planning [3]. Transitional-care interventions reduce 30-day readmission (pooled relative risk 0.82) [4]. A well-calibrated risk score could help clinical teams decide which patients to prioritise for these actions (follow-up call, medication reconciliation, early clinic visit), so limited resources go to those most likely to benefit.

**Outcome definition:** The source variable `readmitted` has three values (`<30`, `>30`, `NO`). We define the positive class as `<30` and both `>30` and `NO` as negative. `NO` means no readmission was recorded within the source database (see Limitations).

**Why binary rather than three classes:** 30 days is the standard quality window [3], preventability is highest soon after discharge [5], `>30` and `NO` imply the same discharge decision, and Strack et al. used the same definition [2]. A three-class model is kept as a sensitivity analysis.

## 3. Planned approach

### Feature engineering (fit on training data only)

- `weight` dropped (96.9% missing); `payer_code` and `medical_specialty` kept with "Missing" as its own level; rare levels grouped (medical_specialty: top 10 + Other + Missing).
- `diag_1`-`diag_3` mapped from ICD-9 to the 9 clinical groups of Strack et al. [2] (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other), plus a flag for diabetes with complications.
- Admission type, admission source and discharge disposition grouped into clinically meaningful levels; "Not available" / "NULL" / "Not mapped" merged into one Unknown level per variable.
- `A1Cresult` and `max_glu_serum` "None" (not measured) kept as an informative level.
- Age band as ordinal; utilisation counts log1p-transformed; total prior visits derived.
- The 23 medications summarised as insulin status, number of diabetes drugs and any dose change; drugs used in <1% of encounters (15 of 23) dropped.

### Models (compared under one protocol)

1. **Statistical baseline:** majority-class prediction and a two-variable rule (prior inpatient visits + length of stay, in the spirit of the LACE index [23]) as floor references; **L2-penalised logistic regression** (`class_weight="balanced"`, C tuned on a log grid, elastic-net as a variant) as the main baseline. Logistic regression models the probability of a binary outcome, gives interpretable odds ratios, and is the model Strack et al. used on this dataset [2]. A prespecified A1Cresult x primary-diagnosis interaction replicates their HbA1c finding [2].
2. **Classical ML:** **LightGBM** [24] with native categorical handling (raw ICD-9 3-digit codes and specialty as categoricals), `scale_pos_weight` for imbalance, tuned with Optuna [26] (100 trials) on cross-validated AUPRC, early stopping on validation AUPRC. Tree ensembles remain state of the art on medium-sized tabular data [25]. Random forest [27] (500 trees, balanced weights) as a secondary comparison.
3. **Deep learning:** an **MLP with learned categorical embeddings** [28] (diag_1-diag_3 share one ICD-9 embedding table), 3 hidden blocks (256 -> 128 -> 64, BatchNorm, ReLU, dropout), weighted binary cross-entropy (focal loss [29] as an alternative), AdamW [30], early stopping on validation AUPRC; tuned with Optuna (40 trials) in PyTorch. FT-Transformer [31] is an optional stretch model. Deep models rarely beat boosted trees on tabular data [25]; on this dataset a DNN reached AUROC 0.579 vs. 0.667 for XGBoost [8]. We will report honestly if a well-tuned classical model matches or beats ours.

An average of calibrated LightGBM and MLP probabilities may be evaluated as an optional ensemble.

### Evaluation protocol (declared before modelling)

- **Cohort filtering:** Exclude encounters ending in death or hospice discharge (`discharge_disposition_id` 11, 13, 14, 19, 20, 21; 2,423 encounters), since these patients cannot be readmitted [2], and age bands under 20 years (852 encounters) to match the adult target population. This leaves **98,493 encounters from 69,314 patients (11.4% positive)**. A sensitivity analysis keeps only the first encounter per patient (69,314 encounters, 9.0% positive). All exclusions are counted and reported.
- **Splits:** **70 / 15 / 15** train / validation / held-out test, made **at the patient level** (`StratifiedGroupKFold` [11], grouped on `patient_nbr`, stratified on the outcome) so no patient appears in more than one partition; record-wise splits overstate performance [6]. Proportions and the random seed are fixed in `src/splits.py` and reported in the manuscript.
- **Tuning:** 5-fold patient-grouped stratified cross-validation inside the training set. The validation set is used once per model for early stopping, calibration and threshold selection. The test set is evaluated once at the end, after all choices are frozen, following TRIPOD+AI [12].
- **Sensitivity split:** a pseudo-temporal split by `encounter_id` order (earliest 80% train, latest 20% test) as a rough drift check, since true dates are not available [1].
- **Metrics:** **AUPRC (primary)**, because it is more informative than ROC when positives are rare [14], and **AUROC (co-primary)** for comparison with published work [8]; Brier score and calibration slope/intercept [15]; sensitivity, specificity, PPV, NPV and F1 at the operating threshold; alerts per 100 discharges; decision-curve net benefit over 5-30% risk thresholds [16]. Accuracy is not used (predicting "no readmission" for everyone is 88.8% accurate).
- **Operating threshold:** chosen on calibrated validation probabilities, not defaulted to 0.5. Primary rule: the highest threshold reaching **sensitivity >= 0.70**, favouring sensitivity because transitional-care interventions are low-cost and effective [4]. Secondary views: top 20% highest risk (fixed team capacity) and the F2-optimal threshold. Each model gets its own threshold from the same rule.
- **Uncertainty:** 2,000 patient-level bootstrap resamples of the test set for 95% CIs [17]; paired bootstrap (and DeLong [18] for AUROC) for model differences; LightGBM and MLP retrained over 5 random seeds (mean +/- SD).
- **Calibration:** reliability curves (10 quantile bins), calibration slope and intercept, expected calibration error and Brier score [15]. LightGBM and the MLP are recalibrated on validation data (isotonic regression, Platt scaling if isotonic overfits) [19], since class weighting inflates predicted risk [13]; logistic regression is recalibrated only if needed.
- **Subgroups / fairness:** metrics and calibration by sex, age band (<60, 60-79, >=80), race, payer group and admission type, with CIs [10]; groups with fewer than ~100 positives flagged as unreliable.
- **Interpretability:** odds ratios with CIs for logistic regression; SHAP [20] (TreeSHAP [21] global importance, dependence and per-patient waterfall plots) for LightGBM; Integrated Gradients [22] plus permutation importance for the MLP; error analysis of the best model. Drivers reflecting access to care (e.g. payer code) are reported and tested by refitting without them [10].
- **Leakage controls:** all imputation, rare-level grouping, encoding and scaling are fit inside the training data only (scikit-learn pipelines [11]). Imbalance is handled with class weights during training only, not resampling, because resampling distorts predicted risks [13]. Features not available at discharge are excluded.

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

- Data are from 1999-2008 US hospitals, coded in ICD-9, with no columns for newer drug classes such as SGLT2 inhibitors or GLP-1 receptor agonists [1].
- Readmission is only observed within the source database [2], so readmissions elsewhere are recorded as `NO`; the true 30-day rate is likely underestimated.
- Inclusion criteria (1-14-day stays with labs and medications) exclude very short, very long and low-acuity admissions [1].
- Heavy missingness: weight 96.9%, medical_specialty 49.1%, payer_code 39.6%, race 2.2%; HbA1c not measured in 83.3% of encounters and serum glucose in 94.7%.
- 16,773 patients have more than one stay, and 30,248 encounters (29.7%) are repeat stays, which is why splits are grouped by patient [6].
- No exact dates, so a true temporal validation is not possible [1].
- Social, outpatient and detailed clinical factors (vital signs, most lab values, comorbidity scores) are not available. Readmission models generally discriminate poorly [7]; published AUROCs on this dataset are 0.58-0.67 [8].
- The cohort is 74.8% Caucasian and 18.9% African American; other groups are 2% or less each, so subgroup estimates are unstable. Race and sex collection methods are not documented [9], and payer code may encode access to care rather than clinical risk [10].
- Performance on this dataset does not imply performance in any other hospital or population.

## References

1. Clore J, Cios K, DeShazo J, Strack B. [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) [dataset]. UCI Machine Learning Repository; 2014. doi:10.24432/C5230J.
2. Strack B, DeShazo JP, Gennings C, et al. [Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records](https://doi.org/10.1155/2014/781670). *BioMed Res Int.* 2014;2014:781670.
3. Centers for Medicare & Medicaid Services. [Hospital Readmissions Reduction Program (HRRP)](https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp).
4. Leppin AL, Gionfriddo MR, Kessler M, et al. [Preventing 30-day hospital readmissions: a systematic review and meta-analysis of randomized trials](https://doi.org/10.1001/jamainternmed.2014.1608). *JAMA Intern Med.* 2014;174(7):1095-1107.
5. Graham KL, Auerbach AD, Schnipper JL, et al. [Preventability of early versus late hospital readmissions in a national cohort of general medicine patients](https://doi.org/10.7326/M17-1724). *Ann Intern Med.* 2018;168(11):766-774.
6. Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. [The need to approximate the use-case in clinical machine learning](https://doi.org/10.1093/gigascience/gix019). *GigaScience.* 2017;6(5):1-9.
7. Kansagara D, Englander H, Salanitro A, et al. [Risk prediction models for hospital readmission: a systematic review](https://doi.org/10.1001/jama.2011.1515). *JAMA.* 2011;306(15):1688-1698.
8. Emi-Johnson OG, Nkrumah KJ. [Predicting 30-day hospital readmission in patients with diabetes using machine learning on electronic health record data](https://pmc.ncbi.nlm.nih.gov/articles/PMC12085305). *Cureus.* 2025;17(4):e82437.
9. [readmission: Hospital Readmission Data for Patients with Diabetes](https://cloud.r-project.org/web/packages/readmission/refman/readmission.html) (R package documentation). CRAN.
10. Rajkomar A, Hardt M, Howell MD, Corrado G, Chin MH. [Ensuring fairness in machine learning to advance health equity](https://doi.org/10.7326/M18-1990). *Ann Intern Med.* 2018;169(12):866-872.
11. Pedregosa F, Varoquaux G, Gramfort A, et al. [Scikit-learn: machine learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html). *J Mach Learn Res.* 2011;12:2825-2830.
12. Collins GS, Moons KGM, Dhiman P, et al. [TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods](https://doi.org/10.1136/bmj-2023-078378). *BMJ.* 2024;385:e078378.
13. van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. [The harm of class imbalance corrections for risk prediction models](https://doi.org/10.1093/jamia/ocac093). *J Am Med Inform Assoc.* 2022;29(9):1525-1534.
14. Saito T, Rehmsmeier M. [The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets](https://doi.org/10.1371/journal.pone.0118432). *PLoS One.* 2015;10(3):e0118432.
15. Van Calster B, McLernon DJ, van Smeden M, et al. [Calibration: the Achilles heel of predictive analytics](https://doi.org/10.1186/s12916-019-1466-7). *BMC Med.* 2019;17:230.
16. Vickers AJ, Elkin EB. [Decision curve analysis: a novel method for evaluating prediction models](https://doi.org/10.1177/0272989X06295361). *Med Decis Making.* 2006;26(6):565-574.
17. Efron B, Tibshirani RJ. [*An Introduction to the Bootstrap*](https://doi.org/10.1201/9780429246593). Chapman & Hall; 1993.
18. DeLong ER, DeLong DM, Clarke-Pearson DL. [Comparing the areas under two or more correlated receiver operating characteristic curves](https://doi.org/10.2307/2531595). *Biometrics.* 1988;44(3):837-845.
19. Niculescu-Mizil A, Caruana R. [Predicting good probabilities with supervised learning](https://doi.org/10.1145/1102351.1102430). *Proc ICML.* 2005:625-632.
20. Lundberg SM, Lee SI. [A unified approach to interpreting model predictions](https://arxiv.org/abs/1705.07874). *NeurIPS.* 2017.
21. Lundberg SM, Erion G, Chen H, et al. [From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9). *Nat Mach Intell.* 2020;2:56-67.
22. Sundararajan M, Taly A, Yan Q. [Axiomatic attribution for deep networks](https://arxiv.org/abs/1703.01365). *Proc ICML.* 2017.
23. van Walraven C, Dhalla IA, Bell C, et al. [Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community](https://doi.org/10.1503/cmaj.091117). *CMAJ.* 2010;182(6):551-557.
24. Ke G, Meng Q, Finley T, et al. [LightGBM: a highly efficient gradient boosting decision tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree). *NeurIPS.* 2017.
25. Grinsztajn L, Oyallon E, Varoquaux G. [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) *NeurIPS Datasets and Benchmarks.* 2022.
26. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. [Optuna: a next-generation hyperparameter optimization framework](https://doi.org/10.1145/3292500.3330701). *Proc KDD.* 2019.
27. Breiman L. [Random forests](https://doi.org/10.1023/A:1010933404324). *Mach Learn.* 2001;45:5-32.
28. Guo C, Berkhahn F. [Entity embeddings of categorical variables](https://arxiv.org/abs/1604.06737). arXiv:1604.06737; 2016.
29. Lin TY, Goyal P, Girshick R, He K, Dollár P. [Focal loss for dense object detection](https://doi.org/10.1109/ICCV.2017.324). *Proc ICCV.* 2017.
30. Loshchilov I, Hutter F. [Decoupled weight decay regularization](https://arxiv.org/abs/1711.05101). *ICLR.* 2019.
31. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. [Revisiting deep learning models for tabular data](https://arxiv.org/abs/2106.11959). *NeurIPS.* 2021.
