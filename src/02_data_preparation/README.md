# 02 — Data Preparation

The data preparation code is not included in this repository. This page documents the steps and decisions as presented in the final project presentation.

All imputation, clipping, and scaling parameters were fit on the training data only and then applied to the test data, to avoid data leakage.

## Input

| Dataset | Rows | Columns | Target included |
| --- | --- | --- | --- |
| Training | 8,038 | 19 | Yes |
| Test | 1,962 | 18 | No (used only for final inference) |

## Cleaning

| Issue | Variables | Treatment |
| --- | --- | --- |
| Missing categorical values | Exercise Frequency, Type of Treatment | Filled with `"Unknown"` |
| Missing numerical values | Number of Prior Visits, Medications Prescribed | Training-set median |
| Implausible values | Age > 100 | Set to missing, then median-imputed |
| Outliers | Height, BMI, Weight, Adjusted Weight, Length of Stay | Clipped at training-set 1st/99th percentiles |
| Duplicates | All rows, PatientID | None found |

After cleaning, no missing values remained in either dataset.

## Feature engineering

| Step | Variables | Purpose |
| --- | --- | --- |
| Log transform (`log1p`) | Prior Visits, Medications Prescribed, Length of Stay | Reduce skewness |
| Same Day Discharge | Length of Stay = 0 | Capture same-day discharge pattern |
| Comorbidity Count | Has Diabetes + Has Hypertension | Measure chronic condition burden |
| BMI Category | BMI | Capture nonlinear BMI risk |
| Age Group | Age | Capture age-band differences |
| High Prior Utilization | Prior Visits ≥ 5 | Identify high-use patients |
| One-hot encoding | Categorical variables | Convert categories to model inputs; train and test columns aligned |
| Standardization | Continuous numeric variables | `StandardScaler` fit on training data only |

## Feature selection and partitioning

- Removed `PatientID` (identifier, not predictive) and separated the target variable.
- Kept clinical, demographic, hospital, lifestyle, and treatment variables, plus both original and log-transformed counts.
- Split the prepared training data 80/20 into training and validation sets with stratified sampling (`random_state=42`), because only about 17.34% of patients were readmitted.
- No oversampling or undersampling was applied at this stage.

## Output

| Dataset | Rows | Features |
| --- | --- | --- |
| Prepared training features | 8,038 | 39 |
| Prepared test features | 1,962 | 39 |
| Training split | 6,430 | 39 |
| Validation split | 1,608 | 39 |
