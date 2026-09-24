# Automata-Based Feature Engineering for Obesity Prediction

This repository contains the obesity dataset and feature-engineered datasets used in an automata-based obesity prediction study. The feature-engineering process converts original clinical, behavioural, and lifestyle attributes into compact and interpretable representations that can be used for machine-learning and automata-based predictive modelling.

## Dataset Files

### `Obesity_DataSet_2.csv`
Original obesity dataset used as the source dataset.

- Records: 7,481
- Attributes: 16
- Target variable: `BMI_WHO`
- Includes demographic, clinical, behavioural, and lifestyle-related variables such as age, gender, physical activity, smoking history, blood pressure, total cholesterol, diabetes status, height, and depression-related information.

### `16-22.csv`
Original dataset enriched with six derived features generated during feature engineering.

Additional engineered features:

- `Age_Group`
- `Chol_Category`
- `BP_Category`
- `PhysActive_Binary`
- `LongTerm_Smoker`
- `Age_Activity_Interaction`

This representation preserves the original variables while adding structured health-state information.

### `Obesity_automata_based_6_features.csv`
Compact feature-engineered representation containing six obesity-related derived indicators and the target variable.

Features:

- `Obesity_Risk_Score`
- `Physical_Activity_Score`
- `Metabolic_Risk_Index`
- `Cardiovascular_Risk_Score`
- `Age_Group`
- `Unhealthy_Lifestyle_Index`
- `BMI_WHO` — target variable

## Feature-Engineering Workflow

```text
Original Obesity Dataset
        |
        v
Data Preprocessing
        |
        v
Clinical and Behavioural Feature Transformation
        |
        +-----------------------------+
        |                             |
        v                             v
Original + Derived Features     Compact Engineered Features
        |                             |
        v                             v
     16-22.csv           Obesity_automata_based_6_features.csv
```

The overall research framework further evaluates three modelling branches:

1. **Baseline branch:** original obesity-related features with conventional feature selection.
2. **Automata-merged branch:** original features combined with automata-derived features.
3. **Automata-integrated branch:** derived features further processed through weighted, fuzzy, and hybrid automata representations.

## Target Variable

`BMI_WHO` is used as the obesity-classification target.

Records with missing target labels should be excluded before supervised model training. In the current study, 7,384 labelled records are retained from the original 7,481 records.

## Purpose

These datasets support experiments on:

- obesity-risk prediction,
- feature-engineering comparison,
- automata-based health-state representation,
- machine-learning classification,
- weighted and fuzzy risk modelling,
- hybrid automata integration, and
- ensemble-based prediction.

## Requirements

Typical Python packages used with these datasets include:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## Example Loading Code

```python
import pandas as pd

df = pd.read_csv("Obesity_DataSet_2.csv")
print(df.shape)
print(df.columns)
```

## Notes

- The files contain research-oriented feature representations derived from the same obesity dataset.
- Missing target labels should be removed before supervised classification.
- Class balancing should preferably be performed only on the training portion of each cross-validation fold to avoid data leakage.
- Feature selection and data-driven feature-weight estimation should also be fitted using training data during final model evaluation.

## Research Use

The repository is intended for academic and research use related to obesity prediction, healthcare analytics, machine learning, and automata-based feature engineering.
