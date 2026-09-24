# Automata-Based Feature Engineering for Obesity Prediction

This repository contains the datasets and implementation used to study **automata-based feature engineering for obesity prediction**. The project starts from one obesity dataset and constructs three feature-representation branches for comparison: a conventional machine-learning baseline, an automata-merged representation, and a fully automata-integrated representation using weighted, fuzzy, and hybrid automata.

## Repository Structure

```text
obesity_Prediction/
│
├── datasets/
│   ├── Obesity_DataSet_2.csv
│   ├── 16-22.csv
│   └── Obesity_automata_based_6_features.csv
│
├── feature engineering/
│   └── Feature_Engineering.ipynb
│
└── README.md
```

## Dataset

### `datasets/Obesity_DataSet_2.csv`

This is the original dataset used for feature engineering.

- Total records: **7,481**
- Total attributes: **16**
- Target variable: **`BMI_WHO`**
- Labelled records retained for modelling: **7,384**

The target distribution after removing records with missing `BMI_WHO` labels is:

| Class | Records |
|---|---:|
| Obese | 2,623 |
| OverWeight | 2,440 |
| NormWeight | 2,174 |
| UnderWeight | 147 |

The dataset contains demographic, clinical, behavioural, and lifestyle-related variables such as age, gender, race, education, household income, physical activity, smoking history, diabetes status, systolic blood pressure, total cholesterol, alcohol consumption, marital status, work status, height, and depression-related information.

### `datasets/16-22.csv`

This file represents an earlier enriched feature version of the original obesity dataset. It contains the original attributes together with additional derived variables used during feature-engineering development.

### `datasets/Obesity_automata_based_6_features.csv`

This file contains a compact set of engineered obesity-related indicators generated during the earlier development of the automata-based representation.

> The final three-branch implementation used in this repository is provided in `feature engineering/Feature_Engineering.ipynb`.

## Feature Engineering Notebook

### `feature engineering/Feature_Engineering.ipynb`

The notebook implements the complete three-branch feature-engineering workflow.

### Common Preprocessing

The original dataset is loaded and records with missing `BMI_WHO` target labels are removed. The retained dataset contains **7,384 labelled samples**. Categorical variables are encoded when required, while missing input values are handled before modelling.

## Automata-Derived Features

Six state-oriented features are generated from selected clinical and behavioural variables:

| Derived Feature | Rule |
|---|---|
| `Age_State` | Age < 30 → 0; 30–49 → 1; ≥ 50 → 2 |
| `Cholesterol_State` | TotChol < 200 → 0; 200–<240 → 1; ≥ 240 → 2 |
| `BP_State` | Systolic BP < 120 → 0; 120–<140 → 1; ≥ 140 → 2 |
| `PhysActive_Binary` | No → 0; Yes → 1 |
| `Smoke_Binary` | No → 0; Yes → 1 |
| `Age_Activity_Interaction` | Age × Physical Activity Binary |

These variables provide a compact state-based representation of obesity-related health and lifestyle information.

## Three Feature-Engineering Branches

### Branch I — Baseline Classical Feature Engineering

The first branch uses the original obesity variables as the baseline feature space.

Workflow:

```text
Original Features
      ↓
Categorical Encoding
      ↓
Mutual-Information Feature Selection
      ↓
Top 15 Features
      ↓
Baseline Feature Space
```

The final unbalanced Branch I feature space contains:

```text
7,384 samples × 15 selected features
```

### Branch II — Automata-Merged Features

The second branch combines the encoded original feature space with the six automata-derived features.

```text
Original Encoded Features ──────┐
                                ├── Merge → Automata-Merged Feature Space
Six Automata Features ──────────┘
```

The implementation produces:

- Original encoded features: **16**
- Automata-derived features: **6**
- Final Branch II features: **22**

Final feature-space size:

```text
7,384 samples × 22 features
```

### Branch III — Automata-Integrated Feature Engineering

The third branch represents the main automata-based feature-engineering stage.

```text
Six Automata-Derived Features
            ↓
     Weighted Automata
            ↓
       Fuzzy Automata
            ↓
       Hybrid Automata
            ↓
Automata-Integrated Feature Space
```

The final Branch III feature space contains:

1. `Age_State`
2. `Cholesterol_State`
3. `BP_State`
4. `PhysActive_Binary`
5. `Smoke_Binary`
6. `Age_Activity_Interaction`
7. `Weighted_Automata_Score`
8. `Fuzzy_Automata_Score`
9. `Hybrid_Automata_Score`

Final feature-space size:

```text
7,384 samples × 9 features
```

## Weighted Automata

Mutual information is used to estimate the relative relevance of the six derived features with respect to the obesity target. The derived features are normalized before weighted aggregation so that variables with larger numerical ranges do not dominate the score.

The weighted score follows:

```text
S_W = Σ (w_i × z_i)
```

where `w_i` is the normalized importance weight and `z_i` is the corresponding normalized automata-derived feature.

## Fuzzy Automata

The fuzzy component models gradual health-risk transitions using membership functions for age, systolic blood pressure, cholesterol, physical inactivity, and smoking.

Example fuzzy rules include:

- Senior age AND high blood pressure AND inactivity → high risk
- High cholesterol AND smoking → high risk
- Senior age AND high cholesterol → high risk
- High blood pressure AND smoking → high risk

The implemented fuzzy component produces non-zero fuzzy-risk values for **1,159 of the 7,384 labelled samples**, showing selective activation of higher-risk clinical and behavioural combinations.

## Hybrid Automata

The hybrid stage integrates:

- normalized weighted automata score,
- fuzzy automata score, and
- normalized age–activity interaction.

The resulting `Hybrid_Automata_Score` is added to the final Branch III feature space.

## Class Balancing

The notebook generates balanced versions of all three branches using random oversampling. The majority obesity class contains **2,623 samples**, so balancing produces:

```text
2,623 × 4 classes = 10,492 samples
```

Generated balanced outputs:

```text
branch1_baseline_balanced.csv
branch2_automata_merged_balanced.csv
branch3_automata_integrated_balanced.csv
```

## Generated Outputs

When the notebook is executed, an `outputs` directory is created containing:

```text
outputs/
├── branch1_baseline_features.csv
├── branch2_automata_merged_features.csv
├── branch3_automata_integrated_features.csv
├── branch1_baseline_balanced.csv
├── branch2_automata_merged_balanced.csv
├── branch3_automata_integrated_balanced.csv
├── branch1_feature_importance.csv
└── automata_feature_weights.csv
```

## Requirements

The notebook uses:

- Python
- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- Jupyter Notebook

Install the required packages with:

```bash
pip install numpy pandas scikit-learn imbalanced-learn jupyter
```

## Running the Notebook

Clone the repository:

```bash
git clone <repository-url>
cd obesity-automata-feature-engineering
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
feature engineering/Feature_Engineering.ipynb
```

Because the dataset is stored in a separate `datasets` folder, make sure the dataset path in the notebook points to the correct location.

If the notebook kernel is running from the `feature engineering` folder, use:

```python
DATASET_FILE = "../datasets/Obesity_DataSet_2.csv"
```

If the kernel is running from the repository root, use:

```python
DATASET_FILE = "datasets/Obesity_DataSet_2.csv"
```

Then run the notebook cells in order.

## Overall Workflow

```text
                    Original Obesity Dataset
                             │
                             ▼
                      Data Preprocessing
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
          Branch I        Branch II       Branch III
              │              │              │
       Original Features  Original +       Automata-
              │          Automata Features  Derived Features
       Feature Selection      │              │
              │              Merge      Weighted Automata
              │               │              │
              │               │         Fuzzy Automata
              │               │              │
              │               │         Hybrid Automata
              ▼               ▼              ▼
        Baseline Space   Merged Space   Integrated Space
              │               │              │
              └───────────────┼──────────────┘
                              ▼
                     Class Balancing
                              │
                              ▼
                     ML / Ensemble Models
```

## Experimental Note

The balanced CSV files are useful for inspecting the generated feature spaces and for preliminary experiments. For final cross-validation experiments, class balancing, feature selection, and data-driven weight estimation should be fitted using only the training portion of each fold to avoid information leakage.

## Research Purpose

This repository supports research on:

- obesity-risk prediction,
- healthcare feature engineering,
- machine-learning classification,
- state-based health representation,
- weighted automata,
- fuzzy automata,
- hybrid automata,
- feature-space comparison, and
- ensemble-based predictive modelling.

## Citation

If this repository is used in academic work, cite the associated research paper or repository record when available.

## License

Add the appropriate license for the dataset and source code before public distribution. Ensure that use of the original dataset complies with its source terms and conditions.
