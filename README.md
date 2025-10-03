# Applied Machine Learning Project - Obesity Level Prediction (Multiclass)
End-to-end machine learning project to predict **obesity level** from lifestyle and physiological attributes.  
Built during my MSc (CS) Applied Machine Learning module to demonstrate **data preprocessing, feature selection, model development, hyperparameter tuning, and evaluation** on a real-world dataset.

---

## Goal
Build a reliable **multiclass classifier** for `NObeyesdad` (7 classes, from *Insufficient_Weight* to *Obesity_Type_III*) and compare classic ML models under a consistent pipeline.

---

## Dataset
- **Name:** *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* (UCI, 2020)  
- **Records / Features:** 2,111 rows, 17 attributes (mixed numeric + categorical)  
- **Geography:** Mexico, Peru, Colombia  
- **Target:** `NObeyesdad` (7 classes)  
- **Why it matters:** Supports **public health** personalization, population-level interventions, and **industry** use cases (wellness, product targeting).

> Class distribution is reasonably **balanced**; **no missing values** observed in this release.

---

## Pipeline (What I built)
**1) Data Quality & Scaling**
- Verified **no missingness** and **reasonable class balance**.
- Standardized numeric features (mean 0, std 1) to stabilize distance-based models and speed up gradient methods.
- Applied skewness-aware transformations for variables with heavy skew (e.g., `Age`, `NCP`).

**2) Encoding**
- Applied **label encoding** for categorical attributes (large cardinality/ordinal patterns in this dataset).

**3) Feature Selection (Filter + Wrapper + Model-based)**
- **Mutual Information (filter)** to capture non-linear relevance (handles mixed types).
- **RFE w/ Logistic Regression (wrapper)** to iteratively prune features.
- **Random Forest Feature Importance** as an additional signal.
- Final feature set = (MI ∪ RFE) ∩ {RF importance ≥ threshold}.

**4) Models & Optimization**
- Compared **Logistic Regression**, **Random Forest**, **SVM**, **Gradient Boosting (GBM)** — all **supervised**.
- **RandomizedSearchCV** + **10-fold CV** per model for efficient hyperparameter tuning on a broad search space.

---

## Models & Best Configurations

### Logistic Regression (Multinomial)
- **Best (example):** `C=3.0561`, `penalty='l1'`, `solver='saga'`
- Strengths: simple, fast, **interpretable**; surprisingly strong F1 across classes.

### Random Forest
- **Best (example):** `n_estimators=161`, `max_depth=11`, `min_samples_split=9`, `min_samples_leaf=4`, `bootstrap=True`, `max_features=None`
- Strengths: handles non-linearities & interactions; robust; built-in importance.

### Support Vector Machine (SVC)
- **Best (example):** `C=9.1683`, `kernel='linear'`  *(kernel search explored poly/RBF; linear emerged as strongest/cleanest for this data)*

### Gradient Boosting (GBM)
- **Best (example):** `n_estimators=153`, `learning_rate=0.264`, `max_depth=3`, `min_samples_split=9`, `min_samples_leaf=17`, `subsample≈0.80`

> Full parameter tables and search notes are documented in the report/appendix.

---

## Results (Macro / Weighted F1)
- **SVM** → **0.96 / 0.96** (best overall balance; very solid per-class precision/recall)
- **Logistic Regression** → **0.96 / 0.96** (close second; wins on interpretability)
- **GBM** → **0.95 / 0.95**
- **Random Forest** → **0.92 / 0.92** (slightly weaker on *Normal_Weight* class)

> Confusion matrices show consistently strong separation for *Obesity_Type_I/II/III*; occasional overlap appears between *Normal_Weight* and *Overweight_Level_I/II*.

---

## 🛠 Tech Stack
- **Python 3.10**, Jupyter Notebook  
- **pandas, numpy, scikit-learn** (LogReg, RF, SVC, GBM), **scipy**  
- **matplotlib / seaborn** for EDA & diagnostics

---

## 🗂 Repo Structure
```
├── data/ # (Optional) dataset placeholder
├── AML Coursework.ipynb # Full EDA → FS → tuning → eval
└── README.md
```

---



