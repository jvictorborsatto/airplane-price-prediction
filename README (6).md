# ✈️ Airplane Price Prediction

End-to-end machine learning project that predicts the **list price of aircraft** from their technical specifications (engine, speed, weight, range, dimensions, etc.). The project covers data cleaning, unit conversion, a comparison of missing-value imputation strategies, model selection, feature selection, hyperparameter tuning with a **Design of Experiments (DOE)** approach, cross-validation and a final hold-out evaluation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Headline result:** tuned XGBoost reaches **R² ≈ 0.93 in 5-fold cross-validation** (MAE ≈ US$ 201k on a mean price of ≈ US$ 2.4M).

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Key Results](#key-results)
5. [Visualizations](#visualizations)
6. [Limitations and Honest Notes](#limitations-and-honest-notes)
7. [Repository Structure](#repository-structure)
8. [How to Run](#how-to-run)
9. [Tech Stack](#tech-stack)
10. [Possible Next Steps](#possible-next-steps)
11. [Author](#author)

---

## Project Goals

- Build a regression model that estimates aircraft price from technical specifications.
- Compare different strategies for handling missing values instead of simply dropping rows.
- Compare several regression algorithms and select the best one with a fair, leakage-aware validation scheme.
- Tune hyperparameters systematically using a fractional-factorial **DOE** instead of an arbitrary grid or random search.
- Evaluate the final model with cross-validation and on data it never saw during development.

---

## Dataset

- **Source:** [Plane Price Prediction – Kaggle (rafsunahmad)](https://www.kaggle.com/datasets/rafsunahmad/plane-price-prediction)
- **Size:** 517 aircraft models, 16 columns.
- **Target:** `Price` (US$).
- **Engine types:** Piston, Propjet and Jet.

| Feature | Description |
|---|---|
| `Engine Type` | Piston / Propjet / Jet (categorical) |
| `HP or lbs thr ea engine` | Horsepower (piston) or pounds of thrust (jet) per engine |
| `Max speed Knots`, `Rcmnd cruise Knots` | Maximum and recommended cruise speed |
| `Stall Knots dirty` | Stall speed (dirty configuration) |
| `All eng rate of climb`, `Eng out rate of climb` | Rate of climb with all engines / one engine out |
| `Takeoff over 50ft`, `Landing over 50ft` | Takeoff and landing distance over a 50 ft obstacle |
| `Empty weight lbs` | Empty weight |
| `Range N.M.` | Range in nautical miles |
| `Length ft/in`, `Wing span ft/in` | Dimensions (converted to meters) |
| `Fuel gal/lbs` | Fuel capacity (dropped: mixes two units) |

> Please check the dataset license on Kaggle before redistributing the CSV.

---

## Project Workflow

### 1. Data cleaning and feature engineering
- Converted `Length ft/in` and `Wing span ft/in` (format `feet/inches`) into **meters** (`Length_m`, `Wing_Span_m`).
- Converted numeric columns stored as text (thousand separators, etc.) to `float`.
- Dropped `Model Name` (identifier) and `Fuel gal/lbs` (inconsistent units: gallons for piston, pounds for jets).
- One-hot encoded `Engine Type`.
- Removed rows with missing `Price` (the target is never imputed).
- Plain `dropna()` would have discarded about **20 % of the data** (517 → 421 rows), which motivated the imputation study below.

### 2. Missing-value imputation study
The correlation matrix showed strong relationships between the numeric variables, so missing values can be predicted from the other columns. Three column-wise strategies were compared on each feature (MAE, RMSE, R²):

| Column | XGBoost R² | Linear Reg. R² | Mode R² |
|---|---|---|---|
| HP / thrust per engine | **0.970** | 0.892 | −0.117 |
| Max speed (knots) | **0.991** | 0.978 | −0.316 |
| Stall speed (knots) | **0.887** | 0.863 | −0.356 |
| Eng-out rate of climb | **0.925** | 0.857 | −0.489 |
| Takeoff over 50 ft | **0.845** | 0.771 | −3.433 |
| Range (N.M.) | **0.866** | 0.792 | −0.392 |
| Length (m) | **0.939** | 0.881 | −0.444 |

A **hybrid** method was built, choosing the best technique per column (XGBoost for most columns, Linear Regression for landing distance, Mode for recommended cruise speed).

Impact on the downstream price model (same XGBoost, same split):

| Imputation method | MAE (US$) | RMSE (US$) | R² |
|---|---|---|---|
| Hybrid (best per column) | 202,108 | **261,455** | **0.8764** |
| XGBoost | **201,601** | 263,247 | 0.8747 |
| Linear Regression | 206,240 | 265,805 | 0.8723 |
| Simple (median / mode) | 210,734 | 266,908 | 0.8712 |

Model-based imputation clearly wins at the **column level** (Mode even produces negative R²), but the gain at the **final price prediction** is small (≈ 0.005 R²) and within what a single split can resolve.

### 3. Model selection
Baseline comparison on the same train/test split (before tuning):

| Model | MAE (US$) | RMSE (US$) | R² |
|---|---|---|---|
| Random Forest | 205,356 | 268,030 | 0.8701 |
| Gradient Boosting | 196,692 | 253,930 | 0.8834 |
| Linear Regression | 195,424 | 260,135 | 0.8777 |
| **XGBoost** | **185,502** | **243,419** | **0.8929** |
| LightGBM | 201,057 | 259,239 | 0.8785 |
| CatBoost | 203,500 | 256,983 | 0.8806 |

XGBoost was the best baseline and was selected for tuning.

### 4. Feature selection
Two approaches were applied using **only the development data** (the test set stayed untouched):
- Forward selection with 5-fold CV (MAE as criterion).
- **RFECV** (Recursive Feature Elimination with CV), which kept **10 of 14 features**:
  `HP/thrust per engine`, `Max speed`, `Recommended cruise speed`, `All-engine rate of climb`, `Engine-out rate of climb`, `Empty weight`, `Range`, `Length`, `Wing span`, `Engine Type (Piston)`.

### 5. Hyperparameter tuning with Design of Experiments (DOE)
Instead of a full grid, a **fractional-factorial design** was used:

- 9 hyperparameters × 7 levels each → a full factorial would need 7⁹ = **40,353,607 runs**.
- The design uses **49 runs**, built from two base factors varied across all 7 × 7 combinations, with the remaining factors generated as linear combinations modulo 7.
- Each run was trained on the training set and scored on a separate validation set.
- Main effects of each hyperparameter were computed from the 49 results.

**Best configuration found:**

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 650 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.6 |
| `min_child_weight` | 4 |
| `gamma` | 0.03 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.5 |

The final model was retrained on train + validation data with these parameters.

---

## Key Results

| Evaluation | MAE (US$) | RMSE (US$) | R² |
|---|---|---|---|
| DOE validation set | 183,093 | 242,953 | 0.934 |
| **5-fold cross-validation (mean ± std)** | **201,355 ± 15,363** | **280,479 ± 24,546** | **0.925 ± 0.010** |
| Hold-out test (internal split, n = 65) | 178,952 | 235,860 | 0.899 |
| Hold-out test (external set, n = 66)* | 157,891 | 215,083 | 0.958 |

\* Complete cases only (rows with missing values were removed from this set).

**Reading the results:** the cross-validation score is the most reliable estimate, since it uses all folds and reports variability. The two small test sets bracket it (R² 0.90 – 0.96), which is expected with only ~65 samples. A realistic expectation for new aircraft similar to those in the data is **R² ≈ 0.90 – 0.93**.

---

## Visualizations

> Export the figures from the notebook into an `images/` folder to display them here.

| Correlation matrix | Price by engine type |
|---|---|
| ![Correlation](images/correlation_matrix.png) | ![Price by engine](images/price_by_engine_type.png) |

| Actual vs. predicted | Residuals vs. predicted |
|---|---|
| ![Actual vs Predicted](images/actual_vs_predicted.png) | ![Residuals](images/residuals.png) |

| Feature importance | K-fold performance |
|---|---|
| ![Feature importance](images/feature_importance.png) | ![K-fold](images/kfold_performance.png) |

---

## Limitations and Honest Notes

- **Small dataset** (a few hundred aircraft): test metrics vary noticeably depending on the split, which is why cross-validation is the headline number.
- **Repeated models:** the same aircraft model can appear several times with different engine configurations. Random splits can therefore place near-duplicates in both train and test sets. A group-based split (e.g. `GroupKFold` by model name) would be a stricter evaluation.
- **Test preprocessing:** the external test set was evaluated on complete cases only (rows with missing values dropped), which makes it slightly easier than real-world data. Applying the same fitted imputation to the test set would be a more faithful setup.
- **Imputation impact is modest:** large gains at the column level translate into small gains for price prediction.
- **DOE aliasing:** a 49-run design can keep at most 8 factors orthogonal at 7 levels, so with 9 factors some effects are confounded (e.g. `gamma` with `reg_alpha`, and `min_child_weight` with `reg_lambda`). The effect tables should be read as guidance, not as clean isolated effects.
- **Price scale:** prices range from inexpensive piston aircraft to multi-million-dollar jets, so the absolute error (≈ US$ 200k) is proportionally much larger for cheaper planes. Relative error by engine type would be a useful complement.
- The dataset is a historical snapshot, so predictions do not reflect current market prices.

---

## Repository Structure

```
airplane-price-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── Plane Price.csv
├── notebooks/
│   └── airplane_price_prediction.ipynb
├── images/
│   └── (exported figures)
└── models/
    ├── xgboost_best_model.json
    └── xgboost_doe7_results.csv
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/airplane-price-prediction.git
cd airplane-price-prediction

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook notebooks/airplane_price_prediction.ipynb
```

You can also open it directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/airplane-price-prediction/blob/main/notebooks/airplane_price_prediction.ipynb)

**Suggested `requirements.txt`:**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
xgboost
lightgbm
catboost
pyDOE2
requests
jupyter
```

---

## Tech Stack

**Python** · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn · Jupyter / Google Colab

---

## Possible Next Steps

- Predict `log(Price)` to handle the strong right skew of the target.
- Add relative-error metrics (MAPE / MdAPE) per engine type.
- Use `GroupKFold` by aircraft model to remove near-duplicate leakage.
- Wrap imputation, encoding and model into a single scikit-learn `Pipeline` to apply identical preprocessing to any new data.
- Add SHAP values for model interpretability.
- Deploy a small Streamlit app where users enter specifications and receive a price estimate.

---

## Author

**<Your Name>**
[LinkedIn](https://www.linkedin.com/in/<your-profile>) · [GitHub](https://github.com/<your-username>)

If you found this project useful, feel free to ⭐ the repository.
