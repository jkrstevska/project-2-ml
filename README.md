![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 2 ML | Customer Churn: From Data to Predictions

*Data Science & Machine Learning — Week 6*

**Project 2 — Machine Learning: From Data to Predictions**
Data Science & Machine Learning, Week 6

A supervised machine learning pipeline that takes a raw telecom customer dataset, prepares it inside a scikit-learn pipeline, compares six model families with cross-validation against a baseline, evaluates the final model once on held-out data, and saves it so it can predict for new customers.

**Slides:** [link to the presentation]

---

## 1. Project Overview

**Business objective:** A telecom company loses about 1 in 4 customers. Its retention team can contact customers with an offer, but it can’t contact everyone. The goal is to predict **which current customers are most likely to leave**, so the team knows who to contact before they go.

The project follows the 11-step ML workflow:
**Frame** (target, metric) → **Look** (EDA, data quality) → **Split once** → **Baseline** → **Pipeline** → **Compare** → **Tune** → **Threshold** → **Test once** → **Interpret** → **Save**.



## 2. The Prediction Task

| | |
|---|---|
| **Question** | Which current customers are likely to churn? |
| **Target** | `Churn`, positive class = `Yes` (1) |
| **Who uses it** | The retention team |
| **Decision it informs** | Who gets a call or a retention offer |
| **Metric** | **F2** |
| **Baseline** | Always “Yes” (contact everyone): F2 = 0.644 |

**Why F2:** missing a customer who is about to leave (a lost customer) costs more than sending an offer to someone who would have stayed. F2 weights recall more heavily than precision, so it rewards catching leavers. Unlike recall alone, it can’t be gamed by flagging everyone.

**Why not accuracy:** the target is imbalanced (73.5% stay, 26.5% leave). A model that predicts “nobody leaves” is 73.5% accurate and catches zero churners.

Supporting metrics: average precision (independent of the threshold), recall, precision, F1, accuracy and the share of customers flagged.



## 3. The Dataset

- **Source:** Telco Customer Churn, an IBM sample dataset, fetched with `python download_data.py telco`
- **Licence:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **File:** `data/raw/telco_churn.csv`
- **Size:** 7,043 rows × 21 columns raw. 19 features after dropping `customerID` (an identifier) and separating the target `Churn`.
- **Grain:** one row = one customer
- **Features:** demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account information (`tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`) and services (`PhoneService`, `MultipleLines`, `InternetService` and six internet add-ons)
- **Data quality:** verified in notebook 01:
  - `TotalCharges` was stored as text, with 11 blanks. All 11 are customers with `tenure = 0` (not billed yet), so they were converted to numeric and filled with 0.
  - “No internet service” and “No phone service” were kept as their own category, so “No” means “has the service but not this add-on”.
  - No columns that would be unknown at prediction time were found. Every remaining feature is known for a current customer.
- **Split:** stratified 80/20 train/test split with a fixed `random_state`: 5,634 training rows and 1,409 test rows, 26.5% churn in both. The test set is used only once, in notebook 03.



## 4. Pipeline Design

All preprocessing is done **inside a scikit-learn `Pipeline`**, so cross-validation refits it on every fold and nothing leaks from validation or test data into training.

- **`ColumnTransformer`** with one branch per column type:
  - **Numeric** (`tenure`, `MonthlyCharges`, `TotalCharges`): median imputation → `StandardScaler`
  - **Categorical** (16 columns): most-frequent imputation → `OneHotEncoder(drop="if_binary", handle_unknown="ignore")`
- **Model:** `LogisticRegression(C=0.1, class_weight="balanced")`
- **Threshold:** the whole pipeline is wrapped in `TunedThresholdClassifierCV`, which chooses the decision threshold that maximises F2 on the training folds (**0.32** instead of the default 0.50)

**Class imbalance** is handled in three ways: a stratified split and stratified 5-fold cross-validation, `class_weight="balanced"`, and the tuned threshold.



## 5. Repository Structure

```
project-2-ml/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv                        # downloaded dataset (gitignored)
│   └── README.md                                  # dataset descriptions and licences
│
├── models/
│   └── final_model.joblib                         # saved pipeline + threshold (gitignored, recreated by notebook 03)
│
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb                  # framing, data quality, EDA, split, baselines
│   ├── 02_pipelines_and_models.ipynb              # pipeline, features, model comparison, tuning, threshold
│   └── 03_evaluation_and_interpretation.ipynb     # report: test once, error analysis, drivers, save and reload
│
├── src/
│   └── functions.py                               # reusable functions (loading, splitting, preprocessing, comparison)
│
├── download_data.py                               # fetches the dataset into data/raw/
├── requirements.txt
├── RUBRIC.md
└── README.md
```

**Reusable functions in `src/functions.py`:**

| Name | Purpose |
|---|---|
| `ROOT`, `RAW`, `MODELS` | Project paths, defined once |
| `RANDOM_STATE`, `CV`, `SCORING` | The seed, the 5-fold stratified splitter and the metrics used in every notebook |
| `load_data()` | Reads the raw file and converts `TotalCharges` to numeric |
| `split_data(df)` | Drops the ID, encodes the target and makes the stratified train/test split |
| `build_preprocessor(numeric, categorical)` | Returns the `ColumnTransformer` (impute, scale, encode) |
| `add_features(X, features)` | Adds the engineered features tested in notebook 02 |
| `compare_models(models, X, y, cv, scoring)` | Cross-validates several models on the same folds and returns a table sorted by F2 |



## 6. Modelling

Every step was run on the training set only, with the same 5-fold stratified cross-validation:

| Step | Technique | Result |
|---|---|---|
| Baseline | `DummyClassifier` (always “No”, always “Yes”) | Always “Yes”: **F2 0.644**, the bar to beat |
| First model | Logistic regression in the pipeline | F2 0.720 ± 0.005, above the baseline by far more than the fold spread |
| Feature engineering | `n_addons`, `new_customer`, `con_serv`, added inside the pipeline | None improved F2 beyond the fold spread, so none were kept |
| Model comparison | 6 families: logistic regression, KNN, SVM, decision tree, random forest, HistGradientBoosting | All beat the baseline; logistic regression and HistGradientBoosting carried forward |
| Tuning | `GridSearchCV`, refit on F2 | Best LR `C=0.1` (F2 0.723); best HGB `learning_rate=0.03, max_depth=3, min_samples_leaf=50` (F2 0.725) |
| Threshold | `TunedThresholdClassifierCV` with nested cross-validation | Threshold ≈ 0.32, chosen by F2 |

**Final comparison at tuned thresholds (nested cross-validation):**

| Model | F2 | F2 std | Recall | Precision | Threshold |
|---|---|---|---|---|---|
| **Logistic regression (C=0.1)** | **0.749** | 0.005 | 0.917 | 0.433 | 0.32 |
| HistGradientBoosting (tuned) | 0.742 | 0.009 | 0.915 | 0.424 | 0.34 |
| Baseline: always “Yes” | 0.644 | – | 1.000 | 0.265 | – |

The two models are tied: the gap (0.007) is smaller than the spread across folds. **Logistic regression was chosen** because it is simpler, more stable across folds, faster, and its coefficients can be explained to the retention team. The best tree models were all shallow (`max_depth=3`), which suggests churn here is driven by a few features acting mostly on their own. That is why a linear model performs just as well.



## 7. Key Findings

**Test set (1,409 customers, used once):**

| Metric | Test | CV estimate | Baseline (always “Yes”) |
|---|---|---|---|
| **F2** | **0.756** | 0.749 ± 0.005 | 0.644 |
| Recall | 0.920 | 0.917 ± 0.009 | 1.000 |
| Precision | 0.442 | 0.433 ± 0.011 | 0.265 |
| F1 | 0.597 | 0.588 ± 0.009 | 0.420 |
| Accuracy | 0.670 | 0.659 ± 0.015 | 0.265 |

The test score matches the cross-validated estimate, so it is a trustworthy estimate for new customers.

| Question | Verdict | Key evidence |
|---|---|---|
| Is the model better than contacting everyone? | **Yes** | Test F2 0.756 vs 0.644. It caught **344 of 374** leavers (92%) and missed 30. About 4 in 10 contacted customers were real leavers, against 2.6 in 10 at random. |
| What drives the predictions? | **Tenure, then contract** | Permutation importance (drop in test F2): `tenure` 0.145, `Contract` 0.034, `TotalCharges` 0.017, all others ≤ 0.005. Without tenure, the model falls below the baseline. |
| Do payment method and tech support matter? | **Not on their own** | Strongly linked to churn in the EDA, but near zero importance in the model: their information overlaps with tenure and contract. |
| Where does the model fail? | **Loyal-looking leavers** | Recall is 98% for month-to-month, 65% for one-year and only **10% for two-year** contracts. Missed churners have a median tenure of 56 months, against 8 for those caught. |

**The story:** customers don’t leave at random. The newest customers on month-to-month contracts are by far the most likely to leave, and the model finds them reliably. To find the same 344 leavers by contacting customers at random, the team would need about 1,300 calls instead of 779. A retention team working from this model should use the risk score as a **priority list** and contact customers from the highest risk down. It should invest in the **first months** of a customer’s life (onboarding, an early check-in) and give month-to-month customers a reason to move to a **longer contract**. The model should not be relied on to spot long-standing customers who are about to leave.



## 8. Limitations and Next Steps

**Limitations:**
- **It flags 55% of customers.** If the team can’t contact that many, rank by probability and start at the top.
- **The probabilities are not calibrated.** `class_weight="balanced"` pushes them up, so a score of 0.6 does not mean a 60% chance of leaving. Use the scores to rank customers.
- **It shows who is at risk, not why.** The features are associations, not causes. The model can’t say which offer will make a customer stay.
- **One snapshot, no dates.** Customer behaviour and prices change over time, so the model should be checked on newer data and retrained regularly.

**With more time:**
- **Cost-based threshold:** with the real cost of an offer and the value of a kept customer, choose the threshold that saves the most money instead of maximising F2.
- **Missed churners:** look more closely at the churners on long contracts and look for features that could reveal them, such as complaints or recent usage.
- **Newer data:** validate the model on later customers to check it still holds.



## 9. How to Run

The model was built with **scikit-learn 1.9.0**. `TunedThresholdClassifierCV` needs scikit-learn ≥ 1.5, and a saved model may not load under a different version.

```bash
# 1. Clone the repository
git clone https://github.com/[your-username]/[your-repo].git
cd [your-repo]

# 2. Check the libraries are installed (a current Anaconda install has them)
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, joblib; print('all present')"
# Only if that fails:
pip install -r requirements.txt

# 3. Download the data into data/raw/
python download_data.py telco

# 4. Open the notebooks and run them in order: 01 → 02 → 03
jupyter lab notebooks/
```

Notebook 03 rebuilds the final model, saves it to `models/final_model.joblib` and reloads it to predict. To predict for a new customer:

```python
import joblib
import pandas as pd

model = joblib.load("models/final_model.joblib")["model"]   # pipeline + tuned threshold (0.32)

new_customer = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0, "TotalCharges": 150.0,
}])

print(model.predict_proba(new_customer)[:, 1])   # churn risk score
print(model.predict(new_customer))               # 1 = flag for the retention team
```

The model takes the raw column values; all encoding and scaling happen inside the pipeline.



## 10. Author

Jovana Krstevska
