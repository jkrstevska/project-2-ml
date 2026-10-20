"""Your reusable functions live here.

The rule from the brief: Python logic goes in `.py` files, and the notebooks
hold the narrative. The moment a cell grows past a few lines, or you find
yourself pasting it a second time, move it here and call it from the notebook.

To use this module from a notebook in `notebooks/`:

    import sys
    sys.path.append("..")
    from src.functions import *

The four below are only there to show the shape. Rename them, change the
arguments, write your own. This is a starting point, not an interface you have
to implement.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

RANDOM_STATE = 0

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
SCORING = {
    "f2": make_scorer(fbeta_score, beta=2),
    "average_precision": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
    "accuracy": "accuracy",
}


def load_data():
    """
    Read the raw Telco file and apply the fixes that do not learn from the data.
    """

    df = pd.read_csv(ROOT/RAW/'telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    return df

def split_data(df, test_size=0.2, random_state=RANDOM_STATE):
    """
    Drop the ID, encode the target, and make the stratified train/test split.
    """    

    X = df.drop(columns=['customerID','Churn'])
    y = (df['Churn']== 'Yes').astype(int)

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

def build_preprocessor(numeric, categorical):
    """Return a ColumnTransformer that prepares each type of column."""

    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(drop='if_binary', handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric),
        ('cat', cat_pipe, categorical)
    ])

    return preprocessor

def add_features(X, features=('n_addons', 'new_customer', 'con_serv')):
    X = X.copy()
    addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    if 'n_addons' in features:
        X['n_addons'] = (X[addons] == 'Yes').sum(axis=1)
    if 'new_customer' in features:
        X['new_customer'] = (X['tenure'] <= 12).astype(int)
    if 'con_serv' in features:
        X['con_serv'] = ((X['Contract'] == 'Month-to-month') & (X['InternetService'] == 'Fiber optic')).astype(int)

    return X

def compare_models(models, X, y, cv, scoring):
    """Cross-validate each model on the same folds and return a tidy table."""
    rows = []
    for name, model in models.items():
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        rows.append({'model': name,
                     'F1': results['test_f1'].mean(),
                     'F1 std': results['test_f1'].std(),
                     'recall': results['test_recall'].mean(),
                     'precision': results['test_precision'].mean(),
                     'accuracy': results['test_accuracy'].mean()
                     })
    comparison = pd.DataFrame(rows).set_index('model').sort_values('F1', ascending=False)

    return comparison.round(3)



def plot_errors(y_true, y_pred):
    """Show where the model goes wrong: a confusion matrix or the residuals."""
    pass
