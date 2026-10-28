"""Preprocessing: every column has its own independent pipeline.

To change how one field is handled, edit its entry in COLUMN_SPECS only.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_RAW, TARGET = "Depression Yes/No", "depression"
EDUCATION_ORDER = ["None", "Primary School", "Secondary School", "Bachelors Degree"]


# ---- small builders: one pipeline per column --------------------------
def numeric(strategy="median", scale=True):
    steps = [("impute", SimpleImputer(strategy=strategy))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def nominal(min_frequency=None):
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)),
    ])


def ordinal(order, scale=True):
    steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[order],
                                  handle_unknown="use_encoded_value",
                                  unknown_value=-1)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


# ---- the per-field configuration (edit here) --------------------------
COLUMN_SPECS = {
    "age":                numeric(),
    "gad7_total_score":   numeric(),
    "gender":             nominal(),
    "marital_status":     nominal(),
    "household_head":     nominal(),
    #"village":            nominal(min_frequency=5),   # lump rare villages
    "level_of_education": ordinal(EDUCATION_ORDER),
}


def build_preprocessor(specs=COLUMN_SPECS):
    """One ColumnTransformer entry per column, named after the column."""
    return ColumnTransformer(
        [(col, pipe, [col]) for col, pipe in specs.items()],
        verbose_feature_names_out=True,
    )


def prepare_data(dataset: pd.DataFrame, specs=COLUMN_SPECS):
    """Clean the target, return X, y (rows with unusable target dropped)."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            f"prepare_data needs a pandas DataFrame with column names, got {type(dataset).__name__}. "
            "Load the file with pd.read_csv / pd.read_excel and do not call .values or .to_numpy().")
    ds = dataset.rename(columns={TARGET_RAW: TARGET})
    ds[TARGET] = (ds[TARGET].astype(str).str.strip().str.lower()
                  .map({"yes": 1, "no": 0}))
    ds = ds.dropna(subset=[TARGET])
    missing = [c for c in specs if c not in ds.columns]
    if missing:
        raise KeyError(f"Columns missing from dataset: {missing}")
    return ds[list(specs)], ds[TARGET].astype(int)


def split_data(X, y, test_size=0.2):
    # Split BEFORE fitting any preprocessing (the pipeline fits it per fold)
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE)