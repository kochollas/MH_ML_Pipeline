import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# --- 1. Clean the target FIRST ---
dataset = dataset.rename(columns={"Depression Yes/No": "depression"})
print(dataset["depression"].value_counts(dropna=False))

dataset["depression"] = (
    dataset["depression"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
)
dataset = dataset.dropna(subset=["depression"])
y = dataset["depression"].astype(int)

# --- 2. THEN build X ---
numeric_cols = ["age", "gad7_total_score"]
nominal_cols = ["gender", "marital_status", "household_head", "village"]
ordinal_cols = ["level_of_education"]

X = dataset[numeric_cols + nominal_cols + ordinal_cols]

# --- 3. Split BEFORE fitting any preprocessing ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 4. Preprocessing ---
# Order lowest -> highest; must match your exact spellings
education_order = [["None", "Primary School", "Secondary School", "Bachelors Degree"]]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),

    ("nom", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ]), nominal_cols),

    ("ord", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=education_order,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("scaler", StandardScaler()),
    ]), ordinal_cols),
])