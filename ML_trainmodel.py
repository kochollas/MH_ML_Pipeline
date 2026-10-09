
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, accuracy_score, confusion_matrix, roc_curve,
)


# ----------------------------------------------------------------------
# 3. The 5 models
# ----------------------------------------------------------------------
RANDOM_STATE = 42

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(
        kernel="rbf", C=1.0, probability=True, class_weight="balanced",
        random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=4, class_weight="balanced",
        random_state=RANDOM_STATE),
}



# Wrap each model so preprocessing is fitted inside CV folds (no leakage)
models = {name: Pipeline([("prep", preprocessor), ("clf", clf)])
          for name, clf in models.items()}
# ----------------------------------------------------------------------
# 4. Cross-validation on the training set
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
           "f1": "f1", "recall": "recall", "precision": "precision"}

cv_rows = []
for name, pipe in pipelines.items():
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    row = {"Model": name}
    for m in scoring:
        row[f"CV {m}"] = f"{res[f'test_{m}'].mean():.3f} ± {res[f'test_{m}'].std():.3f}"
    cv_rows.append(row)

cv_table = pd.DataFrame(cv_rows).set_index("Model")
print("\n=== Cross-validation (train set, 5-fold x 3 repeats) ===")
print(cv_table.to_string())

# ----------------------------------------------------------------------
# 5. Fit on full training set, evaluate on held-out test set
# ----------------------------------------------------------------------
test_rows, roc_data, fitted = [], {}, {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    test_rows.append({
        "Model": name,
        "ROC AUC": roc_auc_score(y_test, prob),
        "PR AUC": average_precision_score(y_test, prob),
        "Accuracy": accuracy_score(y_test, pred),
        "Sensitivity": recall_score(y_test, pred),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Precision": precision_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred),
    })
    roc_data[name] = roc_curve(y_test, prob)

test_table = pd.DataFrame(test_rows).set_index("Model").round(3)
test_table = test_table.sort_values("ROC AUC", ascending=False)
print("\n=== Held-out test set (threshold = 0.5) ===")
print(test_table.to_string())

# ----------------------------------------------------------------------
# 6. Plot ROC curves, save results and best model
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, _) in roc_data.items():
    auc = test_table.loc[name, "ROC AUC"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves: depression prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)

cv_table.to_csv("cv_results.csv")
test_table.to_csv("test_results.csv")

best = test_table.index[0]
joblib.dump(fitted[best], "best_model.joblib")
print(f"\nBest model by test ROC AUC: {best} (saved to best_model.joblib)")
print("Saved: roc_curves.png, cv_results.csv, test_results.csv")