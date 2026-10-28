import sys
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt  # backend auto-selected: window if available, else file-only
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, accuracy_score, confusion_matrix, roc_curve,
)

from ML_preprocess import RANDOM_STATE, build_preprocessor, prepare_data, split_data

# ----------------------------------------------------------------------
# Data: pass a CSV/XLSX path:  python ML_trainmodel.py data.csv
# ----------------------------------------------------------------------
if len(sys.argv) < 2:
    sys.exit("Usage: python ML_trainmodel.py <path to .csv or .xlsx>")
path = sys.argv[1]
dataset = pd.read_excel(path) if path.lower().endswith((".xls", ".xlsx")) else pd.read_csv(path)
print("Loaded", path, "->", dataset.shape, "| columns:", list(dataset.columns))
X, y = prepare_data(dataset)
X_train, X_test, y_train, y_test = split_data(X, y)

# ----------------------------------------------------------------------
# Models (n_jobs=1 inside models: parallelism happens at the model level)
# ----------------------------------------------------------------------
base_models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        n_jobs=1, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(
        kernel="rbf", C=1.0, probability=True, class_weight="balanced",
        random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=4, class_weight="balanced",
        random_state=RANDOM_STATE),
}

# Preprocessing lives inside the pipeline so it is re-fitted in every CV fold
pipelines = {n: Pipeline([("prep", build_preprocessor()), ("clf", clf)])
             for n, clf in base_models.items()}

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
           "f1": "f1", "recall": "recall", "precision": "precision"}


def test_metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "ROC AUC": roc_auc_score(y_true, prob),
        "PR AUC": average_precision_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Sensitivity": recall_score(y_true, pred),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Precision": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred),
    }


def run_model(name, pipe):
    """CV on train, then fit on full train and score on test: one job per model."""
    res = cross_validate(clone(pipe), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_row = {f"CV {m}": f"{res[f'test_{m}'].mean():.3f} ± {res[f'test_{m}'].std():.3f}"
              for m in scoring}
    cv_auc = res["test_roc_auc"].mean()

    fitted = clone(pipe).fit(X_train, y_train)
    prob = fitted.predict_proba(X_test)[:, 1]
    return name, cv_row, cv_auc, test_metrics(y_test, prob), roc_curve(y_test, prob), fitted


# All models run in parallel (replaces the two sequential loops)
results = Parallel(n_jobs=-1)(delayed(run_model)(n, p) for n, p in pipelines.items())
names, cv_rows, cv_aucs, test_rows, rocs, fitted = map(list, zip(*results))

cv_table = pd.DataFrame(cv_rows, index=names)
cv_table.index.name = "Model"
test_table = pd.DataFrame(test_rows, index=names).round(3)
test_table.index.name = "Model"
test_table = test_table.sort_values("ROC AUC", ascending=False)

print("\n=== Cross-validation (train set, 5-fold x 3 repeats) ===")
print(cv_table.to_string())
print("\n=== Held-out test set (threshold = 0.5) ===")
print(test_table.to_string())

# ----------------------------------------------------------------------
# ROC plot and saving
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, _) in zip(names, rocs):
    ax.plot(fpr, tpr, label=f"{name} (AUC={test_table.loc[name, 'ROC AUC']:.2f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves: depression prediction")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("roc_curves.png", dpi=150)
print("ROC plot saved to:", os.path.abspath("roc_curves.png"))
plt.show()  # displays the plot when run in an IDE/notebook/desktop session

cv_table.to_csv("cv_results.csv")
test_table.to_csv("test_results.csv")

# Choose the model by CROSS-VALIDATION AUC, not the test set (avoids optimistic bias)
best_i = int(np.argmax(cv_aucs))
joblib.dump(fitted[best_i], "best_model.joblib")
print(f"\nBest model by CV ROC AUC: {names[best_i]} (saved to best_model.joblib)")
print("Saved: roc_curves.png, cv_results.csv, test_results.csv")