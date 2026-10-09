# Mental Health ML Pipeline

Machine learning pipeline to predict depression (Yes/No) from demographic and screening-score variables.

## Models
Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, compared using repeated stratified cross-validation and a held-out test set (ROC AUC, PR AUC, sensitivity, specificity, F1). Feature importance is assessed with SHAP.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib shap jupyterlab
```

## Usage
1. Place the dataset in `data/` (not tracked by Git).
2. Run the notebook or script to preprocess, train, and evaluate the models.

## Data
Data can be accessed as described on info
