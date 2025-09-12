# Loan Status Classification (Interview Exercise)

Predict loan_status from tabular application data and establish a clean, reproducible baseline for risk triage and iteration.

## 📦 What's inside
- Notebook: Loans_modified_interview.ipynb
- Task: Binary/multi-class classification of loan_status
- Baseline model: DecisionTreeClassifier (max_depth=3)
- Result (baseline): ~0.71 test accuracy with confusion-matrix evaluation
- Focus: Clear preprocessing, simple baseline, and next-step roadmap

## 🧰 Tech stack
- Python • pandas • numpy
- Scikit-learn (train_test_split, DecisionTreeClassifier, metrics)
- Matplotlib • Seaborn (EDA & plots)

## 📂 Data
- Source file: loans_modified.csv
- Target column: loan_status
- Typical prep steps: drop non-predictive IDs, handle missing values, remove duplicates, one-hot encode categoricals with pd.get_dummies.
Note: Replace/adjust the file name & path if your dataset differs.

```
# 1) Clone your repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4) Launch Jupyter and run the notebook
jupyter notebook Loans_modified_interview.ipynb
```

## 📝 What the notebook does
- Load & Inspect – read CSV, basic schema checks, class balance overview.
- Clean & Prepare – drop ID-like columns, handle nulls/dupes, get_dummies for categoricals.
- Train/Test Split – hold-out validation with random_state=42.
- Model (Baseline) – Decision Tree (max_depth=3) for a transparent first pass.
- Evaluate – accuracy score + confusion matrix to spot error patterns.
- Takeaways – where features/quality limit performance; next experiments to try.

## 📈 Baseline result
- DecisionTree (depth=3): ~0.71 test accuracy
- Confusion matrix included in the notebook for class-wise performance

## 🧪 Next steps (roadmap)
- Models: Logistic Regression, Random Forest, Gradient Boosting / XGBoost, calibrated probabilities
- Validation: Stratified K-fold, ROC-AUC & PR-AUC, precision/recall at operating thresholds
- Features: Targeted feature engineering, scaling where needed, interaction terms
- Imbalance (if present): Class weights, imblearn (SMOTE), threshold tuning
- Explainability: Feature importance, permutation importance, SHAP
- Packaging: Pipeline + ColumnTransformer; save model (joblib/pickle) for reuse

## 🧑‍💻 Reproducibility tips
- Pin versions in requirements.txt
- Set random_state=42 for splits/models
- Keep raw vs. processed data distinct (if you add a /data folder)
