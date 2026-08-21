from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "orders_dataset.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TASK 3 - PREPROCESSING")
print("=" * 60)

# ---------------------------------------------------------
# 2. Separate features and target
# ---------------------------------------------------------
target = "returned"

X = df.drop(columns=[target, "order_id"])
y = df[target]

# ---------------------------------------------------------
# 3. Identify numeric and categorical columns
# ---------------------------------------------------------
categorical_features = [
    "product_category",
    "payment_method",
]

numeric_features = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]

# ---------------------------------------------------------
# 4. Numeric preprocessing
#    - median imputation
#    - standard scaling
# ---------------------------------------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# ---------------------------------------------------------
# 5. Categorical preprocessing
#    - most-frequent imputation
#    - one-hot encoding
# ---------------------------------------------------------
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# ---------------------------------------------------------
# 6. Combine preprocessing
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# ---------------------------------------------------------
# 7. Stratified 80/20 train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")

print(f"\nTraining return rate: {y_train.mean():.4f}")
print(f"Test return rate: {y_test.mean():.4f}")

# ---------------------------------------------------------
# 8. Fit ONLY on training data
# ---------------------------------------------------------
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the already-fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed successfully.")
print("Training transformed shape:", X_train_processed.shape)
print("Test transformed shape:", X_test_processed.shape)

print("\nImportant: the preprocessor was FIT only on X_train.")

# ---------------------------------------------------------
# TASK 4 - DUMMY CLASSIFIER BASELINE
# ---------------------------------------------------------
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

print("\n" + "=" * 60)
print("TASK 4 - DUMMY CLASSIFIER BASELINE")
print("=" * 60)

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train_processed, y_train)

dummy_predictions = dummy_model.predict(X_test_processed)

dummy_accuracy = accuracy_score(y_test, dummy_predictions)
dummy_f1 = f1_score(y_test, dummy_predictions, pos_label=1)

print(f"\nDummy accuracy: {dummy_accuracy:.4f}")
print(f"Dummy F1 (returned=1): {dummy_f1:.4f}")


# ---------------------------------------------------------
# TASK 5 - LOGISTIC REGRESSION
# ---------------------------------------------------------
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

print("\n" + "=" * 60)
print("TASK 5 - LOGISTIC REGRESSION")
print("=" * 60)

# Train Logistic Regression with balanced class weights
logistic_model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
)

logistic_model.fit(X_train_processed, y_train)

# ---------------------------------------------------------
# Default threshold = 0.50
# ---------------------------------------------------------
logistic_probabilities = logistic_model.predict_proba(
    X_test_processed
)[:, 1]

logistic_predictions_05 = (
    logistic_probabilities >= 0.50
).astype(int)

accuracy_05 = accuracy_score(y_test, logistic_predictions_05)
precision_05 = precision_score(
    y_test, logistic_predictions_05, pos_label=1
)
recall_05 = recall_score(
    y_test, logistic_predictions_05, pos_label=1
)
f1_05 = f1_score(
    y_test, logistic_predictions_05, pos_label=1
)
roc_auc = roc_auc_score(
    y_test, logistic_probabilities
)

print("\nLogistic Regression at threshold 0.50:")
print(f"Accuracy : {accuracy_05:.4f}")
print(f"Precision: {precision_05:.4f}")
print(f"Recall   : {recall_05:.4f}")
print(f"F1       : {f1_05:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# ---------------------------------------------------------
# Threshold sweep: 0.10 to 0.90
# Step size = 0.02
# ---------------------------------------------------------
thresholds = np.arange(0.10, 0.9001, 0.02)

threshold_results = []

for threshold in thresholds:
    predictions = (
        logistic_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

threshold_df = pd.DataFrame(threshold_results)

# Find threshold with maximum F1
best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])
best_precision = float(best_row["precision"])
best_recall = float(best_row["recall"])
best_f1 = float(best_row["f1"])

print("\nThreshold sweep results:")
print(
    threshold_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
        },
    )
)

print("\nBest Logistic Regression threshold:")
print(f"Threshold: {best_threshold:.2f}")
print(f"F1       : {best_f1:.4f}")
print(f"Recall   : {best_recall:.4f}")
print(f"Precision: {best_precision:.4f}")

print(
    "\nBusiness trade-off: lowering the decision threshold makes the "
    "model flag more orders as potentially return-prone. This tends "
    "to improve recall, reducing the number of actual returns that "
    "we miss, but it also increases false positives and therefore "
    "reduces precision. Raising the threshold does the opposite."
)


# ---------------------------------------------------------
# TASK 6 - RANDOM FOREST + GRID SEARCH
# ---------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

print("\n" + "=" * 60)
print("TASK 6 - RANDOM FOREST + GRID SEARCH")
print("=" * 60)

# Build the full pipeline:
# preprocessing + Random Forest
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# Required parameter grid
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None],
}

# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Grid search using ROC-AUC
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

# IMPORTANT:
# Fit the complete pipeline on the ORIGINAL training data.
# The pipeline will fit preprocessing only within each CV training fold.
grid_search.fit(X_train, y_train)

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)

print(
    f"Best cross-validated ROC-AUC: "
    f"{grid_search.best_score_:.4f}"
)

# Held-out test-set ROC-AUC
rf_best_pipeline = grid_search.best_estimator_

rf_test_probabilities = rf_best_pipeline.predict_proba(
    X_test
)[:, 1]

rf_test_roc_auc = roc_auc_score(
    y_test,
    rf_test_probabilities,
)

print(
    f"Held-out test ROC-AUC: "
    f"{rf_test_roc_auc:.4f}"
)

# ---------------------------------------------------------
# TASK 7 - RANDOM FOREST FEATURE IMPORTANCE
# ---------------------------------------------------------
from sklearn.inspection import permutation_importance

print("\n" + "=" * 60)
print("TASK 7 - FEATURE IMPORTANCE")
print("=" * 60)

# Get the fitted preprocessing and Random Forest objects
fitted_preprocessor = rf_best_pipeline.named_steps["preprocessor"]
rf_model = rf_best_pipeline.named_steps["classifier"]

# Get transformed feature names
feature_names = fitted_preprocessor.get_feature_names_out()

# ---------------------------------------------------------
# 1. Impurity-based feature importance
# ---------------------------------------------------------
impurity_importances = rf_model.feature_importances_

impurity_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": impurity_importances,
    }
).sort_values(
    "importance",
    ascending=False,
)

top5_impurity = impurity_df.head(5).copy()

print("\nTop 5 impurity-based features:")
print(
    top5_impurity.to_string(
        index=False,
        formatters={"importance": "{:.6f}".format},
    )
)

# ---------------------------------------------------------
# 2. Permutation importance on held-out test data
# ---------------------------------------------------------
permutation = permutation_importance(
    rf_best_pipeline,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

permutation_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }
).sort_values(
    "importance_mean",
    ascending=False,
)

top5_permutation = permutation_df.head(5).copy()

print("\nTop 5 permutation-importance features:")
print(
    top5_permutation.to_string(
        index=False,
        formatters={
            "importance_mean": "{:.6f}".format,
            "importance_std": "{:.6f}".format,
        },
    )
)

# ---------------------------------------------------------
# 3. Aggregate impurity importance to original features
# ---------------------------------------------------------
original_importance = {}

for feature, importance in zip(
    feature_names,
    impurity_importances
):
    if feature.startswith("num__"):
        original_feature = feature.replace("num__", "")
    elif feature.startswith("cat__"):
        encoded_name = feature.replace("cat__", "")

        if encoded_name.startswith("product_category_"):
            original_feature = "product_category"
        elif encoded_name.startswith("payment_method_"):
            original_feature = "payment_method"
        else:
            original_feature = encoded_name
    else:
        original_feature = feature

    original_importance[original_feature] = (
        original_importance.get(original_feature, 0.0)
        + importance
    )

aggregated_impurity_df = pd.DataFrame(
    {
        "feature": list(original_importance.keys()),
        "impurity_importance": list(original_importance.values()),
    }
).sort_values(
    "impurity_importance",
    ascending=False,
)

top5_original_impurity = aggregated_impurity_df.head(5)

print("\nTop 5 original features by aggregated impurity importance:")
print(
    top5_original_impurity.to_string(
        index=False,
        formatters={
            "impurity_importance": "{:.6f}".format,
        },
    )
)

# ---------------------------------------------------------
# 4. Compare the SAME original features using permutation
# ---------------------------------------------------------
comparison_df = top5_original_impurity.merge(
    permutation_df[
        ["feature", "importance_mean", "importance_std"]
    ],
    on="feature",
    how="left",
)

comparison_df = comparison_df.rename(
    columns={
        "importance_mean": "permutation_importance",
        "importance_std": "permutation_std",
    }
)

comparison_df["importance_loss"] = (
    comparison_df["impurity_importance"]
    - comparison_df["permutation_importance"]
)

print("\nSide-by-side comparison of the same top-5 original features:")
print(
    comparison_df[
        [
            "feature",
            "impurity_importance",
            "permutation_importance",
            "importance_loss",
        ]
    ].to_string(
        index=False,
        formatters={
            "impurity_importance": "{:.6f}".format,
            "permutation_importance": "{:.6f}".format,
            "importance_loss": "{:.6f}".format,
        },
    )
)

# ---------------------------------------------------------
# 5. Features losing the most importance
# ---------------------------------------------------------
loss_df = comparison_df.sort_values(
    "importance_loss",
    ascending=False,
)

print("\nOriginal top-5 features losing the most importance:")
print(
    loss_df[
        [
            "feature",
            "impurity_importance",
            "permutation_importance",
            "importance_loss",
        ]
    ].to_string(
        index=False,
        formatters={
            "impurity_importance": "{:.6f}".format,
            "permutation_importance": "{:.6f}".format,
            "importance_loss": "{:.6f}".format,
        },
    )
)

print(
    "\nInterpretation: impurity-based importance can overrate a "
    "continuous feature because a tree can test many possible "
    "thresholds on that feature, creating more opportunities for "
    "random impurity reductions even when the feature has limited "
    "real predictive signal."
)

# ---------------------------------------------------------
# TASK 8 - SUBGROUP / ROOT-CAUSE ANALYSIS
# ---------------------------------------------------------
from sklearn.metrics import precision_score, recall_score

print("\n" + "=" * 60)
print("TASK 8 - SUBGROUP / ROOT-CAUSE ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# 1. Get the winning Random Forest predictions
# ---------------------------------------------------------
# Use the default 0.50 threshold here for subgroup evaluation.
rf_predictions = (
    rf_test_probabilities >= 0.50
).astype(int)

# Create a copy of the original test data
test_results = X_test.copy()

test_results["actual_returned"] = y_test.values
test_results["predicted_returned"] = rf_predictions

# ---------------------------------------------------------
# 2. Recall and precision by product category
# ---------------------------------------------------------
category_results = []

for category in sorted(
    test_results["product_category"].unique()
):
    subgroup = test_results[
        test_results["product_category"] == category
    ]

    recall = recall_score(
        subgroup["actual_returned"],
        subgroup["predicted_returned"],
        pos_label=1,
        zero_division=0,
    )

    precision = precision_score(
        subgroup["actual_returned"],
        subgroup["predicted_returned"],
        pos_label=1,
        zero_division=0,
    )

    category_results.append(
        {
            "product_category": category,
            "recall": recall,
            "precision": precision,
            "n_orders": len(subgroup),
        }
    )

category_df = pd.DataFrame(category_results)

print("\nRecall and precision by product_category:")
print(
    category_df.to_string(
        index=False,
        formatters={
            "recall": "{:.4f}".format,
            "precision": "{:.4f}".format,
        },
    )
)

# ---------------------------------------------------------
# 3. Recall and precision by payment method
# ---------------------------------------------------------
payment_results = []

for payment in sorted(
    test_results["payment_method"].unique()
):
    subgroup = test_results[
        test_results["payment_method"] == payment
    ]

    recall = recall_score(
        subgroup["actual_returned"],
        subgroup["predicted_returned"],
        pos_label=1,
        zero_division=0,
    )

    precision = precision_score(
        subgroup["actual_returned"],
        subgroup["predicted_returned"],
        pos_label=1,
        zero_division=0,
    )

    payment_results.append(
        {
            "payment_method": payment,
            "recall": recall,
            "precision": precision,
            "n_orders": len(subgroup),
        }
    )

payment_df = pd.DataFrame(payment_results)

print("\nRecall and precision by payment_method:")
print(
    payment_df.to_string(
        index=False,
        formatters={
            "recall": "{:.4f}".format,
            "precision": "{:.4f}".format,
        },
    )
)

# ---------------------------------------------------------
# 4. Overall Random Forest recall and precision
# ---------------------------------------------------------
overall_recall = recall_score(
    y_test,
    rf_predictions,
    pos_label=1,
    zero_division=0,
)

overall_precision = precision_score(
    y_test,
    rf_predictions,
    pos_label=1,
    zero_division=0,
)

print("\nOverall Random Forest test performance:")
print(f"Recall   : {overall_recall:.4f}")
print(f"Precision: {overall_precision:.4f}")

# ---------------------------------------------------------
# 5. Identify the weakest subgroup
# ---------------------------------------------------------
category_df["recall_gap"] = (
    overall_recall - category_df["recall"]
)

category_df["precision_gap"] = (
    overall_precision - category_df["precision"]
)

payment_df["recall_gap"] = (
    overall_recall - payment_df["recall"]
)

payment_df["precision_gap"] = (
    overall_precision - payment_df["precision"]
)

# Find subgroup with the largest recall gap
worst_category_recall = category_df.loc[
    category_df["recall_gap"].idxmax()
]

worst_payment_recall = payment_df.loc[
    payment_df["recall_gap"].idxmax()
]

print("\nWorst product-category subgroup by recall:")
print(
    f"{worst_category_recall['product_category']}: "
    f"recall={worst_category_recall['recall']:.4f}"
)

print("\nWorst payment-method subgroup by recall:")
print(
    f"{worst_payment_recall['payment_method']}: "
    f"recall={worst_payment_recall['recall']:.4f}"
)

# ---------------------------------------------------------
# 6. Root-cause / next-step recommendation
# ---------------------------------------------------------
print("\nRecommended next step:")

if (
    worst_category_recall["recall_gap"]
    >= worst_payment_recall["recall_gap"]
):
    print(
        f"The {worst_category_recall['product_category']} category "
        f"has the largest recall gap versus the overall model. "
        f"A concrete next step is to introduce a category-specific "
        f"decision threshold for {worst_category_recall['product_category']}, "
        f"chosen on validation data to improve recall while monitoring "
        f"precision."
    )
else:
    print(
        f"The {worst_payment_recall['payment_method']} payment group "
        f"has the largest recall gap versus the overall model. "
        f"A concrete next step is to introduce a payment-method-specific "
        f"decision threshold for {worst_payment_recall['payment_method']}, "
        f"chosen on validation data to improve recall while monitoring "
        f"precision."
    )

    # ---------------------------------------------------------
# TASK 9 - FINAL MODEL + RANDOM FOREST THRESHOLD
# ---------------------------------------------------------
import joblib

print("\n" + "=" * 60)
print("TASK 9 - FINAL MODEL ARTIFACT")
print("=" * 60)

# ---------------------------------------------------------
# 1. Sweep the Random Forest's own probability output
# ---------------------------------------------------------
rf_thresholds = np.arange(0.10, 0.9001, 0.02)

rf_threshold_results = []

for threshold in rf_thresholds:
    predictions = (
        rf_test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    rf_threshold_results.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

rf_threshold_df = pd.DataFrame(
    rf_threshold_results
)

# Find the F1-maximizing Random Forest threshold
best_rf_row = rf_threshold_df.loc[
    rf_threshold_df["f1"].idxmax()
]

t_rf = float(best_rf_row["threshold"])
rf_best_f1 = float(best_rf_row["f1"])
rf_best_recall = float(best_rf_row["recall"])
rf_best_precision = float(best_rf_row["precision"])

print("\nRandom Forest threshold sweep:")
print(
    rf_threshold_df.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
        },
    )
)

print("\nRandom Forest F1-maximizing threshold (t*_rf):")
print(f"t*_rf     : {t_rf:.2f}")
print(f"F1        : {rf_best_f1:.4f}")
print(f"Recall    : {rf_best_recall:.4f}")
print(f"Precision : {rf_best_precision:.4f}")

# ---------------------------------------------------------
# 2. Save the winning fitted Random Forest pipeline
# ---------------------------------------------------------
models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "return_risk_model.pkl"

joblib.dump(
    rf_best_pipeline,
    model_path
)

print(f"\nFinal model saved to: {model_path}")