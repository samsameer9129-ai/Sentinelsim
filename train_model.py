
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sim_swap_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sim_swap_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# These are the inputs the model will use.
# Do not include account_id: it is an identifier, not a useful signal.
FEATURES = [
    "hours_since_sim_change",
    "sim_change_detected",
    "new_device",
    "failed_logins",
    "transaction_amount",
    "average_transaction_amount",
    "transaction_count_24h",
    "unusual_location",
    "recent_credential_change",
]

TARGET = "is_fraud"


# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run generate_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TARGET].isna().any():
        raise ValueError("The target column contains missing labels.")

    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels 0 and 1, got: {labels}"
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "The dataset must contain both legitimate and fraud examples."
        )

    # Convert feature columns to numeric values.
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinite values with missing values.
    df[FEATURES] = df[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    # Reject columns that contain no usable values at all.
    all_missing = [
        col for col in FEATURES if df[col].isna().all()
    ]
    if all_missing:
        raise ValueError(
            f"Features have no usable values: {all_missing}"
        )

    return df


# --------------------------------------------------
# 3. Train and evaluate model
# --------------------------------------------------

def main():
    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("=" * 60)
    print("SIM-SWAP FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    print(f"\nDataset rows: {len(df):,}")
    print(f"Input features: {len(FEATURES)}")

    print("\nLabel distribution:")
    print(
        y.map({
            0: "Legitimate",
            1: "Simulated fraud"
        }).value_counts().to_string()
    )

    # Keep both classes represented in the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Imputation is included in the pipeline so preprocessing is
    # learned from training data only.
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Predictions and fraud probabilities.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics.
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Simulated fraud"],
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    tn, fp, fn, tp = matrix.ravel()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print("Rows = actual class; columns = predicted class")
    print("                    Predicted legitimate | Predicted fraud")
    print(f"Actual legitimate:  {tn:>10}           | {fp:>10}")
    print(f"Actual fraud:       {fn:>10}           | {tp:>10}")

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"Average precision: {avg_precision:.4f}")

    # Feature importance from the fitted Random Forest.
    classifier = model.named_steps["classifier"]
    importance = pd.Series(
        classifier.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    print(importance.to_string())

    # Save model and metadata.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "threshold": 0.50,
        "model_name": "Random Forest",
        "dataset_type": "synthetic",
        "random_state": RANDOM_STATE,
    }

    joblib.dump(artifact, MODEL_PATH)

    metrics_text = (
        "SIM-SWAP FRAUD DETECTION MODEL EVALUATION\n"
        "Dataset: Synthetic demonstration data\n\n"
        f"Rows: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"ROC-AUC: {roc_auc:.4f}\n"
        f"Average precision: {avg_precision:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
        "Confusion matrix (actual rows, predicted columns):\n"
        f"{matrix}\n\n"
        "Feature importance:\n"
        f"{importance.to_string()}\n"
    )

    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
