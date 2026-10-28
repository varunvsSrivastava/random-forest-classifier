
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.prepare_data import split_dataset


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained classifier on test data."""

    predictions = model.predict(X_test)

    print("===== Model Evaluation =====")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Malignant precision: "
        f"{precision_score(y_test, predictions, pos_label=0):.4f}"
    )
    print(
        "Malignant recall: "
        f"{recall_score(y_test, predictions, pos_label=0):.4f}"
    )
    print(
        "Malignant F1-score: "
        f"{f1_score(y_test, predictions, pos_label=0):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            digits=4,
        )
    )


if __name__ == "__main__":
    model_path = Path("models/random_forest_model.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            "Saved model not found. Run python -m src.save_model first."
        )

    model = joblib.load(model_path)
    _, X_test, _, y_test = split_dataset()

    evaluate_model(model, X_test, y_test)
