
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.prepare_data import split_dataset
from src.train_model import train_model


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
    model, _, X_test, _, y_test = train_model(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    evaluate_model(model, X_test, y_test)
