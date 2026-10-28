
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

from src.prepare_data import split_dataset


def plot_confusion_matrix(model, X_test, y_test):
    """Plot and save the confusion matrix."""

    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    plt.close()


def plot_feature_importance(model):
    """Plot and save the top 10 feature importances."""

    dataset = load_breast_cancer()
    importances = model.feature_importances_
    top_indices = importances.argsort()[-10:][::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[top_indices],
        y=dataset.feature_names[top_indices],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png")
    plt.close()


if __name__ == "__main__":
    model_path = Path("models/random_forest_model.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            "Saved model not found. Run python -m src.save_model first."
        )

    model = joblib.load(model_path)
    _, X_test, _, y_test = split_dataset()

    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model)

    print("Saved: reports/confusion_matrix.png")
    print("Saved: reports/feature_importance.png")
