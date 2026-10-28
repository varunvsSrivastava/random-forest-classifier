
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from src.prepare_data import split_dataset
from src.train_model import train_model
from sklearn.datasets import load_breast_cancer


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
    model, _, X_test, _, y_test = train_model(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model)

    print("Saved: reports/confusion_matrix.png")
    print("Saved: reports/feature_importance.png")
