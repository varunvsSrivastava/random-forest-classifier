import joblib
from pathlib import Path

from sklearn.datasets import load_breast_cancer


def predict_single_sample():
    """Predict the class of one sample using the saved model."""

    model_path = Path("models/random_forest_model.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            "Saved model not found. Run python -m src.save_model first."
        )

    model = joblib.load(model_path)

    dataset = load_breast_cancer()

    sample = dataset.data[0].reshape(1, -1)
    actual_label = dataset.target[0]

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    print("===== Single Sample Prediction =====")
    print(f"Predicted class: {dataset.target_names[prediction]}")
    print(f"Actual class: {dataset.target_names[actual_label]}")
    print(f"Prediction correct: {prediction == actual_label}")

    print("\n===== Class Probabilities =====")
    for class_index, probability in enumerate(probabilities):
        class_name = dataset.target_names[class_index]
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    predict_single_sample()
