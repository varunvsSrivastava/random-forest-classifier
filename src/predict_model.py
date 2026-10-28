import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

from src.prepare_data import split_dataset

def evaluate_saved_model():
    """Load the saved model and evaluate it on the test set."""
    
    model_path = Path("models/random_forest_model.joblib")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}, "
            "Run python -m src.save_model first."
        )
        
    model = joblib.load(model_path)
    
    X_train, X_test, y_train, y_test = split_dataset()
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    print("===== Saved Model Verification =====")
    print(f"Model loaded from: {model_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Predictions generated: {len(y_pred)}")
    
if __name__=="__main__":
    evaluate_saved_model()