import joblib
from pathlib import Path

from src.hyperparameter_tuning import tune_hyperparameters

def save_best_model():
    """Save the best Random Forest model found by grid search."""
    
    best_model, best_params, best_score = tune_hyperparameters()
    
    model_path = Path("models/random_forest_model.joblib")
    model_path.parent.mkdir(parents=True,exist_ok=True)
    
    joblib.dump(best_model, model_path)
    
    print(f"Best parameters: {best_params}")
    print(f"Mean CV accuracy: {best_score:.4f}")
    print(f"Model saved to: {model_path}")
    
if __name__=="__main__":
    save_best_model()