import pytest
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from src.prepare_data import split_dataset

def test_model_training_and_prediction():
    X_train,X_test,y_train,y_test = split_dataset()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    assert predictions.shape == y_test.shape
    assert set(np.unique(predictions)).issubset({0,1})
    
def test_saved_model_prediction():
    model_path = Path("models/random_forest_model.joblib")
    
    assert model_path.exists(), "Saved model file does not exist."
    
    model = joblib.load(model_path)
    dataset = load_breast_cancer()
    
    sample = dataset.data[0].reshape(1,-1)
    prediction = model.predict(sample)
    
    assert prediction.shape == (1,)
    assert prediction[0] in {0,1}
    
def test_model_rejects_wrong_numbers_of_features():
    model = joblib.load("models/random_forest_model.joblib")
    
    invalid_sample = np.zeros((1,29))
    
    with pytest.raises(ValueError):
        model.predict(invalid_sample)