import numpy as np

from src.load_data import load_dataset
from src.prepare_data import split_dataset

def test_dataset_loading():
    X, y, feature_names, target_names = load_dataset() 
    
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert len(feature_names) == 30
    assert list(target_names) == ["malignant","benign"]
    assert np.isfinite(X).all()
    
def test_train_test_split():
    X_train, X_test, y_train, y_test = split_dataset()
    
    assert X_train.shape == (455,30)
    assert X_test.shape == (114,30)
    assert y_train.shape == (455,)
    assert y_test.shape == (114,)
    
    assert set(np.unique(y_train)) == {0,1}
    assert set(np.unique(y_test)) == {0,1}