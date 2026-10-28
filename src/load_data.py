
from sklearn.datasets import load_breast_cancer
import numpy as np

def load_dataset():
    """Load and return the Breast Cancer Wisconsin dataset."""
    dataset = load_breast_cancer()
    
    X = dataset.data
    y = dataset.target

    return X,y,dataset.feature_names,dataset.target_names

def explore_dataset(X,y,feature_names,target_names):
    """Display basic information about the dataset."""
    
    print("===== Dataset Overview =====")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    print("\n===== Feature Information =====")
    print(f"Number of feature names: {len(feature_names)}")
    print(f"Data type: {X.dtype}")
    print(f"All feature values are finite: {np.isfinite(X).all()}")
    
    print("\n===== Missing values checks =====")
    print(f"Missing values in X: {np.isnan(X).sum()}")
    print(f"Missing values in y: {np.isnan(y).sum()}")
    
    print("\n===== Target Distribution =====")
    unique_labels, counts = np.unique(y, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        print(f"{target_names[label]}:{count} samples")
        
    print("\n===== First Five Samples =====")
    print(X[:5])    
    
    print("\n===== First Five Target Labels =====")
    print(y[:5])

if __name__ == "__main__":
    X,y,feature_names,target_names = load_dataset()
    explore_dataset(X,y,feature_names,target_names)
    
