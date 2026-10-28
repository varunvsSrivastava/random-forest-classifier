from sklearn.model_selection import train_test_split
from src.load_data import load_dataset
import numpy as np

def split_dataset(test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    
    X,y,feature_names,target_names = load_dataset()
    
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=test_size,random_state=random_state,stratify=y
    )
    
    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_dataset()
    
    print("===== Split Verification =====")

    print(f"Total samples: {len(X_train) + len(X_test)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")

    print(f"\nTraining labels shape: {y_train.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    print("\n===== Data Integrity Checks =====")
    print(f"All training features finite: {np.isfinite(X_train).all()}")
    print(f"All testing features finite: {np.isfinite(X_test).all()}")
    print(f"Training and testing sample counts match: "
          f"{len(X_train) + len(X_test) == 569}")

    print("\n===== Class Distribution =====")
    print(f"Training Malignant: {(y_train == 0).sum()}")
    print(f"Training Benign: {(y_train == 1).sum()}")
    print(f"Testing Malignant: {(y_test == 0).sum()}")
    print(f"Testing Benign: {(y_test == 1).sum()}")
