from sklearn.ensemble import RandomForestClassifier
from src.prepare_data import split_dataset

def train_model(max_depth=None,min_samples_leaf=1,min_samples_split=2,n_estimators=100,random_state=42):
    """Train a Ramdom forest classifier with a configurable depth."""
    
    X_train,X_test,y_train,y_test = split_dataset()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        random_state=random_state
    )
    
    model.fit(X_train,y_train)
    
    return model,X_train,X_test,y_train,y_test

if __name__=="__main__":
    model,X_train,X_test,y_train,y_test = train_model(
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=2
    )
    
    train_accuracy = model.score(X_train,y_train)
    test_accuracy = model.score(X_test,y_test)
    
    print("===== Random Forest Training =====")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    print(f"Accuracy gap: {train_accuracy - test_accuracy:.4f}")