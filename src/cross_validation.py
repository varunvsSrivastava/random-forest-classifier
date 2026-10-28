from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.prepare_data import split_dataset

def evaluate_with_cross_validation():
    """Evaluate a Random Forest model using stratified 5-fold cross-validation."""
    
    X_train,X_test,y_train,y_test = split_dataset()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=5,
        random_state=42
    )
    
    cv = StratifiedKFold(
        n_splits=5,shuffle=True,random_state=42
    )
    
    scores = cross_val_score(
        model,X_train,y_train,cv=cv,scoring="accuracy",
    )
    
    print("===== 5-fold Cross-Validation =====")
    for fold_number,score in enumerate(scores,start=1):
        print(f"Fold {fold_number} accuracy: {score:.4f}")
    
    print(f"\nMean CV accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    
if __name__=="__main__":
    evaluate_with_cross_validation()