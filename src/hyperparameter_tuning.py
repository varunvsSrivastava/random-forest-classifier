from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.prepare_data import split_dataset

def tune_hyperparameters():
    """Find suitable Random Forest hyperparamater using grid search."""
    
    X_train, X_test, y_trian, y_test = split_dataset()
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    param_grid = {
        "max_depth" : [3,4,5],
        "min_samples_split" : [2,10],
        "min_samples_leaf" : [1,5]
    }
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train,y_trian)
    
    print("===== Grid Search Results =====")
    print(f"Best paramaters: {grid_search.best_params_}")
    print(f"Best mean CV accuracy: {grid_search.best_score_:.4f}")
    
    print("\n===== Top 5 Configuration =====")
    results = grid_search.cv_results_
    ranked_indices = results["rank_test_score"].argsort()
    
    for index in ranked_indices[:5]:
        print(
            f"Parameters: {results['params'][index]}, "
            f"Mean CV accuracy: {results['mean_test_score'][index]:.4f}, "
            f"Std: {results['std_test_score'][index]:.4f}, "
        )
    
    return grid_search.best_estimator_,grid_search.best_params_, grid_search.best_score_
        
def evaluate_best_model():
    """Evaluate the best tuned model on the held-out test set."""
    
    X_train,X_test,y_train,y_test = split_dataset()
    
    model = RandomForestClassifier(
        n_estimators=100,random_state=42
    )
    
    param_grid = {
        "max_depth": [3,4,5],"min_samples_split": [2,10],"min_samples_leaf": [1,5]
    }
    
    cv = StratifiedKFold(
        n_splits=5,shuffle=True,random_state=42
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )
    
    grid_search.fit (X_train,y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n===== Model Test Evaluation =====")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Mean CV accuracy: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy_score(y_test,y_pred):.4f}")
    
    print(
        "Malignant precision: "
        f"{precision_score(y_test,y_pred,pos_label=0):.4f}"
    )
    
    print(
        "Malignant recall: "
        f"{recall_score(y_test,y_pred,pos_label=0):.4f}"
    )
    print(
        "Malignant F1-score: "
        f"{f1_score(y_test,y_pred,pos_label=0):.4f}"
    )
    
    print("\nConfusion Matrix (labels=[0,1]):")
    print(confusion_matrix(y_test,y_pred,labels=[0,1]))
    
    print("\nClassification Report: ")
    print(
        classification_report(
            y_test,y_pred,labels=[0,1],target_names=["malignant","benign"],digits=4
        )
    )
    
        
if __name__=="__main__":
    tune_hyperparameters()
    evaluate_best_model()