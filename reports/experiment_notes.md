# Random Forest Experiment Notes

## Objective
Evaluate how Random Forest hyperparameters affect training accuracy, testing accuracy, and the accuracy gap.

## Dataset
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Total samples: 569
- Features: 30
- Training samples: 455
- Testing samples: 114
- Random state: 42

## Observations
1. Increasing `max_depth` did not improve test accuracy in the experiments performed.
2. `max_depth=6` produced lower test accuracy than the other tested depth configurations.
3. Increasing `min_samples_leaf` from 1 to 5 reduced the training accuracy and accuracy gap.
4. Setting `min_samples_split=10` did not change the reported metrics in the tested configuration.
5. Several configurations achieved the same test accuracy, so further validation is needed.

## Conclusion
The current experiments suggest that limiting tree depth and increasing minimum leaf size may help control overfitting. Cross-validation and class-specific evaluation are needed before selecting the final model.

## Cross-Validation Results

- Method: Stratified 5-fold cross-validation
- Scoring metric: Accuracy
- Mean CV accuracy: 95.38%
- Standard deviation: 1.76 percentage points
- Held-out test accuracy: 95.61%

## Interpretation
The mean cross-validation accuracy is close to the held-out test accuracy. The fold scores range from 93.41% to 97.80%, indicating some variation across validation splits. Further evaluation using precision, recall, F1-score, and confusion matrices is required before selecting the final model.

## Grid Search Results

- Method: GridSearchCV with stratified 5-fold cross-validation
- Hyperparameter combinations: 12
- Total model fits: 60
- Scoring metric: Accuracy
- Best parameters:
  - `max_depth=5`
  - `min_samples_leaf=1`
  - `min_samples_split=2`
- Best mean CV accuracy: 96.04%
- Standard deviation: 1.79 percentage points

### Interpretation
GridSearchCV identified a configuration with a higher mean cross-validation accuracy than the previously evaluated manual configuration. The selected configuration must still be evaluated on the untouched test set before final conclusions are made.
