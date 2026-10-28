# Random Forest Classifier

## Project Overview

This project implements a Random Forest classification pipeline using the Breast Cancer Wisconsin (Diagnostic) dataset. It covers data preparation, model training, evaluation, hyperparameter tuning, cross-validation, model persistence, visualization, and automated testing.

**Purpose:** Learn and demonstrate a structured, reusable machine learning workflow.

## Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** Built-in scikit-learn dataset
- **Samples:** 569
- **Features:** 30 numerical features
- **Classes:** Malignant and benign
- **Training samples:** 455
- **Testing samples:** 114

Target labels:
- `0`: Malignant
- `1`: Benign

## Technologies

- Python
- scikit-learn
- NumPy
- pandas
- Matplotlib
- Seaborn
- Joblib
- pytest

## Project Structure

```text
random-forest-classifier/
├── data/
├── models/
│   └── random_forest_model.joblib
├── reports/
│   ├── confusion_matrix.png
│   ├── experiment_notes.md
│   └── feature_importance.png
├── src/
│   ├── load_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── visualize_model.py
│   ├── cross_validation.py
│   ├── hyperparameter_tuning.py
│   ├── save_model.py
│   ├── predict_model.py
│   └── predict_single.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_integration.py
│   ├── test_model.py
│   └── test_visualization.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run commands from the project root directory.

Load and inspect the dataset:

```bash
python -m src.load_data
```

Prepare the train-test split:

```bash
python -m src.prepare_data
```

Train the model:

```bash
python -m src.train_model
```

Evaluate the model:

```bash
python -m src.evaluate_model
```

Run cross-validation:

```bash
python -m src.cross_validation
```

Tune hyperparameters:

```bash
python -m src.hyperparameter_tuning
```

Save the best model:

```bash
python -m src.save_model
```

Verify the saved model:

```bash
python -m src.predict_model
```

Predict a single sample:

```bash
python -m src.predict_single
```

Generate visualizations:

```bash
python -m src.visualize_model
```

## Model Evaluation

The best configuration identified by GridSearchCV was:

- `n_estimators=100`
- `max_depth=5`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `random_state=42`

Reported results:

| Metric | Result |
|---|---:|
| Mean 5-fold CV accuracy | 96.04% |
| Test accuracy | 95.61% |
| Malignant precision | 95.12% |
| Malignant recall | 92.86% |
| Malignant F1-score | 93.98% |

The test set contains 114 samples. Results may vary if the dataset, split, random seed, or model configuration changes.

## Visualizations

Generated reports are saved in the `reports/` directory:

- `confusion_matrix.png`
- `feature_importance.png`

Detailed experimentation observations are documented in `reports/experiment_notes.md`.

## Testing

Run all automated tests:

```bash
python -m pytest -v
```

The current test suite covers dataset loading, train-test splitting, model training, saved-model predictions, invalid feature input, integration, and visualization file generation.

## Limitations

- This project is intended for educational purposes.
- The dataset is a benchmark dataset, not a substitute for clinical validation.
- Accuracy alone does not establish that a model is suitable for medical diagnosis.
- Feature importance indicates how the model uses features, not whether those features cause an outcome.

## Future Improvements

- Add more comprehensive unit and integration tests.
- Compare Random Forest against a Decision Tree and other classifiers.
- Evaluate precision, recall, F1-score, and confusion matrices across validation folds.
- Improve configuration management and reduce duplicated training logic.
- Add a validated input interface for new samples.

## License

No license has been specified for this project yet.
