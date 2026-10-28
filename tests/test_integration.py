
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.load_data import load_dataset
from src.prepare_data import split_dataset


def test_end_to_end_training_pipeline():
    X, y, feature_names, target_names = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=5,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert len(feature_names) == 30
    assert len(target_names) == 2
    assert predictions.shape == y_test.shape
    assert 0.0 <= accuracy <= 1.0
