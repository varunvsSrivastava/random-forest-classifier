
from pathlib import Path

from src.train_model import train_model
from src.visualize_model import (
    plot_confusion_matrix,
    plot_feature_importance,
)


def test_visualization_files_are_created():
    model, _, X_test, _, y_test = train_model(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    plot_confusion_matrix(model, X_test, y_test)
    plot_feature_importance(model)

    assert Path("reports/confusion_matrix.png").exists()
    assert Path("reports/feature_importance.png").exists()
