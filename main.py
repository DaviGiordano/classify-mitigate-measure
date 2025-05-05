from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.dataset_utils import load_adult, save_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from fairgbm import FairGBMClassifier
from utils.model import Model
from typing import List

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
import mlflow


EXPERIMENT_NAME = "compare_models"


def make_column_transformer(
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> ColumnTransformer:
    """Create a reusable preprocessing transformer."""
    return ColumnTransformer(
        [
            ("scale", StandardScaler(), numerical_cols),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="passthrough",
    )


def main():

    dataset = load_adult()
    column_transformer = make_column_transformer(
        dataset.categorical_cols,
        dataset.numerical_cols,
    )
    save_dataset(
        dataset,
        column_transformer,
        "./data",
    )
    clfs = {
        "log_reg": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "grad_boost": GradientBoostingClassifier(),
        "hist_grad_boost": HistGradientBoostingClassifier(),
        "fairgbm_clf": FairGBMClassifier(
            constraint_type="FNR,FPR",
        ),
    }

    for name, clf in clfs.items():
        model = Model(
            column_transformer,
            estimator=clf,
            dataset=dataset,
        )
        with mlflow.start_run(run_name=name):
            model.fit_predict()
            results = model.evaluate()
            mlflow.log_param("model_name", name)
            # model.plot_metrics()


if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT_NAME)

    main()
