from sklearn.compose import ColumnTransformer
from src.dataset_wrapper import DatasetWrapper
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def load_adult(
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> DatasetWrapper:
    """Load and split the Adult dataset, wrapped in a Dataset dataclass."""

    SENSITIVE_NAME = "sex"
    TARGET_NAME = "income"

    adult = fetch_adult(as_frame=True)
    X = adult.data.drop(columns="fnlwgt")
    y = adult.target.map({"<=50K": 0, ">50K": 1})

    categorical_cols = X.select_dtypes(include="category").columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X[categorical_cols] = X[categorical_cols].astype("object")
    X.fillna("Missing", inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    sensitive_train = X_train[SENSITIVE_NAME]
    sensitive_test = X_test[SENSITIVE_NAME]

    return DatasetWrapper(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_train=sensitive_train,
        sensitive_test=sensitive_test,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_name=TARGET_NAME,
        sensitive_name=SENSITIVE_NAME,
        dataset_name="adult",
    )


def save_dataset(
    dataset: DatasetWrapper,
    column_transformer: ColumnTransformer,
    base_path: str = "data",
):
    """Save dataset in both raw and processed forms."""

    def _save_set_datasets(X_train, X_test, y_train, y_test, target_dir):
        X_train.to_csv(
            os.path.join(target_dir, "X_train.csv"),
            index=False,
        )
        X_test.to_csv(
            os.path.join(target_dir, "X_test.csv"),
            index=False,
        )
        pd.Series(y_train, name="y").to_csv(
            os.path.join(target_dir, "y_train.csv"),
            index=False,
        )
        pd.Series(y_test, name="y").to_csv(
            os.path.join(target_dir, "y_test.csv"),
            index=False,
        )

        xy_train = X_train.copy()
        xy_train["y"] = y_train
        xy_train.to_csv(
            os.path.join(target_dir, "Xy_train.csv"),
            index=False,
        )
        xy_test = X_test.copy()
        xy_test["y"] = y_test
        xy_test.to_csv(
            os.path.join(target_dir, "Xy_test.csv"),
            index=False,
        )

    # Directories
    raw_dir = os.path.join(base_path, dataset.dataset_name, "raw")
    proc_dir = os.path.join(base_path, dataset.dataset_name, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    # Save raw
    _save_set_datasets(
        dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test, raw_dir
    )

    # Processed: fit and transform
    Xt_train = column_transformer.fit_transform(dataset.X_train)
    Xt_test = column_transformer.transform(dataset.X_test)

    # Get feature names from column transformer
    feature_names = column_transformer.get_feature_names_out()

    # Create DataFrames with proper column names
    X_train_proc = pd.DataFrame(
        Xt_train, columns=feature_names, index=dataset.X_train.index
    )
    X_test_proc = pd.DataFrame(
        Xt_test, columns=feature_names, index=dataset.X_test.index
    )

    # Save processed splits
    _save_set_datasets(
        X_train_proc, X_test_proc, dataset.y_train, dataset.y_test, proc_dir
    )
