from fairgbm import FairGBMClassifier
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio,
    equalized_odds_ratio,
    true_positive_rate,
    true_negative_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import OrdinalEncoder
from src.custom_metrics import positive_rate
from src.dataset_wrapper import DatasetWrapper
from src.mlflow_decorator import log_dict_output
from flatten_dict import flatten


class Model:
    def __init__(
        self,
        column_transformer: ColumnTransformer,
        estimator: BaseEstimator,
        dataset: DatasetWrapper,
    ):
        self.column_transformer = column_transformer
        self.estimator = estimator
        self.dataset = dataset
        self.pipeline = Pipeline(
            [
                ("column_transformer", self.column_transformer),
                ("estimator", self.estimator),
            ]
        )
        self.y_test_pred = None
        self.y_test_prob = None

    def fit_predict(self):

        if isinstance(self.estimator, FairGBMClassifier):
            enc = OrdinalEncoder()
            sensitive_train_encoded = enc.fit_transform(
                pd.DataFrame(self.dataset.sensitive_train)
            )
            self.pipeline.fit(
                self.dataset.X_train,
                self.dataset.y_train,
                estimator__constraint_group=np.ravel(sensitive_train_encoded),
            )
        else:
            self.pipeline.fit(
                self.dataset.X_train,
                self.dataset.y_train,
            )

        self.y_test_pred = self.pipeline.predict(self.dataset.X_test)
        if hasattr(self.estimator, "predict_proba"):
            self.y_test_prob = self.pipeline.predict_proba(self.dataset.X_test)[:, 1]

    @log_dict_output
    def evaluate(self) -> dict:

        performance_metrics = self._evaluate_overall_performance()
        overall_fairness_metrics = self._evaluate_overall_fairness()
        subgroup_fairness_metrics = self._evaluate_subgroup_fairness()

        return (
            performance_metrics | overall_fairness_metrics | subgroup_fairness_metrics
        )

    def _evaluate_overall_performance(self):
        performance_kwargs = {
            "y_true": self.dataset.y_test,
            "y_pred": self.y_test_pred,
        }
        return {
            "overall.accuracy": accuracy_score(**performance_kwargs),
            "overall.precision": precision_score(**performance_kwargs),
            "overall.recall": recall_score(**performance_kwargs),
            "overall.f1": f1_score(**performance_kwargs),
            "overall.true_positive_rate": true_positive_rate(**performance_kwargs),
            "overall.true_negative_rate": true_negative_rate(**performance_kwargs),
            "overall.false_positive_rate": false_positive_rate(**performance_kwargs),
            "overall.false_negative_rate": false_negative_rate(**performance_kwargs),
            "overall.selection_rate": selection_rate(**performance_kwargs),
        }

    def _evaluate_overall_fairness(self):
        fairness_kwargs = {
            "y_true": self.dataset.y_test,
            "y_pred": self.y_test_pred,
            "sensitive_features": self.dataset.sensitive_test,
        }

        return {
            "dem_parity_diff": demographic_parity_difference(**fairness_kwargs),
            "dem_parity_ratio": demographic_parity_ratio(**fairness_kwargs),
            "eq_odds_diff": equalized_odds_difference(**fairness_kwargs),
            "eq_odds_ratio": equalized_odds_ratio(**fairness_kwargs),
        }

    def _evaluate_subgroup_fairness(self):
        subgroup_metrics_fns = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "true_positive_rate": true_positive_rate,
            "true_negative_rate": true_negative_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "selection_rate": selection_rate,
            "count": count,
            "positive_rate": positive_rate,
        }
        metric_frame = MetricFrame(
            metrics=subgroup_metrics_fns,
            y_true=self.dataset.y_test,
            y_pred=self.y_test_pred,
            sensitive_features=self.dataset.sensitive_test,
        )
        return flatten(
            metric_frame.by_group.to_dict(orient="index"),
            reducer="dot",
        )
