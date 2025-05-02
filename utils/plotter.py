from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
    accuracy_score,
)
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    false_negative_rate,
    count,
    selection_rate,
)


class Plotter:
    def __init__(self):
        self.metrics = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "selection_rate": selection_rate,
        }

    def confusion_plot(self, y_pred, y_test_pred, output_path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_pred,
            y_test_pred,
            ax=ax,
        )
        if output_path:
            fig.savefig(output_path)
        return fig

    def subgroup_metrics_plots(
        self, y_test, y_test_pred, sensitive_test, output_path=None
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        metric_frame = MetricFrame(
            metrics=self.metrics,
            y_true=y_test,
            y_pred=y_test_pred,
            sensitive_features=sensitive_test,
        )
        ax = metric_frame.by_group.plot(
            kind="bar",
            ylim=[0, 1],
            subplots=True,
            layout=[3, 3],
            figsize=[12, 8],
            legend=False,
            title="Metrics by subgroup",
        )
        if output_path:
            fig.savefig(output_path)
        return fig

    def overall_metrics_plots(
        self, y_test, y_test_pred, sensitive_test, output_path=None
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        metric_frame = MetricFrame(
            metrics=self.metrics,
            y_true=y_test,
            y_pred=y_test_pred,
            sensitive_features=sensitive_test,
        )
        ax = metric_frame.overall.plot(
            kind="bar",
            ylim=[0, 1],
        )
        if output_path:
            fig.savefig(output_path)
        return fig
