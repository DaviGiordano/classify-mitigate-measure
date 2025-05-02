import functools
import numbers
import mlflow


def log_dict_output(func):
    """Decorator: log a dict return-value to the active MLflow run
    (numeric values → metrics, everything else → params)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        if not isinstance(out, dict):
            raise TypeError("Function must return a dict")
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run")
        for k, v in out.items():
            if isinstance(v, numbers.Number):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, str(v))
        return out

    return wrapper
