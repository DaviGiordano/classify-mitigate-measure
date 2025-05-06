from fairlearn.datasets import fetch_adult
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import List
import pandas as pd
import os
import pickle
from pathlib import Path


@dataclass(frozen=True)
class DatasetWrapper:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sensitive_train: pd.Series
    sensitive_test: pd.Series
    categorical_cols: List[str]
    numerical_cols: List[str]
    target_name: str
    sensitive_name: str
    dataset_name: str
