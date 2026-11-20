"""Shared helpers: paths, feature lists, preprocessing pipeline, data split."""
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "login_attempts.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

NUM_FEATURES = ["hour_of_day", "failed_attempts", "ip_reputation_score",
                "requests_per_min", "session_duration_s", "bytes_transferred_kb",
                "uses_vpn"]
CAT_FEATURES = ["country", "device_type", "auth_method"]
TARGET = "label"          # 0 = normal, 1 = suspicious
GROUP = "user_region"     # NOT a model feature; used only for the fairness audit


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_preprocessor(num=None, cat=None) -> ColumnTransformer:
    """Missing values -> imputation, categoricals -> one-hot, numerics -> standardised."""
    num = NUM_FEATURES if num is None else num
    cat = CAT_FEATURES if cat is None else cat
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def split_data(df: pd.DataFrame):
    """Stratified 75/25 split. Returns train_df, test_df (full frames, keeps GROUP)."""
    return train_test_split(df, test_size=0.25, stratify=df[TARGET],
                            random_state=RANDOM_STATE)


def save_json(name: str, obj) -> None:
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(name: str):
    with open(OUT / name) as f:
        return json.load(f)
