"""Classical baselines (methodology Sec 9.1, items 2-3): byte-bigram
histogram features feeding SVM and Random Forest. Cheap, interpretable,
expected to plateau on compressed/compound classes — that plateau is itself
part of the comparison (Sec 5 table).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

N_BIGRAMS = 256 * 256


def bigram_histogram(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    if len(arr) < 2:
        return np.zeros(N_BIGRAMS, dtype=np.float32)
    pairs = arr[:-1].astype(np.int32) * 256 + arr[1:].astype(np.int32)
    hist = np.bincount(pairs, minlength=N_BIGRAMS).astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def featurize(fragments: list) -> tuple:
    X = np.stack([bigram_histogram(f["data"]) for f in fragments])
    y = np.array([f["label"] for f in fragments])
    return X, y


def make_svm(seed: int) -> SVC:
    return SVC(kernel="rbf", probability=True, random_state=seed)


def make_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


def make_xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(n_estimators=200, random_state=seed, tree_method="hist")
