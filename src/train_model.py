"""
train_model.py  —  v5  (target: 93-95% honest accuracy)
=========================================================

ROOT CAUSE OF LOW ACCURACY (diagnosed from confusion matrix):
  • "aggressive" throws away 80% of Positive training data (31k→5k).
    Model forgets Positive → misclassifies 1,385 Positive reviews → 78% accuracy.
  • "moderate" gives Neutral too little signal (only 3k Neutral in train).

SOLUTION — "optimal" strategy (new):
  • Positive  : undersample to 4× Negative  (~22k) — keeps strong Positive signal
  • Negative  : keep all (~5.6k)
  • Neutral   : oversample to 3× original   (~9k)
  Net result: Neutral gets 3× more training exposure vs moderate,
  Positive keeps 4× more data vs aggressive. Best of both worlds.
  Expected accuracy: 91–94% on honest test set.

ALSO FIXED:
  • HTML <br> artifacts stripped from Cleaned_Text before TF-IDF
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample


# ─────────────────────────────────────────────────────────────
# Strip residual HTML artifacts from pre-cleaned text
# ─────────────────────────────────────────────────────────────


def clean_residual_artifacts(text_series: pd.Series) -> pd.Series:
    """Remove 'br', 'nbsp' etc. that survived earlier preprocessing."""
    _ARTIFACTS = r"\b(br|nbsp|amp|lt|gt|quot|apos|href|div|span|img|src)\b"
    return (
        text_series.str.replace(_ARTIFACTS, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ─────────────────────────────────────────────────────────────
# Dataset rebalancing — training split ONLY
# ─────────────────────────────────────────────────────────────


def rebalance_dataset(X_sparse, y_series, strategy: str = "optimal"):
    """
    "optimal"    → Pos=4×Neg, Neutral=3×original capped at 1.5×Neg  ← USE THIS
    "moderate"   → Pos=3×Neg, Neutral=2×original capped at Neg
    "aggressive" → all classes equal to Neg count
    "none"       → no rebalancing
    """
    if strategy == "none":
        return X_sparse, np.array(y_series)

    y_arr = np.array(y_series)
    classes, counts = np.unique(y_arr, return_counts=True)
    count_dict = dict(zip(classes, counts))

    print(f"\n  Rebalancing (strategy='{strategy}') ...")
    print(f"  Before: { {k: int(v) for k, v in count_dict.items()} }")

    neg_count = count_dict.get("Negative", 0)
    neu_count = count_dict.get("Neutral", 0)
    pos_count = count_dict.get("Positive", 0)

    if strategy == "optimal":
        target = {
            "Negative": neg_count,
            "Neutral": min(neu_count * 3, int(neg_count * 1.5)),
            "Positive": min(neg_count * 4, pos_count),
        }
    elif strategy == "aggressive":
        target = {cls: neg_count for cls in classes}
    else:  # moderate
        target = {
            "Negative": neg_count,
            "Neutral": min(neu_count * 2, neg_count),
            "Positive": neg_count * 3,
        }

    indices_rebal = []
    for cls in classes:
        cls_idx = np.where(y_arr == cls)[0]
        t = target.get(cls, len(cls_idx))
        chosen = resample(
            cls_idx, replace=(t > len(cls_idx)), n_samples=t, random_state=42
        )
        indices_rebal.append(chosen)

    all_idx = np.concatenate(indices_rebal)
    np.random.default_rng(42).shuffle(all_idx)

    X_r = X_sparse[all_idx]
    y_r = y_arr[all_idx]

    new_cls, new_cnt = np.unique(y_r, return_counts=True)
    total = len(y_r)
    print(f"  After  : ", end="")
    for c, n in zip(new_cls, new_cnt):
        print(f"{c}={n:,}({100*n/total:.1f}%)  ", end="")
    print(f"\n  Total  : {total:,} samples")

    return X_r, y_r


# ─────────────────────────────────────────────────────────────
# Score helpers
# ─────────────────────────────────────────────────────────────


def score_to_sentiment(score: float) -> str:
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"


def apply_score_correction(y_series, score_series, confidence: str = "all"):
    y = y_series.copy()
    scores = score_series.copy()
    if confidence == "high":
        mask = scores.isin([1, 5])
    elif confidence == "medium":
        mask = scores.isin([1, 2, 4, 5])
    else:
        mask = scores.notna()
    y[mask] = scores[mask].apply(score_to_sentiment)
    return y


# ─────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────


def build_ensemble():
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
    svm_cal = CalibratedClassifierCV(svm, cv=3, method="sigmoid")
    lr = LogisticRegression(
        C=4.0, class_weight="balanced", max_iter=3000, solver="saga", random_state=42
    )
    nb = MultinomialNB(alpha=0.05)

    return VotingClassifier(
        estimators=[("svm", svm_cal), ("lr", lr), ("nb", nb)],
        voting="soft",
        weights=[3, 2, 1],
        n_jobs=1,
    )


def build_single_model(model_type: str):
    if model_type == "svm":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
    elif model_type == "lr":
        return LogisticRegression(
            C=4.0,
            class_weight="balanced",
            max_iter=3000,
            solver="saga",
            random_state=42,
        )
    else:
        return MultinomialNB(alpha=0.05)


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────


def train_model(
    X,
    y,
    model_type: str = "ensemble",
    score_series=None,
    score_correction: str = "all",
    rebalance: str = "optimal",
):
    # Step 1 — label correction only (score never used as feature)
    if score_series is not None:
        print(f"  -> Label correction (confidence='{score_correction}') ...")
        y_clean = apply_score_correction(y, score_series, confidence=score_correction)
        corrections = (y_clean != y).sum()
        print(
            f"     Corrected: {corrections:,} / {len(y):,} ({100*corrections/len(y):.1f}%)"
        )
        for lbl, cnt in y_clean.value_counts().items():
            print(f"       {lbl:10s}: {cnt:6,}  ({100*cnt/len(y_clean):.1f}%)")
    else:
        y_clean = y.copy()

    # Step 2 — split FIRST (test set = real distribution, no leakage)
    y_arr = np.array(y_clean)
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X,
        y_arr,
        test_size=0.2,
        random_state=42,
        stratify=y_arr,
    )
    print(
        f"\n  Split - Train: {X_train_raw.shape[0]:,}  Test: {X_test.shape[0]:,}  (no leakage)"
    )

    # Step 3 — rebalance training split only
    X_train, y_train = rebalance_dataset(
        X_train_raw, pd.Series(y_train_raw), strategy=rebalance
    )
    print(
        f"\n  Training : {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features"
    )
    print(
        f"  Test     : {X_test.shape[0]:,} samples  (original distribution, honest eval)"
    )
    print(f"\n  Class distribution (train):")
    for lbl, cnt in zip(*np.unique(y_train, return_counts=True)):
        bar = "#" * int(cnt / len(y_train) * 30)
        print(f"    {lbl:10s}: {cnt:5,} ({100*cnt/len(y_train):.1f}%) {bar}")

    # Step 4 — build and fit
    print(
        f"\n  Building {'Voting Ensemble (SVM+LR+NB)' if model_type=='ensemble' else model_type.upper()} ..."
    )
    model = (
        build_ensemble() if model_type == "ensemble" else build_single_model(model_type)
    )
    print("  Fitting ... (this takes 2-5 minutes)")
    model.fit(X_train, y_train)

    # Step 5 — evaluate
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    label_order = sorted(np.unique(y_arr).tolist())
    report = classification_report(y_test, pred, labels=label_order, digits=4)
    cm = confusion_matrix(y_test, pred, labels=label_order)

    return model, accuracy, report, cm, label_order
