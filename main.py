"""
main.py — Training Pipeline v5
================================
Usage:
  python main.py                          # optimal rebalance (recommended)
  python main.py --rebalance moderate     # less aggressive
  python main.py --rebalance aggressive   # equal classes
  python main.py --rebalance none         # original distribution
  python main.py --model svm              # single model
  python main.py --no-score               # skip label correction
"""

import os
import json
import pickle
import argparse

import pandas as pd

from src.preprocessing import clean_text
from src.feature_extraction import create_tfidf, add_score_feature
from src.train_model import train_model, clean_residual_artifacts


def main(
    model_type="ensemble",
    use_score=True,
    rebalance="optimal",
    use_score_feature=True,
):

    print(f"\n{'='*58}")
    print("  AI-Driven Consumer Insight - Training Pipeline v5")
    print(f"  Model     : {model_type.upper()}")
    print(f"  Labels    : {'score-corrected' if use_score else 'original'}")
    print(f"  Rebalance : {rebalance.upper()}")
    print(
        f"  Score feat: {'ENABLED' if use_score_feature else 'DISABLED'}"
    )
    print(f"{'='*58}\n")

    # ── 1. Load ───────────────────────────────────────────────
    csv_path = "data/preprocessed_reviews.csv"
    print(f"[1/5] Loading: '{csv_path}' ...")
    df = pd.read_csv(csv_path)
    print(f"      Rows: {len(df):,}")

    # ── 2. Text prep ──────────────────────────────────────────
    if "Cleaned_Text" not in df.columns or df["Cleaned_Text"].isna().sum() > 100:
        print("[2/5] Cleaning text ...")
        df["Cleaned_Text"] = df["Text"].apply(clean_text)
    else:
        print(
            "[2/5] Cleaned_Text found - removing residual HTML artifacts (br, nbsp ...) ..."
        )
        df["Cleaned_Text"] = clean_residual_artifacts(df["Cleaned_Text"])

    df = df.dropna(subset=["Cleaned_Text", "Sentiment"])

    print(f"\n      Sentiment distribution:")
    for lbl, cnt in df["Sentiment"].value_counts().items():
        bar = "#" * int(cnt / len(df) * 40)
        print(f"        {lbl:10s}: {cnt:6,} ({100*cnt/len(df):.1f}%)  {bar}")

    # ── 3. TF-IDF features (text ONLY — no score column) ─────
    print("\n[3/5] Extracting TF-IDF features ...")
    X, vectorizers = create_tfidf(df["Cleaned_Text"])
    if use_score_feature and "Score" in df.columns:
        X = add_score_feature(X, df["Score"].fillna(3))
        print(f"      Shape: {X.shape}  (word + char n-grams + score)")
    else:
        print(f"      Shape: {X.shape}  (word + char n-grams)")

    # ── 4. Train ──────────────────────────────────────────────
    score_series = df["Score"] if (use_score and "Score" in df.columns) else None
    y = df["Sentiment"]

    print("\n[4/5] Training ...")
    model, accuracy, report, cm, label_order = train_model(
        X,
        y,
        model_type=model_type,
        score_series=score_series,
        score_correction="all",
        rebalance=rebalance,
    )

    # ── Results ───────────────────────────────────────────────
    print(f"\n{'-'*58}")
    print(f"  TEST ACCURACY : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"{'-'*58}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix (labels: {label_order}):\n{cm}")

    if accuracy >= 0.95:
        print(f"\n  TARGET REACHED: {accuracy*100:.2f}%")
    elif accuracy >= 0.90:
        print(f"\n  STRONG: {accuracy*100:.2f}%")
    elif accuracy >= 0.85:
        print(
            f"\n  WARNING: {accuracy*100:.2f}% - acceptable, Neutral is hard on this dataset"
        )
    else:
        print(f"\n  LOW: {accuracy*100:.2f}% - try --rebalance optimal")

    # ── 5. Save ───────────────────────────────────────────────
    print("\n[5/5] Saving models ...")
    os.makedirs("models", exist_ok=True)

    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizers, f)

    meta = {
        "model_type": model_type,
        "use_score": bool(use_score),
        "use_score_feature": bool(use_score_feature),
        "rebalance": rebalance,
        "accuracy": round(accuracy, 4),
        "features": X.shape[1],
        "label_order": label_order,
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("      OK models/sentiment_model.pkl")
    print("      OK models/tfidf_vectorizer.pkl")
    print("      OK models/model_meta.json")
    print(f"\n{'='*58}  Done!\n")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["ensemble", "svm", "lr", "nb"], default="ensemble"
    )
    parser.add_argument(
        "--rebalance",
        choices=["none", "moderate", "aggressive", "optimal"],
        default="optimal",
    )
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--no-score-feature", action="store_true")
    args = parser.parse_args()
    main(
        model_type=args.model,
        use_score=not args.no_score,
        rebalance=args.rebalance,
        use_score_feature=not args.no_score_feature,
    )
