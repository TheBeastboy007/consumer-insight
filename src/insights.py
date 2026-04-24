"""
insights.py
-----------
Generates product-level and corpus-level insights from a reviews DataFrame.

Expected DataFrame columns:
  - ProductId   : str
  - Sentiment   : "Positive" | "Neutral" | "Negative"
  - Cleaned_Text: pre-processed review text (space-separated tokens)
  - Score       : numeric star rating (optional)
  - Time        : unix timestamp (optional)
"""

from collections import Counter
import re
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Extra noise tokens to suppress in word charts
# ──────────────────────────────────────────────
_FREQ_NOISE = {
    "br",
    "one",
    "get",
    "also",
    "would",
    "use",
    "make",
    "really",
    "much",
    "well",
    "even",
    "time",
    "like",
    "just",
    "still",
    "go",
    "got",
    "give",
    "come",
    "though",
    "thing",
    "way",
    "see",
    "back",
    "two",
    "three",
    "first",
    "year",
    "day",
    "could",
    "used",
    "try",
    "ive",
    "im",
    "dont",
    "didnt",
    "cant",
    "wasnt",
    "isnt",
    "wont",
    "said",
    "say",
    "put",
    "made",
    "want",
    "need",
    "lot",
    "little",
    "bit",
    "many",
    "every",
    "another",
    "something",
    "nothing",
}


# ──────────────────────────────────────────────
# 1. Corpus-level distribution
# ──────────────────────────────────────────────


def sentiment_distribution(df: pd.DataFrame) -> dict:
    counts = df["Sentiment"].value_counts()
    total = len(df)
    return {
        label: {"count": int(cnt), "pct": round(100 * cnt / total, 1)}
        for label, cnt in counts.items()
    }


# ──────────────────────────────────────────────
# 2. Word frequency per sentiment
# ──────────────────────────────────────────────


def top_words(df: pd.DataFrame, sentiment: str, top_n: int = 20) -> list:
    """Top meaningful words for a sentiment class (noise filtered)."""
    subset = df[df["Sentiment"] == sentiment]["Cleaned_Text"].dropna()
    all_words = " ".join(subset).split()
    all_words = [w for w in all_words if w not in _FREQ_NOISE and len(w) > 2]
    return Counter(all_words).most_common(top_n)


def word_freq_all_sentiments(df: pd.DataFrame, top_n: int = 20) -> dict:
    return {s: top_words(df, s, top_n) for s in ["Positive", "Neutral", "Negative"]}


# ──────────────────────────────────────────────
# 3. Product-level insights  (with filter modes)
# ──────────────────────────────────────────────


def product_sentiment_summary(
    df: pd.DataFrame,
    top_n: int = 10,
    filter_by: str = "review_count",
) -> pd.DataFrame:
    """
    filter_by options:
      "review_count" → most-reviewed (default)
      "positive"     → highest % positive
      "negative"     → highest % negative
      "neutral"      → most balanced spread
      "best"         → high positive * log(volume)
      "worst"        → high negative * log(volume)
    """
    grouped = (
        df.groupby("ProductId")["Sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .mul(100)
        .round(1)
    )
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in grouped.columns:
            grouped[col] = 0.0

    counts = df.groupby("ProductId")["Sentiment"].count().rename("review_count")
    result = grouped.join(counts)
    result = result[result["review_count"] >= 3]

    def _label(row):
        if row["Positive"] >= 80:
            return "🏆 Best"
        elif row["Negative"] >= 30:
            return "🚨 Worst"
        elif row["Neutral"] >= 40:
            return "⚖️ Balanced"
        elif row["Negative"] >= 15:
            return "⚠️ Risky"
        else:
            return "✅ Good"

    result["performance_label"] = result.apply(_label, axis=1)
    result["dominant_sentiment"] = result[["Positive", "Neutral", "Negative"]].idxmax(
        axis=1
    )

    if filter_by == "positive":
        result = result.sort_values("Positive", ascending=False)
    elif filter_by == "negative":
        result = result.sort_values("Negative", ascending=False)
    elif filter_by == "neutral":
        result["_bal"] = (
            (result["Positive"] - 33).abs()
            + (result["Neutral"] - 33).abs()
            + (result["Negative"] - 33).abs()
        )
        result = result.sort_values("_bal").drop(columns="_bal")
    elif filter_by == "best":
        result["_sc"] = result["Positive"] * np.log1p(result["review_count"])
        result = result.sort_values("_sc", ascending=False).drop(columns="_sc")
    elif filter_by == "worst":
        result["_sc"] = result["Negative"] * np.log1p(result["review_count"])
        result = result.sort_values("_sc", ascending=False).drop(columns="_sc")
    else:
        result = result.sort_values("review_count", ascending=False)

    return result.head(top_n).reset_index()[
        [
            "ProductId",
            "review_count",
            "Positive",
            "Neutral",
            "Negative",
            "dominant_sentiment",
            "performance_label",
        ]
    ]


# ──────────────────────────────────────────────
# 4. Aspect-Based Sentiment  (fixed keyword logic)
# ──────────────────────────────────────────────

ASPECT_MAP = {
    "Taste": (
        {
            "delicious",
            "yummy",
            "tasty",
            "amazing",
            "great",
            "good",
            "love",
            "best",
            "wonderful",
            "excellent",
            "perfect",
            "sweet",
            "savory",
            "flavorful",
            "rich",
            "smooth",
            "fresh",
            "nice",
            "fantastic",
        },
        {
            "bland",
            "awful",
            "terrible",
            "disgusting",
            "bitter",
            "sour",
            "nasty",
            "horrible",
            "bad",
            "worst",
            "fake",
            "artificial",
            "stale",
            "gross",
            "yuck",
            "muddy",
        },
    ),
    "Quality": (
        {
            "quality",
            "excellent",
            "premium",
            "solid",
            "durable",
            "sturdy",
            "perfect",
            "fresh",
            "superior",
            "top",
            "high",
            "great",
            "good",
        },
        {
            "poor",
            "cheap",
            "flimsy",
            "broke",
            "broken",
            "defective",
            "terrible",
            "awful",
            "bad",
            "worst",
            "stale",
            "rotten",
            "expired",
            "low",
            "subpar",
        },
    ),
    "Packaging": (
        {
            "nice",
            "secure",
            "sealed",
            "intact",
            "protected",
            "sturdy",
            "safe",
            "well",
            "good",
            "perfect",
        },
        {
            "damaged",
            "broken",
            "crushed",
            "leaked",
            "torn",
            "open",
            "terrible",
            "awful",
            "bad",
            "poor",
            "messy",
            "dented",
        },
    ),
    "Price": (
        {
            "cheap",
            "affordable",
            "reasonable",
            "worth",
            "value",
            "deal",
            "inexpensive",
            "great",
            "good",
            "fair",
            "excellent",
            "less",
        },
        {
            "expensive",
            "overpriced",
            "costly",
            "pricey",
            "outrageous",
            "ridiculous",
            "waste",
            "rip",
            "much",
        },
    ),
    "Shipping": (
        {
            "fast",
            "quick",
            "speedy",
            "early",
            "prompt",
            "great",
            "excellent",
            "perfect",
            "arrived",
            "received",
        },
        {
            "slow",
            "late",
            "delayed",
            "lost",
            "missing",
            "damaged",
            "terrible",
            "awful",
            "bad",
            "never",
        },
    ),
    "Service": (
        {
            "helpful",
            "friendly",
            "responsive",
            "great",
            "good",
            "excellent",
            "professional",
            "kind",
            "polite",
        },
        {
            "rude",
            "unhelpful",
            "slow",
            "terrible",
            "awful",
            "bad",
            "ignored",
            "unprofessional",
            "horrible",
        },
    ),
}

_NEGATIONS = {
    "not",
    "no",
    "never",
    "isnt",
    "wasnt",
    "dont",
    "doesnt",
    "didnt",
    "cannot",
    "cant",
    "wont",
    "hardly",
    "barely",
}


def _tokenize(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())


def _check_negation(tokens: list, idx: int, window: int = 3) -> bool:
    start = max(0, idx - window)
    return any(t in _NEGATIONS for t in tokens[start:idx])


def aspect_sentiment(text: str) -> dict:
    """
    Aspect-level sentiment with per-aspect keyword sets + negation detection.

    "The taste is amazing but the packaging arrived damaged and it was overpriced"
    → Taste: Positive, Packaging: Negative, Price: Negative
    """
    tokens = _tokenize(text)
    results = {}

    for aspect, (pos_kws, neg_kws) in ASPECT_MAP.items():
        pos_hits = 0
        neg_hits = 0

        for i, token in enumerate(tokens):
            negated = _check_negation(tokens, i)
            if token in pos_kws:
                neg_hits += 1 if negated else 0
                pos_hits += 0 if negated else 1
            elif token in neg_kws:
                pos_hits += 1 if negated else 0
                neg_hits += 0 if negated else 1

        if pos_hits == 0 and neg_hits == 0:
            continue

        if pos_hits > neg_hits:
            results[aspect] = "Positive"
        elif neg_hits > pos_hits:
            results[aspect] = "Negative"
        else:
            results[aspect] = "Neutral"

    return results


# ──────────────────────────────────────────────
# 5. Summary stats
# ──────────────────────────────────────────────


def summary_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    dist = sentiment_distribution(df)
    avg_score = round(df["Score"].mean(), 2) if "Score" in df.columns else None
    unique_products = df["ProductId"].nunique() if "ProductId" in df.columns else None
    return {
        "total_reviews": total,
        "unique_products": unique_products,
        "average_score": avg_score,
        "sentiment_distribution": dist,
    }
