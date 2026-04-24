"""
sentiment_model.py
------------------
Inference wrapper for the trained ensemble model.

Handles:
  - Loading dual vectorizers (word + char) from pickle
  - Transforming single reviews at inference time
  - Score-based rule override for high-confidence predictions
  - Returning structured prediction dicts for app.py
"""

import pickle
import math
from pathlib import Path

from src.preprocessing import clean_text
from src.feature_extraction import transform_tfidf, add_score_feature
from src.train_model import score_to_sentiment

MODEL_PATH = Path("models/sentiment_model.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")


class SentimentModel:
    """
    Inference wrapper for the trained ensemble.

    Usage:
        model = SentimentModel()
        result = model.predict_with_confidence("This product is amazing!")
        # → {"label": "Positive", "confidence": 0.94, "scores": {...}}
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        vectorizer_path=VECTORIZER_PATH,
    ):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizers = pickle.load(f)

        self._has_proba = hasattr(self.model, "predict_proba")

    # ── Internal helpers ─────────────────────────────────────────────────

    def _vectorize(self, texts: list, scores: list = None):
        """Clean → TF-IDF → optionally append score feature."""
        cleaned = [clean_text(t) for t in texts]

        # Handle both dict vectorizers (new) and single vectorizer (legacy)
        if isinstance(self.vectorizers, dict):
            X = transform_tfidf(cleaned, self.vectorizers)
        else:
            X = self.vectorizers.transform(cleaned)

        if scores is not None:
            X = add_score_feature(X, scores, fit=False)

        return X

    def _softmax(self, values: list) -> list:
        exp_v = [math.exp(v) for v in values]
        total = sum(exp_v)
        return [e / total for e in exp_v]

    def _get_scores(self, X) -> tuple:
        """Returns (classes, score_dict) using proba or decision_function."""
        classes = list(self.model.classes_)

        if self._has_proba:
            proba = self.model.predict_proba(X)[0]
            scores = dict(zip(classes, [round(float(p), 4) for p in proba]))
        else:
            decision = self.model.decision_function(X)[0]
            if hasattr(decision, "__iter__"):
                softmax = self._softmax([float(d) for d in decision])
            else:
                softmax = [1.0]
                classes = [self.model.predict(X)[0]]
            scores = dict(zip(classes, [round(s, 4) for s in softmax]))

        return classes, scores

    # ── Public API ───────────────────────────────────────────────────────

    def predict(self, texts: list, scores: list = None) -> list:
        """Batch predict — returns list of label strings."""
        X = self._vectorize(texts, scores)
        return list(self.model.predict(X))

    def predict_with_confidence(
        self,
        text: str,
        star_rating: float = None,
        use_score_rule: bool = True,
    ) -> dict:
        """
        Predict sentiment for a single review with confidence scores.

        Parameters
        ----------
        text           : raw (uncleaned) review text
        star_rating    : optional 1-5 star rating
                         If provided AND use_score_rule=True AND rating is
                         1, 2, 4, or 5 → score-based rule overrides NLP prediction
                         (these are near-certain cases)
        use_score_rule : whether to allow score override (default True)

        Returns
        -------
        dict with:
          label      : "Positive" | "Neutral" | "Negative"
          confidence : float 0-1
          scores     : {class: probability} for all classes
          method     : "rule" | "model" (how the prediction was made)
        """
        # ── Score-based rule (high confidence shortcut) ──────────────────
        if use_score_rule and star_rating is not None:
            if star_rating in [1, 2, 4, 5]:
                rule_label = score_to_sentiment(star_rating)
                # Still run the model to get scores for display
                score_list = [star_rating] if star_rating else None
                X = self._vectorize([text], score_list)
                _, model_scores = self._get_scores(X)

                # Boost the rule-predicted class to 0.95 confidence
                adjusted = {k: 0.017 for k in model_scores}
                adjusted[rule_label] = 0.95
                # Distribute remaining 0.05 to others
                others = [k for k in adjusted if k != rule_label]
                if others:
                    per_other = 0.05 / len(others)
                    for k in others:
                        adjusted[k] = round(per_other, 4)
                adjusted[rule_label] = round(0.95, 4)

                return {
                    "label": rule_label,
                    "confidence": 0.95,
                    "scores": adjusted,
                    "method": "rule",
                }

        # ── NLP model prediction ─────────────────────────────────────────
        score_list = [star_rating] if star_rating is not None else None
        X = self._vectorize([text], score_list)
        label = self.model.predict(X)[0]
        _, scores = self._get_scores(X)
        confidence = scores.get(label, max(scores.values()))

        return {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "method": "model",
        }
