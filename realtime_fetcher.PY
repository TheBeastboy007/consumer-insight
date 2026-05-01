"""
realtime_fetcher.py
====================
Fetches live Amazon product reviews using RapidAPI.
Used by the Real-Time page in app.py.

SETUP:
1. Go to https://rapidapi.com
2. Search "Real-Time Amazon Data" by Axesso or "Amazon Product Reviews"
3. Subscribe to the FREE tier (500 requests/month free)
4. Copy your RapidAPI key
5. Paste it in RAPIDAPI_KEY below OR set it as environment variable
"""

import requests
import pandas as pd
from datetime import datetime
import time

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAPIDAPI_KEY = (
    "181a7ba538mshd0c0e2d50fe791ap11418ajsn156a94455dd8"  # <-- paste your key here
)
RAPIDAPI_HOST = "real-time-amazon-data.p.rapidapi.com"
BASE_URL = "https://real-time-amazon-data.p.rapidapi.com"
# ────────────────────────────────────────────────────────────────────────────


def fetch_product_reviews(asin: str, page: int = 1, country: str = "US") -> dict:
    """
    Fetch reviews for an Amazon product by ASIN.

    Parameters
    ----------
    asin    : Amazon product ID e.g. 'B08N5WRWNW'
    page    : page number (1-10)
    country : marketplace country code (US, IN, GB, etc.)

    Returns
    -------
    dict with keys: reviews (list), total_reviews (int), product_title (str)
    """
    url = f"{BASE_URL}/product-reviews"
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}
    params = {
        "asin": asin,
        "country": country,
        "page": str(page),
        "sort_by": "TOP_REVIEWS",
        "star_rating": "ALL",
        "verified_purchases_only": "false",
        "images_or_videos_only": "false",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        reviews = []
        raw_reviews = data.get("data", {}).get("reviews", [])
        total_reviews = data.get("data", {}).get("total_reviews", 0)

        # API returns title under different keys depending on endpoint version
        data_block = data.get("data", {})
        product_title = (
            data_block.get("product_title")
            or data_block.get("title")
            or data_block.get("product_name")
            or data_block.get("name")
            or data.get("product_title")
            or data.get("title")
            or "Unknown Product"
        )

        for r in raw_reviews:
            reviews.append(
                {
                    "ReviewId": r.get("review_id", ""),
                    "ProductId": asin,
                    "Text": r.get("review_comment", ""),
                    "Score": int(r.get("review_star_rating", 3)),
                    "Title": r.get("review_title", ""),
                    "Verified": r.get("is_verified_purchase", False),
                    "Date": r.get("review_date", ""),
                    "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return {
            "reviews": reviews,
            "total_reviews": total_reviews,
            "product_title": product_title,
            "asin": asin,
            "success": True,
            "error": None,
        }

    except requests.exceptions.Timeout:
        return _error_response(asin, "Request timed out. Try again.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return _error_response(asin, "Rate limit reached. Wait 1 minute.")
        elif e.response.status_code == 403:
            return _error_response(asin, "Invalid API key. Check RAPIDAPI_KEY.")
        return _error_response(asin, f"HTTP error: {e.response.status_code}")
    except Exception as e:
        return _error_response(asin, str(e))


def fetch_product_search(keyword: str, country: str = "US") -> list:
    """
    Search Amazon products by keyword and return list of ASINs + titles.
    Useful so the user can pick a product without knowing the ASIN.
    """
    url = f"{BASE_URL}/search"
    headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}
    params = {"query": keyword, "country": country, "category_id": "aps", "page": "1"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        products = []
        for p in data.get("data", {}).get("products", [])[:10]:
            asin = p.get("asin", "")
            title = p.get("product_title", "Unknown")
            rating = p.get("product_star_rating", "N/A")
            num_ratings = p.get("product_num_ratings", 0)
            if asin:
                products.append(
                    {
                        "asin": asin,
                        "title": title[:80] + "..." if len(title) > 80 else title,
                        "rating": rating,
                        "num_ratings": num_ratings,
                    }
                )
        return products

    except Exception:
        return []


def reviews_to_dataframe(reviews: list) -> pd.DataFrame:
    """Convert raw review list to a DataFrame with Sentiment label derived from Score."""
    if not reviews:
        return pd.DataFrame()

    df = pd.DataFrame(reviews)

    # Derive sentiment label from star rating (same scheme as training data)
    def score_to_sentiment(score):
        if score >= 4:
            return "Positive"
        elif score == 3:
            return "Neutral"
        else:
            return "Negative"

    df["Sentiment"] = df["Score"].apply(score_to_sentiment)
    df["Cleaned_Text"] = ""  # will be filled by preprocessing in app
    return df


def _error_response(asin: str, message: str) -> dict:
    return {
        "reviews": [],
        "total_reviews": 0,
        "product_title": "Unknown",
        "asin": asin,
        "success": False,
        "error": message,
    }


def is_api_configured() -> bool:
    """Check if the API key has been set by the user."""
    return RAPIDAPI_KEY != "YOUR_RAPIDAPI_KEY_HERE" and len(RAPIDAPI_KEY) > 20
