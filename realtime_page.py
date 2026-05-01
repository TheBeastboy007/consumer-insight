"""
realtime_page.py
=================
Streamlit page: Real-Time Amazon Review Analysis
Plug this into app.py as Page 4.

HOW TO ADD TO app.py:
----------------------
1. Add "Real-Time Analysis" to your sidebar radio options
2. Import and call render_realtime_page() when that option is selected

Example in app.py:
    from realtime_page import render_realtime_page
    ...
    page = st.sidebar.radio("Navigation", [
        "Predict Sentiment",
        "Dataset Analytics",
        "Product Insights",
        "Real-Time Analysis"       # <-- add this
    ])
    ...
    elif page == "Real-Time Analysis":
        render_realtime_page(model, vectorizers, meta)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

from realtime_fetcher import (
    fetch_product_reviews,
    fetch_product_search,
    reviews_to_dataframe,
    is_api_configured,
    RAPIDAPI_KEY,
)

# ── Sentiment colors (match existing app theme) ──────────────────────────────
SENT_COLORS = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}


def _predict_reviews(df: pd.DataFrame, model, vectorizers, meta=None) -> pd.DataFrame:
    """Run the trained model on a dataframe of reviews."""
    from src.preprocessing import clean_text
    from src.feature_extraction import transform_tfidf, add_score_feature

    meta = meta or {}

    results = []
    for _, row in df.iterrows():
        text = str(row.get("Text", ""))
        if not text.strip():
            results.append(
                {
                    "Predicted": "Unknown",
                    "Confidence": 0.0,
                    "Pos_Prob": 0.0,
                    "Neu_Prob": 0.0,
                    "Neg_Prob": 0.0,
                }
            )
            continue

        try:
            cleaned = clean_text(text)
            if isinstance(vectorizers, dict):
                X = transform_tfidf([cleaned], vectorizers)
            else:
                X = vectorizers.transform([cleaned])

            # Keep feature shape aligned with models trained with score feature.
            if meta.get("use_score_feature", False):
                score = row.get("Score", 3)
                try:
                    score = float(score)
                except Exception:
                    score = 3.0
                X = add_score_feature(X, [score])

            labels = list(getattr(model, "classes_", ["Positive", "Neutral", "Negative"]))
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                prob_dict = dict(zip(labels, [float(p) for p in probs]))
            elif hasattr(model, "decision_function"):
                # Convert decision scores to pseudo-probabilities via softmax.
                import math

                decision = model.decision_function(X)[0]
                vals = (
                    [float(v) for v in decision]
                    if hasattr(decision, "__iter__")
                    else [float(decision)]
                )
                exps = [math.exp(v) for v in vals]
                total = sum(exps) if sum(exps) else 1.0
                probs = [e / total for e in exps]
                prob_dict = dict(zip(labels, probs))
            else:
                predicted_label = model.predict(X)[0]
                prob_dict = {
                    c: (1.0 if c == predicted_label else 0.0)
                    for c in ["Positive", "Neutral", "Negative"]
                }

            pos = prob_dict.get("Positive", 0.0)
            neu = prob_dict.get("Neutral", 0.0)
            neg = prob_dict.get("Negative", 0.0)

            predicted = max(prob_dict, key=prob_dict.get)
            confidence = max(prob_dict.values()) * 100

            results.append(
                {
                    "Predicted": predicted,
                    "Confidence": round(confidence, 1),
                    "Pos_Prob": round(pos * 100, 1),
                    "Neu_Prob": round(neu * 100, 1),
                    "Neg_Prob": round(neg * 100, 1),
                }
            )
        except Exception:
            results.append(
                {
                    "Predicted": "Error",
                    "Confidence": 0.0,
                    "Pos_Prob": 0.0,
                    "Neu_Prob": 0.0,
                    "Neg_Prob": 0.0,
                }
            )

    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


def _sentiment_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Predicted"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    colors = [SENT_COLORS.get(s, "#999") for s in counts["Sentiment"]]
    fig = go.Figure(
        go.Pie(
            labels=counts["Sentiment"],
            values=counts["Count"],
            hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
    )
    return fig


def _confidence_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Confidence",
        color="Predicted",
        color_discrete_map=SENT_COLORS,
        nbins=20,
        barmode="overlay",
        labels={"Confidence": "Confidence (%)", "count": "Reviews"},
    )
    fig.update_layout(
        height=250,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
    )
    return fig


def _star_vs_predicted_bar(df: pd.DataFrame) -> go.Figure:
    """Show how the model prediction compares to star rating label."""
    if "Score" not in df.columns:
        return None
    cross = pd.crosstab(df["Score"], df["Predicted"])
    fig = go.Figure()
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in cross.columns:
            fig.add_trace(
                go.Bar(
                    x=cross.index,
                    y=cross[sentiment],
                    name=sentiment,
                    marker_color=SENT_COLORS[sentiment],
                )
            )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Star Rating",
        yaxis_title="Review Count",
        height=250,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def render_realtime_page(model, vectorizers, meta):
    """Main render function — call from app.py."""

    st.markdown("## Real-Time Amazon Review Analysis")
    st.markdown(
        "Fetch live reviews from Amazon and instantly predict sentiment using the trained ensemble model."
    )

    # ── API KEY CHECK ──────────────────────────────────────────────────────
    if not is_api_configured():
        st.error("⚠️ RapidAPI key not configured.")
        st.markdown(
            """
**To enable real-time fetching:**

1. Go to [rapidapi.com](https://rapidapi.com) and create a free account
2. Search for **"Real-Time Amazon Data"**
3. Click **Subscribe** → choose the **FREE** plan (500 req/month)
4. Copy your API key from the dashboard
5. Open `realtime_fetcher.py` and paste it into:
```python
RAPIDAPI_KEY = "paste_your_key_here"
```
6. Restart the Streamlit app
        """
        )
        st.info(
            "💡 While you set up the API, you can use the **Demo Mode** below to see how the page works."
        )
        _demo_mode(model, vectorizers, meta)
        return

    # ── TABS ───────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔍 Search by Keyword", "📦 Enter ASIN Directly"])

    with tab1:
        _search_tab(model, vectorizers, meta)

    with tab2:
        _asin_tab(model, vectorizers, meta)


def _search_tab(model, vectorizers, meta):
    """Search Amazon by keyword → pick product → fetch reviews → predict."""
    col1, col2 = st.columns([3, 1])
    with col1:
        keyword = st.text_input(
            "Search Amazon products",
            placeholder="e.g. coffee beans, dog treats, protein powder",
        )
    with col2:
        country = st.selectbox("Market", ["US", "IN", "GB", "CA", "AU"], index=0)

    if st.button("🔍 Search Products", key="search_btn"):
        if not keyword.strip():
            st.warning("Enter a search keyword.")
            return

        with st.spinner("Searching Amazon..."):
            products = fetch_product_search(keyword, country)

        if not products:
            st.error("No products found or API error. Try a different keyword.")
            return

        st.session_state["search_products"] = products
        st.session_state["search_country"] = country

    if "search_products" in st.session_state:
        products = st.session_state["search_products"]
        country = st.session_state.get("search_country", "US")

        options = {
            f"{p['title']} ⭐{p['rating']} ({p['num_ratings']} ratings)": p["asin"]
            for p in products
        }
        selected_label = st.selectbox("Select a product", list(options.keys()))
        selected_asin = options[selected_label]

        n_pages = st.slider("Pages of reviews to fetch (10 reviews per page)", 1, 5, 1)

        if st.button("📥 Fetch & Analyse Reviews", key="fetch_search_btn"):
            _fetch_and_display(selected_asin, n_pages, country, model, vectorizers, meta)


def _asin_tab(model, vectorizers, meta):
    """Directly enter an Amazon ASIN and fetch reviews."""
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        asin = st.text_input(
            "Amazon ASIN",
            placeholder="e.g. B08N5WRWNW",
            help="The ASIN is in the Amazon product URL: amazon.com/dp/ASIN",
        )
    with col2:
        country = st.selectbox(
            "Market", ["US", "IN", "GB", "CA", "AU"], index=0, key="asin_country"
        )
    with col3:
        n_pages = st.slider("Pages", 1, 5, 1, key="asin_pages")

    if st.button("📥 Fetch & Analyse Reviews", key="fetch_asin_btn"):
        if not asin.strip():
            st.warning("Enter an ASIN.")
            return
        _fetch_and_display(asin.strip().upper(), n_pages, country, model, vectorizers, meta)


def _fetch_and_display(asin: str, n_pages: int, country: str, model, vectorizers, meta):
    """Core: fetch reviews → predict → display results."""

    all_reviews = []
    progress = st.progress(0, text="Fetching reviews...")

    for page in range(1, n_pages + 1):
        progress.progress(page / n_pages, text=f"Fetching page {page}/{n_pages}...")
        result = fetch_product_reviews(asin, page=page, country=country)

        if not result["success"]:
            st.error(f"API Error: {result['error']}")
            break

        all_reviews.extend(result["reviews"])
        product_title = result["product_title"]
        time.sleep(0.5)  # be polite to the API

    progress.empty()

    if not all_reviews:
        st.warning(
            "No reviews fetched. The product may have no reviews or the ASIN is invalid."
        )
        return

    # ── Build dataframe and run predictions ───────────────────────────────
    df_raw = reviews_to_dataframe(all_reviews)

    with st.spinner(f"Running sentiment analysis on {len(df_raw)} reviews..."):
        df = _predict_reviews(df_raw, model, vectorizers, meta)

    # ── Store in session for export ────────────────────────────────────────
    st.session_state["rt_results"] = df
    st.session_state["rt_product"] = product_title
    st.session_state["rt_asin"] = asin

    # ── HEADER ─────────────────────────────────────────────────────────────
    st.markdown(f"### 📦 {product_title}")
    st.caption(
        f"ASIN: `{asin}` | Fetched: {datetime.now().strftime('%H:%M:%S')} | Reviews: {len(df)}"
    )

    # ── KPI CARDS ─────────────────────────────────────────────────────────
    pos_pct = (df["Predicted"] == "Positive").mean() * 100
    neu_pct = (df["Predicted"] == "Neutral").mean() * 100
    neg_pct = (df["Predicted"] == "Negative").mean() * 100
    avg_conf = df["Confidence"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🟢 Positive", f"{pos_pct:.1f}%")
    k2.metric("🟡 Neutral", f"{neu_pct:.1f}%")
    k3.metric("🔴 Negative", f"{neg_pct:.1f}%")
    k4.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

    # ── PERFORMANCE LABEL ─────────────────────────────────────────────────
    if pos_pct >= 80:
        label, color = "🏆 Best", "green"
    elif neg_pct >= 30:
        label, color = "❌ Worst", "red"
    elif neg_pct >= 15:
        label, color = "⚠️ Risky", "orange"
    elif neu_pct >= 40:
        label, color = "⚖️ Balanced", "blue"
    else:
        label, color = "✅ Good", "green"

    st.markdown(f"**Product Health:** :{color}[{label}]")

    # ── CHARTS ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sentiment Distribution**")
        st.plotly_chart(_sentiment_donut(df), use_container_width=True)
    with c2:
        st.markdown("**Model Confidence**")
        st.plotly_chart(_confidence_histogram(df), use_container_width=True)
    with c3:
        fig3 = _star_vs_predicted_bar(df)
        if fig3:
            st.markdown("**Stars vs Predicted**")
            st.plotly_chart(fig3, use_container_width=True)

    # ── REVIEW TABLE ──────────────────────────────────────────────────────
    st.markdown("### 📋 Review-Level Results")

    # Color coding
    def color_sentiment(val):
        colors = {"Positive": "#1a3a1a", "Neutral": "#3a3a1a", "Negative": "#3a1a1a"}
        return f"background-color: {colors.get(val, '')}"

    display_cols = [
        "Title",
        "Score",
        "Text",
        "Predicted",
        "Confidence",
        "Pos_Prob",
        "Neu_Prob",
        "Neg_Prob",
        "Date",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    styled = df[display_cols].style.map(color_sentiment, subset=["Predicted"])
    st.dataframe(styled, use_container_width=True, height=400)

    # ── EXPORT ────────────────────────────────────────────────────────────
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results as CSV",
        csv,
        file_name=f"realtime_{asin}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # ── DISAGREEMENT ANALYSIS ─────────────────────────────────────────────
    if "Sentiment" in df.columns and "Predicted" in df.columns:
        disagreements = df[df["Sentiment"] != df["Predicted"]]
        if len(disagreements) > 0:
            agree_rate = (1 - len(disagreements) / len(df)) * 100
            st.markdown(f"### 🔎 Model vs Star Rating Agreement: **{agree_rate:.1f}%**")
            st.caption(
                f"{len(disagreements)} reviews where model prediction differs from star-rating label. "
                "This is expected — text often contradicts the star rating."
            )
            with st.expander(f"View {len(disagreements)} disagreements"):
                dis_cols = ["Text", "Score", "Sentiment", "Predicted", "Confidence"]
                dis_cols = [c for c in dis_cols if c in disagreements.columns]
                st.dataframe(disagreements[dis_cols], use_container_width=True)


def _demo_mode(model, vectorizers, meta):
    """Show the page working with sample reviews when API is not configured."""
    st.markdown("---")
    st.markdown("### 🧪 Demo Mode — Sample Reviews")
    st.info(
        "These are sample reviews to demonstrate the real-time analysis page layout."
    )

    sample_reviews = [
        {
            "ReviewId": "R1",
            "ProductId": "DEMO",
            "Text": "Absolutely love this product, best purchase ever!",
            "Score": 5,
            "Title": "Amazing!",
            "Verified": True,
            "Date": "2025-01-01",
            "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "ReviewId": "R2",
            "ProductId": "DEMO",
            "Text": "Terrible quality, broke after one use. Total waste of money.",
            "Score": 1,
            "Title": "Awful",
            "Verified": True,
            "Date": "2025-01-02",
            "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "ReviewId": "R3",
            "ProductId": "DEMO",
            "Text": "It is okay, nothing special. Does the job but nothing more.",
            "Score": 3,
            "Title": "Average",
            "Verified": False,
            "Date": "2025-01-03",
            "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "ReviewId": "R4",
            "ProductId": "DEMO",
            "Text": "Great taste but the packaging arrived completely damaged.",
            "Score": 3,
            "Title": "Mixed",
            "Verified": True,
            "Date": "2025-01-04",
            "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "ReviewId": "R5",
            "ProductId": "DEMO",
            "Text": "Outstanding flavor and great value for money. Highly recommend!",
            "Score": 5,
            "Title": "Excellent",
            "Verified": True,
            "Date": "2025-01-05",
            "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "ReviewId": "R6",
            "ProductId": "DEMO",
            "Text": "Not worth the price at all. Very disappointing product.",
            "Score": 2,
            "Title": "Disappointed",
            "Verified": False,
            "Date": "2025-01-06",
            "FetchedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    ]

    df_raw = reviews_to_dataframe(sample_reviews)

    if model is not None and vectorizers is not None:
        with st.spinner("Running predictions on demo reviews..."):
            df = _predict_reviews(df_raw, model, vectorizers, meta)
        _display_demo_results(df)
    else:
        st.warning(
            "Model not loaded. Start the app normally with trained model files to see predictions."
        )


def _display_demo_results(df: pd.DataFrame):
    pos_pct = (df["Predicted"] == "Positive").mean() * 100
    neu_pct = (df["Predicted"] == "Neutral").mean() * 100
    neg_pct = (df["Predicted"] == "Negative").mean() * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("🟢 Positive", f"{pos_pct:.1f}%")
    k2.metric("🟡 Neutral", f"{neu_pct:.1f}%")
    k3.metric("🔴 Negative", f"{neg_pct:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_sentiment_donut(df), use_container_width=True)
    with c2:
        st.plotly_chart(_confidence_histogram(df), use_container_width=True)

    display_cols = ["Title", "Score", "Text", "Predicted", "Confidence"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)
