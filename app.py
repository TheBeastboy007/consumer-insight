"""
app.py  --  AI-Driven Consumer Insight System  (v4 -- Claude-inspired UI)
========================================================================
Pages:
  1. Predict Sentiment   -- review + optional star rating -> ensemble prediction
  2. Dataset Analytics   -- upload CSV OR use default dataset
  3. Product Insights    -- 6 filter modes with KPI cards
  4. Real-Time Analysis  -- live Amazon reviews via RapidAPI
"""

import io
import json
import math
import pickle
import pathlib

import pandas as pd
import scipy.sparse as sp
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from realtime_page import render_realtime_page
from src.preprocessing import clean_text
from src.feature_extraction import transform_tfidf, add_score_feature
from src.train_model import score_to_sentiment
from src.insights import (
    sentiment_distribution,
    word_freq_all_sentiments,
    product_sentiment_summary,
    aspect_sentiment,
    summary_stats,
)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Consumer Insight AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-base:       #1a1915;
    --bg-surface:    #211f1b;
    --bg-raised:     #2a2824;
    --bg-overlay:    #332f2a;
    --border-subtle: #3d3a34;
    --border-mid:    #4d4944;
    --text-primary:  #e8e4dc;
    --text-secondary:#a8a49c;
    --text-muted:    #6b6762;
    --accent-warm:   #d97757;
    --accent-amber:  #d4a853;
    --accent-green:  #7bc47a;
    --accent-red:    #d46b6b;
    --accent-blue:   #7aacd4;
    --accent-purple: #a87ad4;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg-base);
    color: var(--text-primary);
}
.stApp { background-color: var(--bg-base); }

section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

div[data-testid="metric-container"] {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: var(--border-mid); }
div[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px !important;
    color: var(--text-primary) !important;
    font-weight: 500;
}

.stTextArea textarea, .stTextInput input {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.2s;
    caret-color: var(--accent-warm);
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-warm) !important;
    box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.12) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: var(--text-muted) !important;
}

.stSelectbox > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-raised);
    border: 1px dashed var(--border-mid);
    border-radius: var(--radius-md);
    padding: 1rem;
}

.stButton > button {
    background: var(--bg-overlay) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    background: var(--bg-overlay) !important;
    border-color: var(--accent-warm) !important;
    color: var(--accent-warm) !important;
}
.stButton > button:active { transform: translateY(1px) !important; }

.primary-btn > button {
    background: var(--accent-warm) !important;
    color: #1a1915 !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}
.primary-btn > button:hover {
    background: #c96a4a !important;
    color: #1a1915 !important;
    border: none !important;
    opacity: 0.92;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border-subtle);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 20px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-warm) !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-warm) !important;
}

hr { border-color: var(--border-subtle) !important; margin: 1.5rem 0; }

.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
    margin-top: 4px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 20px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 16px;
    letter-spacing: 0.01em;
}
.badge-Positive { background: rgba(123,196,122,0.12); color: var(--accent-green); border: 1px solid rgba(123,196,122,0.3); }
.badge-Negative { background: rgba(212,107,107,0.12); color: var(--accent-red);   border: 1px solid rgba(212,107,107,0.3); }
.badge-Neutral  { background: rgba(212,168,83,0.12);  color: var(--accent-amber); border: 1px solid rgba(212,168,83,0.3);  }

.method-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    background: var(--bg-overlay);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    margin-top: 6px;
    letter-spacing: 0.04em;
}

.aspect-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.asp-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px 18px;
    min-width: 120px;
    text-align: center;
    transition: border-color 0.2s;
}
.asp-card:hover { border-color: var(--border-mid); }
.asp-name {
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.asp-Positive { color: var(--accent-green); font-weight: 600; font-size: 14px; }
.asp-Negative { color: var(--accent-red);   font-weight: 600; font-size: 14px; }
.asp-Neutral  { color: var(--accent-amber); font-weight: 600; font-size: 14px; }

.callout {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-warm);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.6;
}

.kpi-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--border-mid); }

.acc-banner {
    background: var(--bg-raised);
    border: 1px solid rgba(123,196,122,0.25);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin: 12px 0 16px 0;
}

.filter-active {
    display: inline-block;
    background: rgba(217,119,87,0.12);
    border: 1px solid rgba(217,119,87,0.3);
    color: var(--accent-warm);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
}

.page-title {
    font-family: 'Instrument Serif', serif;
    font-size: 32px;
    font-weight: 400;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 4px;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 24px;
}

.wordmark {
    font-family: 'Instrument Serif', serif;
    font-size: 20px;
    font-weight: 400;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.wordmark span { color: var(--accent-warm); }

[data-testid="stDataFrame"] { border-radius: var(--radius-md); overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Plot theme
# -----------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#a8a49c", size=12),
    margin=dict(t=20, b=20, l=10, r=10),
    xaxis=dict(gridcolor="#2a2824", linecolor="#3d3a34", tickcolor="#3d3a34"),
    yaxis=dict(gridcolor="#2a2824", linecolor="#3d3a34", tickcolor="#3d3a34"),
)

SENTIMENT_COLORS = {
    "Positive": "#7bc47a",
    "Neutral": "#d4a853",
    "Negative": "#d46b6b",
}

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    model = pickle.load(open("models/sentiment_model.pkl", "rb"))
    vectorizers = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    meta = {}
    if pathlib.Path("models/model_meta.json").exists():
        with open("models/model_meta.json") as f:
            meta = json.load(f)
    return model, vectorizers, meta


@st.cache_data(show_spinner="Loading dataset ...")
def load_default_data():
    for path in [
        "data/preprocessed_reviews.csv",
        "data/processed_reviews.csv",
        "data/reviews.csv",
    ]:
        if pathlib.Path(path).exists():
            return pd.read_csv(path)
    return None


def run_prediction(text, star_rating, model, vectorizers, meta):
    cleaned = clean_text(text)
    X = (
        transform_tfidf([cleaned], vectorizers)
        if isinstance(vectorizers, dict)
        else vectorizers.transform([cleaned])
    )

    # If training included score as a feature, append one score column here.
    # Default to 3 (neutral) when user does not provide a rating.
    use_score_feature = bool(meta.get("use_score_feature", False))
    if use_score_feature:
        score_for_feature = star_rating if star_rating is not None else 3
        X = add_score_feature(X, [score_for_feature])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        nlp_prob = dict(zip(classes, [float(p) for p in proba]))
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)[0]
        classes = list(model.classes_)
        vals = [
            float(d)
            for d in (decision if hasattr(decision, "__iter__") else [decision])
        ]
        exp_v = [math.exp(v) for v in vals]
        total = sum(exp_v)
        nlp_prob = dict(zip(classes, [e / total for e in exp_v]))
    else:
        label = model.predict(X)[0]
        nlp_prob = {
            c: (1.0 if c == label else 0.0) for c in ["Positive", "Neutral", "Negative"]
        }

    STAR_WEIGHT = 0.30
    if use_score_feature:
        scores = {k: round(v, 4) for k, v in nlp_prob.items()}
        method = "◆ NLP + score feature"
    elif star_rating is not None:
        star_prior = {
            1: {"Negative": 0.85, "Neutral": 0.10, "Positive": 0.05},
            2: {"Negative": 0.70, "Neutral": 0.20, "Positive": 0.10},
            3: {"Negative": 0.15, "Neutral": 0.70, "Positive": 0.15},
            4: {"Negative": 0.05, "Neutral": 0.15, "Positive": 0.80},
            5: {"Negative": 0.02, "Neutral": 0.08, "Positive": 0.90},
        }.get(int(star_rating), None)

        if star_prior:
            blended = {}
            for cls in ["Positive", "Neutral", "Negative"]:
                blended[cls] = (1 - STAR_WEIGHT) * nlp_prob.get(
                    cls, 0.0
                ) + STAR_WEIGHT * star_prior.get(cls, 0.0)
            total = sum(blended.values())
            scores = {k: round(v / total, 4) for k, v in blended.items()}
            method = f"NLP + {star_rating} star blend"
        else:
            scores = {k: round(v, 4) for k, v in nlp_prob.items()}
            method = "NLP Ensemble"
    else:
        scores = {k: round(v, 4) for k, v in nlp_prob.items()}
        method = "NLP Ensemble"

    label = max(scores, key=scores.get)
    confidence = scores[label]
    return {
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "method": method,
    }


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='wordmark'>Consumer<span>◆</span>Insight</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='height:2px;background:var(--border-subtle);margin:14px 0 18px 0'></div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "◆ Predict Sentiment",
            "◆ Dataset Analytics",
            "◆ Product Insights",
            "◆ Real-Time Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        "<div style='height:2px;background:var(--border-subtle);margin:18px 0'></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("AI-Driven Consumer Insight System")


# -----------------------------------------------------------------------------
# PAGE 1 -- Predict Sentiment
# -----------------------------------------------------------------------------
if page == "◆ Predict Sentiment":
    st.markdown(
        "<div class='page-title'>Predict Sentiment</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='page-subtitle'>Analyse a product review with the trained ensemble model.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='callout'>Enter any product review below. Adding a star rating is optional.</div>",
        unsafe_allow_html=True,
    )

    review_input = st.text_area(
        "Review Text",
        height=140,
        placeholder="e.g. The taste is amazing but the packaging arrived damaged and it was overpriced.",
    )

    col_star, col_spacer = st.columns([2, 3])
    with col_star:
        star_options = {
            "Not provided": None,
            "1 Star": 1,
            "2 Stars": 2,
            "3 Stars": 3,
            "4 Stars": 4,
            "5 Stars": 5,
        }
        star_choice = st.selectbox("Star Rating (optional)", list(star_options.keys()))
        star_rating = star_options[star_choice]

    st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
    run = st.button("Analyse Review", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not review_input.strip():
            st.warning("Please enter a review first.")
        else:
            try:
                model, vectorizers, meta = load_model()
                result = run_prediction(
                    review_input, star_rating, model, vectorizers, meta
                )
                label = result["label"]
                confidence = result["confidence"]
                scores = result["scores"]
                method = result["method"]

                st.markdown("---")
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.markdown(
                        "<div class='section-label'>Prediction</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='badge badge-{label}'>{'✓' if label=='Positive' else '✗' if label=='Negative' else '-'} {label}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='method-badge'>{method}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence:.1%}")

                with c2:
                    score_df = pd.DataFrame(
                        [{"Sentiment": k, "Probability": v} for k, v in scores.items()]
                    )
                    fig = px.bar(
                        score_df,
                        x="Probability",
                        y="Sentiment",
                        orientation="h",
                        color="Sentiment",
                        color_discrete_map=SENTIMENT_COLORS,
                        text=score_df["Probability"].map(lambda x: f"{x:.1%}"),
                    )
                    fig.update_traces(textposition="outside", textfont_size=12)
                    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=180)
                    fig.update_xaxes(range=[0, 1.1], tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown(
                    "<div class='section-label'>Aspect Analysis</div>",
                    unsafe_allow_html=True,
                )
                aspects = aspect_sentiment(review_input)
                if aspects:
                    cards = "".join(
                        f"<div class='asp-card'><div class='asp-name'>{asp}</div><div class='asp-{sent}'>{sent}</div></div>"
                        for asp, sent in aspects.items()
                    )
                    st.markdown(
                        f"<div class='aspect-grid'>{cards}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='callout'>No specific aspects detected -- try mentioning taste, packaging, price, shipping or quality.</div>",
                        unsafe_allow_html=True,
                    )

            except FileNotFoundError:
                st.error("Model not found. Run `python main.py` to train first.")

    st.markdown("---")
    st.caption("Tip: 5-star or 1-star + text = maximum accuracy via score rule.")


# -----------------------------------------------------------------------------
# PAGE 2 -- Dataset Analytics
# -----------------------------------------------------------------------------
elif page == "◆ Dataset Analytics":
    st.markdown(
        "<div class='page-title'>Dataset Analytics</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='page-subtitle'>Explore sentiment patterns across your review dataset.</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV (needs 'Sentiment' column; 'Score' and 'ProductId' optional)",
        type=["csv"],
    )
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows from uploaded file.")
    else:
        df = load_default_data()
        if df is None:
            st.error("No default dataset found. Please upload a CSV.")
            st.stop()
        st.caption(
            "Using default dataset from `data/`. Upload a CSV above to analyse your own data."
        )

    if "Sentiment" not in df.columns:
        st.error(
            "Dataset must have a 'Sentiment' column (Positive / Neutral / Negative)."
        )
        st.stop()

    if "Cleaned_Text" not in df.columns and "Text" in df.columns:
        with st.spinner("Cleaning text ..."):
            df["Cleaned_Text"] = df["Text"].apply(clean_text)
    elif "Cleaned_Text" not in df.columns:
        df["Cleaned_Text"] = ""

    stats = summary_stats(df)
    dist = stats["sentiment_distribution"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{stats['total_reviews']:,}")
    c2.metric(
        "Unique Products",
        f"{stats['unique_products']:,}" if stats["unique_products"] else "--",
    )
    c3.metric(
        "Avg Star Rating", stats["average_score"] if stats["average_score"] else "--"
    )
    top_label = max(dist, key=lambda k: dist[k]["count"]) if dist else "--"
    c4.metric("Dominant Sentiment", top_label)

    st.markdown("---")

    col_pie, col_bar = st.columns(2)
    pie_df = pd.DataFrame(
        [{"Sentiment": k, "Count": v["count"]} for k, v in dist.items()]
    )

    with col_pie:
        st.markdown(
            "<div class='section-label'>Distribution</div>", unsafe_allow_html=True
        )
        fig_pie = px.pie(
            pie_df,
            names="Sentiment",
            values="Count",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.5,
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont_size=12,
            marker=dict(line=dict(color="#1a1915", width=2)),
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.markdown(
            "<div class='section-label'>Count by Class</div>", unsafe_allow_html=True
        )
        fig_bar = px.bar(
            pie_df.sort_values("Count"),
            x="Count",
            y="Sentiment",
            orientation="h",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="Count",
        )
        fig_bar.update_traces(textposition="outside", textfont_size=11)
        fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div class='section-label'>Top Words by Sentiment</div>",
        unsafe_allow_html=True,
    )
    tab_pos, tab_neg, tab_neu = st.tabs(["Positive", "Negative", "Neutral"])
    word_data = word_freq_all_sentiments(df, top_n=20)

    for tab, sk in zip(
        [tab_pos, tab_neg, tab_neu], ["Positive", "Negative", "Neutral"]
    ):
        with tab:
            wf = word_data.get(sk, [])
            if wf:
                wf_df = pd.DataFrame(wf, columns=["Word", "Count"])
                fig_wf = px.bar(
                    wf_df.sort_values("Count"),
                    x="Count",
                    y="Word",
                    orientation="h",
                    text="Count",
                )
                fig_wf.update_traces(
                    marker_color=SENTIMENT_COLORS[sk],
                    textposition="outside",
                    textfont_size=11,
                )
                fig_wf.update_layout(**PLOTLY_LAYOUT, height=480, showlegend=False)
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.info(f"No {sk} reviews in this dataset.")

    if "Score" in df.columns:
        st.markdown("---")
        st.markdown(
            "<div class='section-label'>Star Rating Distribution</div>",
            unsafe_allow_html=True,
        )
        sc = df["Score"].value_counts().sort_index().reset_index()
        sc.columns = ["Stars", "Count"]
        fig_sc = px.bar(sc, x="Stars", y="Count", text="Count")
        fig_sc.update_traces(
            textposition="outside",
            textfont_size=11,
            marker_color=[
                SENTIMENT_COLORS["Negative"],
                SENTIMENT_COLORS["Negative"],
                SENTIMENT_COLORS["Neutral"],
                SENTIMENT_COLORS["Positive"],
                SENTIMENT_COLORS["Positive"],
            ],
        )
        fig_sc.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=280)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    pos_pct = dist.get("Positive", {}).get("pct", 0)
    neg_pct = dist.get("Negative", {}).get("pct", 0)
    if pos_pct >= 70:
        health_icon, health_text, health_color = (
            "🟢",
            "Strong -- customers are largely satisfied.",
            "#7bc47a",
        )
    elif neg_pct >= 30:
        health_icon, health_text, health_color = (
            "🔴",
            "Critical -- high negative feedback requires immediate action.",
            "#d46b6b",
        )
    else:
        health_icon, health_text, health_color = (
            "🟡",
            "Mixed -- significant room for improvement.",
            "#d4a853",
        )

    st.markdown(
        f"<div class='callout'>"
        f"<span style='color:{health_color};font-weight:600'>{health_icon} {health_text}</span><br>"
        f"<span style='color:var(--text-muted);font-size:13px;margin-top:4px;display:block'>"
        f"Positive: <b>{pos_pct}%</b> &nbsp;·&nbsp; Negative: <b>{neg_pct}%</b> &nbsp;·&nbsp; "
        f"Avg rating: <b>{stats['average_score'] or '--'}</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 3 -- Product Insights
# -----------------------------------------------------------------------------
elif page == "◆ Product Insights":
    st.markdown(
        "<div class='page-title'>Product Insights</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='page-subtitle'>Compare sentiment performance across products.</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV (optional)", type=["csv"], key="prod_upload"
    )
    df = pd.read_csv(uploaded) if uploaded is not None else load_default_data()

    if df is None:
        st.error("No dataset found.")
        st.stop()
    if "Sentiment" not in df.columns or "ProductId" not in df.columns:
        st.error("Dataset must contain 'ProductId' and 'Sentiment' columns.")
        st.stop()

    st.markdown("<div class='section-label'>Filter Mode</div>", unsafe_allow_html=True)
    FILTER_OPTIONS = {
        "Top Positive": "positive",
        "Top Negative": "negative",
        "Most Balanced": "neutral",
        "Best Overall": "best",
        "Worst Overall": "worst",
        "Most Reviewed": "review_count",
    }

    if "prod_filter" not in st.session_state:
        st.session_state.prod_filter = "review_count"

    btn_cols = st.columns(len(FILTER_OPTIONS))
    for col, (label, val) in zip(btn_cols, FILTER_OPTIONS.items()):
        with col:
            if st.button(label, key=f"btn_{val}"):
                st.session_state.prod_filter = val

    active_filter = st.session_state.prod_filter
    active_label = [k for k, v in FILTER_OPTIONS.items() if v == active_filter][0]
    st.markdown(
        f"<div class='filter-active'>● {active_label}</div>", unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    top_n = st.slider("Products to show", 5, 30, 10)
    product_df = product_sentiment_summary(df, top_n=top_n, filter_by=active_filter)

    if product_df.empty:
        st.warning("Not enough data (need >= 3 reviews per product).")
        st.stop()

    best_row = product_df.loc[product_df["Positive"].idxmax()]
    worst_row = product_df.loc[product_df["Negative"].idxmax()]

    k1, k2 = st.columns(2)
    with k1:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='section-label'>Highest Positive Rate</div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:var(--text-muted);margin-bottom:4px'>{best_row['ProductId']}</div>"
            f"<span style='color:var(--accent-green);font-size:26px;font-weight:600;font-family:IBM Plex Mono,monospace'>{best_row['Positive']:.1f}%</span>"
            f"<span style='color:var(--text-muted);font-size:12px;margin-left:8px'>positive · {int(best_row['review_count'])} reviews</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='section-label'>Highest Negative Rate</div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:var(--text-muted);margin-bottom:4px'>{worst_row['ProductId']}</div>"
            f"<span style='color:var(--accent-red);font-size:26px;font-weight:600;font-family:IBM Plex Mono,monospace'>{worst_row['Negative']:.1f}%</span>"
            f"<span style='color:var(--text-muted);font-size:12px;margin-left:8px'>negative · {int(worst_row['review_count'])} reviews</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div class='section-label'>Sentiment Mix per Product</div>",
        unsafe_allow_html=True,
    )
    fig_stack = go.Figure()
    for sentiment, color in SENTIMENT_COLORS.items():
        if sentiment in product_df.columns:
            fig_stack.add_trace(
                go.Bar(
                    name=sentiment,
                    x=product_df["ProductId"],
                    y=product_df[sentiment],
                    marker_color=color,
                    marker_line_width=0,
                )
            )
    fig_stack.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=360,
        xaxis_title=None,
        yaxis_title="% of Reviews",
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12)),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div class='section-label'>Product Summary</div>", unsafe_allow_html=True
    )
    display_df = product_df.copy()
    for col in ["Positive", "Neutral", "Negative"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}%")
    st.dataframe(
        display_df.rename(
            columns={
                "review_count": "Reviews",
                "dominant_sentiment": "Dominant",
                "performance_label": "Label",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown(
        "<div class='section-label'>Performance Label Distribution</div>",
        unsafe_allow_html=True,
    )
    label_counts = product_df["performance_label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    label_color_map = {
        "🏆 Best": "#7bc47a",
        "✅ Good": "#7aacd4",
        "⚖️ Balanced": "#d4a853",
        "⚠️ Risky": "#d4a853",
        "🚨 Worst": "#d46b6b",
    }
    fig_label = px.bar(
        label_counts,
        x="Label",
        y="Count",
        text="Count",
        color="Label",
        color_discrete_map=label_color_map,
    )
    fig_label.update_traces(
        textposition="outside", textfont_size=12, marker_line_width=0
    )
    fig_label.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=260)
    st.plotly_chart(fig_label, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 4 -- Real-Time Analysis
# -----------------------------------------------------------------------------
elif page == "◆ Real-Time Analysis":
    try:
        model, vectorizers, meta = load_model()
        render_realtime_page(model, vectorizers, meta)
    except FileNotFoundError:
        st.error("Model not found. Run `python main.py` to train first.")
        render_realtime_page(None, None, {})
