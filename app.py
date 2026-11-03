import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentiment_utils import (
    predict_from_texts,
    predict_dataframe,
    plot_sentiment_bar,
    wordcloud_for_sentiment,
    explain_text,
)

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🕊️", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #1f2933 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .big-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }
    .accent {
        color: #22c55e;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-top: 0.4rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(55,65,81,0.9);
        font-size: 0.8rem;
        color: #e5e7eb;
    }
    .card {
        background: rgba(15,23,42,0.92);
        border-radius: 1rem;
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(55,65,81,0.9);
        box-shadow: 0 18px 45px rgba(0,0,0,0.65);
    }
    .card-soft {
        background: rgba(15,23,42,0.88);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(55,65,81,0.7);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
    }
    .metric-pill-pos {
        color: #bbf7d0;
        background: rgba(22,163,74,0.2);
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
    }
    .metric-pill-neg {
        color: #fecaca;
        background: rgba(220,38,38,0.22);
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .token-chip-pos {
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.55rem;
        margin: 0.12rem;
        border-radius: 999px;
        background: rgba(22,163,74,0.22);
        color: #bbf7d0;
        font-size: 0.75rem;
    }
    .token-chip-neg {
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.55rem;
        margin: 0.12rem;
        border-radius: 999px;
        background: rgba(220,38,38,0.22);
        color: #fecaca;
        font-size: 0.75rem;
    }
    .token-score {
        font-size: 0.7rem;
        opacity: 0.8;
        margin-left: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pill">🕊️ NLP • TF-IDF • Logistic Regression • Token-level Explainability</div>
    <div class="big-title">Twitter Sentiment Analyzer</div>
    <div class="subtitle">
        Production-style dashboard to score sentiment, visualize polarity, and highlight the words driving each prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["🎯 Live Text Analyzer", "📊 CSV Report Studio"])

with tabs[0]:
    c1, c2 = st.columns([1.35, 1])

    with c1:
        st.markdown('<div class="section-title">Enter or paste text</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        user_text = st.text_area(
            "",
            height=180,
            placeholder="Paste a tweet, product review, or any paragraph to analyze its sentiment and see top contributing tokens…",
        )
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Insight summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if analyze_btn and user_text.strip() != "":
            label, prob, pos_tokens, neg_tokens = explain_text(user_text)

            sentiment_text = "Positive" if label == 1 else "Negative"
            sentiment_emoji = "😊" if label == 1 else "😡"
            sentiment_color = "#22c55e" if label == 1 else "#ef4444"

            st.markdown(
                f"""
                <div style="font-size:0.9rem;color:#9ca3af;margin-bottom:0.35rem;">Predicted sentiment</div>
                <div style="font-size:2rem;font-weight:700;color:{sentiment_color};">
                    {sentiment_text} {sentiment_emoji}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            c_top = st.columns(2)
            with c_top[0]:
                st.markdown('<div class="metric-label">Confidence (positive class)</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="metric-value">{prob:.2f}</div>',
                    unsafe_allow_html=True,
                )
            with c_top[1]:
                bar_val = int(prob * 100)
                st.markdown('<div class="metric-label">Confidence gauge</div>', unsafe_allow_html=True)
                st.progress(bar_val)

            st.markdown("<hr style='border-color:rgba(55,65,81,0.8);margin:0.9rem 0;'>", unsafe_allow_html=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown('<div class="metric-label">Positive drivers</div>', unsafe_allow_html=True)
                if pos_tokens:
                    chips = ""
                    for word, score in pos_tokens:
                        chips += f"<span class='token-chip-pos'>{word}<span class='token-score'>{score:.3f}</span></span>"
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No strong positive tokens detected.")
            with cc2:
                st.markdown('<div class="metric-label">Negative drivers</div>', unsafe_allow_html=True)
                if neg_tokens:
                    chips = ""
                    for word, score in neg_tokens:
                        chips += f"<span class='token-chip-neg'>{word}<span class='token-score'>{score:.3f}</span></span>"
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No strong negative tokens detected.")

        else:
            st.markdown(
                """
                <div style="font-size:0.9rem;color:#9ca3af;">
                    Run an analysis to see prediction, confidence gauges, and the words that push the model towards positive or negative sentiment.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section-title">Upload CSV for batch sentiment report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-soft">Upload a CSV with a <b>text</b> column. The app will score each row and build visual insights for your report.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)

        if "text" not in df_raw.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            df_pred = predict_dataframe(df_raw)

            total = len(df_pred)
            pos = int((df_pred["label"] == 1).sum())
            neg = int((df_pred["label"] == 0).sum())
            pos_rate = (pos / total * 100) if total > 0 else 0.0

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown('<div class="card-soft">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total records</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{total}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="card-soft">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Positive</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{pos}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-pill-pos">High-intent / happy users</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with m3:
                st.markdown('<div class="card-soft">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Negative</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{neg}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-pill-neg">Churn / risk signals</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            g1, g2 = st.columns([1.1, 1.1])

            with g1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Sentiment distribution</div>', unsafe_allow_html=True)
                fig_bar = plot_sentiment_bar(df_pred)
                st.pyplot(fig_bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with g2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Positive signal cloud</div>', unsafe_allow_html=True)
                wc_pos = wordcloud_for_sentiment(df_pred, 1)
                if wc_pos:
                    st.pyplot(wc_pos, use_container_width=True)
                else:
                    st.write("No positive texts found.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Negative friction cloud</div>', unsafe_allow_html=True)
            wc_neg = wordcloud_for_sentiment(df_pred, 0)
            if wc_neg:
                st.pyplot(wc_neg, use_container_width=True)
            else:
                st.write("No negative texts found.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Scored dataset</div>', unsafe_allow_html=True)
            st.dataframe(df_pred, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("No file uploaded yet. Drop a CSV to generate a full sentiment report.")
