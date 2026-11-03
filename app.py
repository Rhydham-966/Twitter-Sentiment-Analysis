import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentiment_utils import (
    predict_from_texts,
    predict_dataframe,
    plot_sentiment_bar,
    wordcloud_for_sentiment,
    explain_text,
)

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Engine",
    page_icon="⚡",
    layout="wide",
)

# ─── Minimal CSS — only colors and fonts, no layout hacks ───────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* Clean dark background */
[data-testid="stAppViewContainer"] {
    background-color: #0f0f0f !important;
}
[data-testid="stSidebar"] {
    background-color: #161616 !important;
}

/* Button styling */
button[kind="primary"] {
    background-color: #e5e5e5 !important;
    color: #0f0f0f !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
}
button[kind="primary"]:hover {
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Sentiment Engine")
    st.caption("1.6M Tweet Pipeline · v2.0")
    st.divider()

    st.markdown("##### 🎚️ Confidence Threshold")
    gating_threshold = st.slider(
        "Confidence Gating Threshold",
        min_value=0.50,
        max_value=0.85,
        value=0.65,
        step=0.05,
        help="Higher = stricter. Borderline predictions become 'Uncertain'.",
    )

    st.success(
        f"**Active: {gating_threshold:.2f}**\n\n"
        f"- P ≥ {gating_threshold:.2f} → **Positive** ✅\n"
        f"- P ≤ {1.0 - gating_threshold:.2f} → **Negative** ❌\n"
        f"- Between → **Uncertain** ⚠️"
    )

    st.divider()
    st.markdown("##### 📐 Model Stack")
    st.markdown(
        "- **35k** Word N-Grams `(1,2)`\n"
        "- **15k** Char N-Grams `char_wb`\n"
        "- **4D** VADER Lexicon Fusion\n"
        "- L2 LogReg `C=2.5` Champion\n"
        "- `<0.001ms` per prediction"
    )
    st.divider()
    st.caption("Built by **Rhydham Shah**")

# ─── Page Title ──────────────────────────────────────────────────
st.markdown("## ⚡ Sentiment Engine")
st.markdown(
    "Analyze any text with hybrid sub-word N-grams, VADER lexicon fusion, "
    "and 3-tier confidence gating. Achieves **≥90% accuracy** on confident predictions."
)
st.markdown("---")

# ─── Tabs ────────────────────────────────────────────────────────
tab_live, tab_csv = st.tabs(["⚡ Live Analyzer", "📊 Batch CSV"])


# ═════════════════════════════════════════════════════════════════
# TAB 1 — Live Analyzer
# ═════════════════════════════════════════════════════════════════
with tab_live:

    user_text = st.text_area(
        "Enter text to analyze",
        height=130,
        placeholder="Type or paste a tweet, product review, customer email, or any text…",
    )

    analyze_btn = st.button("🔍 Analyze Sentiment", type="primary")

    if analyze_btn and user_text.strip():
        decision, prob, pos_tokens, neg_tokens, status = explain_text(
            user_text, top_k=8, threshold=gating_threshold
        )

        st.markdown("---")

        # ── Result Section ──
        col_result, col_conf = st.columns(2)

        with col_result:
            if decision == "Positive":
                emoji, color_hex = "😊", "#22c55e"
            elif decision == "Negative":
                emoji, color_hex = "😡", "#ef4444"
            else:
                emoji, color_hex = "🤔", "#f59e0b"

            st.markdown(
                f"### {emoji} {decision}",
            )
            if status == "High Confidence":
                st.success(f"**{status}** — Model is highly certain about this prediction.")
            else:
                st.warning(f"**{status}** — Prediction falls in the borderline zone.")

        with col_conf:
            st.metric(
                label="P(Positive)",
                value=f"{prob:.1%}",
            )
            st.progress(int(prob * 100))

        st.markdown("---")

        # ── Token Attribution ──
        st.markdown("#### 🧬 Token Attribution — What drove this prediction")

        drv_pos, drv_neg = st.columns(2)

        with drv_pos:
            st.markdown("**▲ Positive Drivers**")
            if pos_tokens:
                for word, score in pos_tokens:
                    score_pct = min(abs(score) / 2.0, 1.0)
                    bar_len = int(score_pct * 20)
                    bar_viz = "🟩" * max(bar_len, 1)
                    st.markdown(
                        f"`{word}` → **+{score:.3f}** {bar_viz}"
                    )
            else:
                st.caption("No positive drivers detected.")

        with drv_neg:
            st.markdown("**▼ Negative Drivers**")
            if neg_tokens:
                for word, score in neg_tokens:
                    score_pct = min(abs(score) / 2.0, 1.0)
                    bar_len = int(score_pct * 20)
                    bar_viz = "🟥" * max(bar_len, 1)
                    st.markdown(
                        f"`{word}` → **{score:.3f}** {bar_viz}"
                    )
            else:
                st.caption("No negative drivers detected.")

    elif analyze_btn:
        st.warning("⚠️ Please enter some text first.")
    else:
        st.info("💡 Type something above and click **Analyze Sentiment** to see results.")


# ═════════════════════════════════════════════════════════════════
# TAB 2 — CSV Batch Studio
# ═════════════════════════════════════════════════════════════════
with tab_csv:

    st.markdown(
        f"Upload a CSV with a `text` column. "
        f"Every row gets scored with 3-tier gating at threshold **{gating_threshold:.2f}**."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)

        if "text" not in df_raw.columns:
            st.error("❌ CSV must contain a **text** column. Check your file and try again.")
        else:
            with st.spinner("🔄 Classifying with hybrid N-grams + VADER lexicon…"):
                df_pred = predict_dataframe(df_raw, threshold=gating_threshold)

            total = len(df_pred)
            pos = int((df_pred["sentiment"] == "Positive").sum())
            neg = int((df_pred["sentiment"] == "Negative").sum())
            neu = int((df_pred["sentiment"] == "Neutral / Uncertain").sum())

            st.markdown("---")

            # ── Stats Row ──
            st.markdown("#### 📊 Summary")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", f"{total:,}")
            m2.metric("Positive 😊", f"{pos:,}", delta=f"{pos/total*100:.1f}%" if total else "0%")
            m3.metric("Negative 😡", f"{neg:,}", delta=f"{neg/total*100:.1f}%" if total else "0%", delta_color="inverse")
            m4.metric("Uncertain 🤔", f"{neu:,}", delta=f"{neu/total*100:.1f}%" if total else "0%", delta_color="off")

            st.markdown("---")

            # ── Charts ──
            st.markdown("#### 📈 Visualizations")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**Sentiment Distribution**")
                fig_bar = plot_sentiment_bar(df_pred)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

            with chart_col2:
                st.markdown("**Positive WordCloud**")
                wc_pos = wordcloud_for_sentiment(df_pred, "Positive")
                if wc_pos:
                    st.pyplot(wc_pos, use_container_width=True)
                    plt.close(wc_pos)
                else:
                    st.caption("Not enough positive texts to generate cloud.")

            st.markdown("")

            st.markdown("**Negative WordCloud**")
            wc_neg = wordcloud_for_sentiment(df_pred, "Negative")
            if wc_neg:
                st.pyplot(wc_neg, use_container_width=True)
                plt.close(wc_neg)
            else:
                st.caption("Not enough negative texts to generate cloud.")

            st.markdown("---")

            # ── Data Table ──
            st.markdown("#### 🗃️ Scored Dataset")
            st.dataframe(df_pred, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("📤 Drop a CSV file above to generate a full sentiment report with charts and word clouds.")
