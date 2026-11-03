# Twitter Sentiment Analysis System with xAI results 
### *1.6 Million Tweet Production Pipeline with Confidence-Gated Selective Classification*

### Dataset link : https://www.kaggle.com/datasets/kazanova/sentiment140


---

## 📌 Executive Summary

This repository provides an enterprise-ready, high-throughput **Sentiment Analysis and Opinion Mining System** trained on the comprehensive 1.6-million tweet **Sentiment140** corpus. 

Traditional social media sentiment models suffer from high noise, slang, typos, sarcasm, and forced binary classifications on ambiguous text. This system overcomes those challenges by unifying:
1. **Hybrid Sub-Word Character & Word N-Grams** (`char_wb`) to eliminate out-of-vocabulary errors caused by typos and slang.
2. **Semantic Emoji & Emoticon Translation** that maps visual expressions (`😊`, `🔥`, `😡`, `💔`, `:)`, `:-()`) to affective tokens before cleaning.
3. **Affective Lexicon Fusion (VADER)** providing human emotional priors across valence dimensions.
4. **Confidence-Gated Selective Classification**, routing ambiguous statements into an explicit **Neutral / Uncertain** tier and achieving **$\ge 90\%$ accuracy** on decisive opinions.
5. **Explainable AI (XAI)** token attribution, exposing the exact mathematical drivers behind every inference.

---

## 🏗️ Architecture & Processing Flow

```
                      Raw User Input (Tweet, Review, Text)
                                       │
                                       ▼
                     [1] Semantic Normalization Engine
      • Unicode Emoji & Emoticon Translation (😊 ➔ "smile happy", 😡 ➔ "angry")
      • Negation Preservation ("can't" ➔ "can not", "won't" ➔ "will not")
      • Noise Cleansing (URLs, @handles stripped; #hashtags extracted)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
     [2] Hybrid FeatureUnion               [3] NLTK VADER Lexicon Engine
   • 35,000 Word N-Grams (1, 2)              • Positive Valence Score
   • 15,000 Sub-word Char (3, 5)             • Negative Valence Score
   • Sublinear TF Scaling                    • Neutral Valence Score
   • Character Boundary `char_wb`            • Rescaled Compound Polarity
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                   Fused High-Dimensional Feature Matrix
                      (50,004 Features per Statement)
                                       │
                                       ▼
             [4] Production Champion Classifier (L2 LogReg, C=2.5)
                   Sub-millisecond Inference (< 0.001 ms/tweet)
                                       │
                                       ▼
                           Posterior Probability P(Pos)
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
  P(Pos) >= Threshold       P(Pos) <= (1 - Threshold)       Between Thresholds
    [Tier 1: Positive]         [Tier 2: Negative]         [Tier 3: Neutral/Ambiguous]
    High Confidence            High Confidence            Selective Abstention
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       ▼
                        [5] Explainable AI (XAI)
               Token-Level Attribution Scoring: Score_i = x_i · w_i
```

---

## 🌟 Core Technical Highlights

### 1. 🔤 Hybrid Sub-Word Representation (`char_wb`)
Social media text is notorious for creative spelling and typos (*"sooooo happy"*, *"amazzzing"*, *"terrrrible"*). 
- Standard word tokenizers treat each variant as an unseen word.
- Our pipeline pairs **35,000 word n-grams** with **15,000 sub-word character n-grams** ($3$ to $5$ characters) restricted to word boundaries (`analyzer='char_wb'`).
- This captures both full idioms and the emotional core of misspelled words without ballooning memory.

### 2. 🎭 Semantic Emoji & Emoticon Translation
Standard preprocessing strips all punctuation and symbols, losing critical emotional intent in modern communication.
- Our normalization engine maps over 30 common emojis and text emoticons (`😊`, `❤️`, `🔥`, `😡`, `😭`, `:)`, `:-(`, `<3`) into standardized semantic tokens *before* regex symbol stripping.
- An input like `"Worst experience 😡💔"` is translated to `"Worst experience angry rage heartbreak sad"`, allowing the model to leverage its trained lexical weights directly.

### 3. 🛡️ Confidence-Gated Selective Classification ($\ge 90\%$ Accuracy)
Real-world text includes mundane, non-opinionated updates (*"sitting in the airport waiting for flight"*). Forcing a binary $50/50$ prediction introduces error.
- The pipeline establishes an explicit **confidence gate**:
  $$\text{Confident if } P(\text{Positive}) \ge \tau \quad \text{or} \quad P(\text{Positive}) \le (1 - \tau)$$
- Statements within the margin $[\tau, 1-\tau]$ are classified as **Neutral / Uncertain**.
- At $\tau = 0.80$, the system delivers **$90.17\%$ accuracy** over decisive tweets.

### 4. 🔍 Explainable AI (Token-Level Attribution)
Every prediction can be audited. By projecting the input vector $x$ onto the model's weight vector $w$:
$$\text{Score}_i = x_i \cdot w_i$$
The dashboard computes and displays the exact words and lexicon signals pushing sentiment positive (green pills) or negative (red pills).

---

## 📊 Empirical Benchmarks

### 1. Multi-Model Architecture Comparison (50,004 Features)

Evaluated under identical 80/20 stratified train/test conditions on balanced subsets:

| Model | Representation | Train Time | Inference Time | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression ($C=2.5$)** | **50k Fused Sparse Matrix** | **3.80s** | **0.004s** | **79.73%** |
| **Linear Support Vector (SVC)** | 50k Fused Sparse Matrix | 5.32s | 0.004s | 79.06% |
| **SGD Classifier (Log-Loss)** | 50k Fused Sparse Matrix | 0.37s | 0.005s | 78.89% |
| **Complement Naive Bayes** | 50k Fused Sparse Matrix | 0.03s | 0.008s | 78.09% |
| **XGBoost (Stacking Meta-Learner)** | Dense 8-Column Meta-Matrix | 0.17s | 0.004s | 76.37% |
| **LightGBM (Stacking Meta-Learner)** | Dense 8-Column Meta-Matrix | 0.19s | 0.015s | 76.27% |
| **XGBoost (Raw Sparse Features)** | 50k Fused Sparse Matrix | 64.22s | 0.029s | 75.84% |

> **Key Observation:** Linear models with calibrated regularization dominate high-dimensional sparse text spaces, running $16\times$ faster than raw gradient-boosted trees while achieving superior accuracy.

---

### 2. Selective Classification Accuracy Progression

| Threshold ($\tau$) | Accuracy | Coverage | Evaluated Tweets | Classification Tier |
| :---: | :---: | :---: | :---: | :--- |
| **$\ge 0.50$** | 79.73% | 100.0% | 20,000 / 20,000 | Baseline (Forced Binary) |
| **$\ge 0.60$** | 83.30% | 87.9% | 17,578 / 20,000 | Low Noise Filtered |
| **$\ge 0.70$** | 86.66% | 74.7% | 14,945 / 20,000 | Confident Sentiment |
| **$\ge 0.75$** | 88.36% | 67.5% | 13,491 / 20,000 | High Opinion Density |
| **$\ge 0.80$** | **90.17%** | **59.4%** | **11,880 / 20,000** | **$\ge 90\%$ Production Target** |
| **$\ge 0.85$** | **91.79%** | **50.0%** | **9,990 / 20,000** | **Ultra-High Precision Tier** |

---


## ⚡ Quickstart Guide

### 1. Prerequisites & Installation

Clone the repository and install dependencies in your virtual environment:

```bash
git clone https://github.com/Rhydham-966/Twitter-Sentiment-Analysis
cd Sentiment-Analysis

# Create and activate environment
python -m venv .venv

# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```






---

### 3. Model Training CLI

Pre-trained models are included in `models/`. To retrain the pipeline from scratch on custom sizes or the full 1.6M dataset:

```bash
# Fast Prototype (100,000 tweets, ~30 seconds):
python train_model.py --sample-size 100000

# Medium Scale (500,000 tweets, ~2 minutes):
python train_model.py --sample-size 500000

# Full Scale (All 1,600,000 tweets, ~4 minutes):
python train_model.py --sample-size full
```

---

### 4. Launch the Interactive Web Dashboard

Launch the Streamlit interface:

```bash
python -m streamlit run app.py
```

Open your browser to `http://localhost:8501`.

#### Dashboard Capabilities:
* **🎯 Live Text Analyzer**: Real-time sentiment prediction, interactive confidence gauge, and color-coded token attribution pills showing positive and negative lexical drivers.
* **📊 CSV Batch Report Studio**: Upload any CSV containing a `text` column to generate 3-tier sentiment counts, interactive bar charts, polarity WordClouds, and exportable scored datasets.
* **⚙️ Tunable Confidence Slider**: Dynamically adjust the confidence gating threshold ($\tau \in [0.50, 0.85]$) in the sidebar.

---

## 👨‍💻 Author & Contact

**Rhydham Shah**  

---
