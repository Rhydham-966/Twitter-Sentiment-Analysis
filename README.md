# 🌟 Twitter Sentiment Analyzer — Explainable AI Dashboard

> End-to-end NLP pipeline using **TF-IDF (unigrams & bigrams) + Logistic Regression** with Explainable AI token attribution, batch CSV processing, and an interactive Streamlit dashboard.



## 🎯 Overview

A complete, production-style **Sentiment Analysis System** that goes beyond basic positive/negative prediction — it provides transparent **Explainable AI (XAI) token attribution**, showing exactly which words drove the sentiment score and why.

Built for:
- Social media sentiment analysis (tweets, comments, feedback)
- Customer support ticket triage and churn risk identification
- Batch scoring and reporting on text datasets via CSV upload
- Educational inspection and technical interview demonstration

---

## ✨ Features

### 🔍 Real-Time Text Analysis
- Instant sentiment classification: **Positive (1)** or **Negative (0)**
- **Confidence score** with a real-time progress gauge
- Low-latency inference powered by a lightweight linear model

### 📊 CSV Batch Processing (Report Studio)
- Upload any CSV file containing a `text` column
- Automated prediction table with confidence probabilities
- Aggregate dataset metrics (total records, positive rate, negative risk signals)
- Interactive sentiment distribution bar chart
- Positive signal and negative friction **word clouds**

### 🧠 Explainable AI (Token Weight Attribution)
- Calculates mathematical token importance using linear model coefficients ($x_i \cdot w_i$)
- Highlights **positive drivers** (words pushing sentiment upward)
- Highlights **negative drivers** (words pulling sentiment downward)
- Delivers complete model transparency and interpretability

### 🎨 Modern Dark-Themed UI
- Two-tab Streamlit dashboard: **Live Text Analyzer** + **CSV Report Studio**
- Custom CSS styling with gradient backgrounds, card containers, and token chips

---

## 🏗️ System Architecture

```
Raw Text Input
      │
      ▼
Regex Text Sanitization
(URLs, @mentions, # symbol, non-letters removed; lowercased)
      │
      ▼
TF-IDF Vectorization
(ngram_range=(1, 2), max_features=10,000, English stop words)
      │
      ▼
Logistic Regression Classifier
(L2 regularization, L-BFGS solver)
      │
      ├──► Predicted Sentiment (Positive / Negative)
      ├──► Posterior Confidence Probability
      └──► Token-Level Weight Attribution (x_i * w_i)
                  │
                  ▼
Streamlit Dashboard / Jupyter Notebooks
```

---

## ⚙️ Tech Stack & Specifications

| Component | Technology | Specification / Details |
|---|---|---|
| **Language** | Python 3.10+ | Core development environment |
| **ML Classifier** | Scikit-Learn | Logistic Regression (L-BFGS, L2 penalty) |
| **Feature Extraction** | Scikit-Learn | `TfidfVectorizer` (`ngram_range=(1, 2)`, `max_features=10000`, `stop_words='english'`) |
| **Dataset Corpus** | NLTK | `twitter_samples` (10,000 balanced tweets: 5,000 positive, 5,000 negative) |
| **Text Preprocessing** | Python `re` (Regex) | Strips URLs, `@user`, `#` symbols, non-letters, and excessive whitespace |
| **Explainability** | Token Attribution | Linear decomposition: $\text{Score}_i = x_i \cdot w_i$ |
| **Visualizations** | Matplotlib, WordCloud | Sentiment bar distributions & polarity word clouds |
| **Web Interface** | Streamlit | Dark-themed, two-tab reactive dashboard |
| **Persistence** | Joblib | Serialized model and vectorizer in `models/` directory |

---

## 📁 Project Structure

```
Sentiment-Analysis/
│
├── README.md                      # Project documentation and specifications
├── requirements.txt               # Project dependencies
├── app.py                         # Streamlit interactive web dashboard
├── sentiment_utils.py             # Inference, explainability, and plotting engine
├── train_model.py                 # Offline training & artifact serialization script
├── test_analysis.py               # Local testing & sanity verification script
│
├── detailed_explanation.ipynb     # Comprehensive beginner-to-advanced learning guide
├── twitter_sentiment_analysis.ipynb # Professional project demonstration notebook
│
└── models/                        # Serialized model artifacts
    ├── sentiment_model.joblib     # Fitted Logistic Regression model
    └── tfidf_vectorizer.joblib    # Fitted TF-IDF Vectorizer
```

---

## 🚀 Setup & Execution Guide

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Sentiment-Analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpus
The training pipeline uses the NLTK `twitter_samples` corpus:
```bash
python -c "import nltk; nltk.download('twitter_samples')"
```

### 5. Train the Model (Optional — Pretrained Models Included)
To retrain the model and regenerate the serialized artifacts in `models/`:
```bash
python train_model.py
```

### 6. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### 7. Run Local CLI Tests or Notebooks
```bash
# Run local verification script
python test_analysis.py

# Open interactive notebooks
jupyter notebook
```

---

## 📊 Verified Model Performance

Evaluated on a **held-out test set of 2,000 tweets** (80/20 stratified split from 10,000 tweets):

| Metric | Measured Score | Details |
|---|---|---|
| **Overall Accuracy** | **75.30%** | 1,506 / 2,000 correct predictions |
| **Negative (Class 0) Precision** | **73.47%** | Low false alarm rate on negative complaints |
| **Negative (Class 0) Recall** | **79.20%** | Catches 792 / 1,000 actual negative tweets |
| **Negative (Class 0) F1-Score** | **0.7623** | Balanced harmonic mean |
| **Positive (Class 1) Precision** | **77.44%** | High precision when predicting positive |
| **Positive (Class 1) Recall** | **71.40%** | Catches 714 / 1,000 actual positive tweets |
| **Positive (Class 1) F1-Score** | **0.7430** | Balanced harmonic mean |
| **Macro Average F1** | **0.7526** | Consistent performance across both classes |

### Confusion Matrix (Test Split: 2,000 Tweets)
```
                  Predicted Negative    Predicted Positive
Actual Negative          792                   208
Actual Positive          286                   714
```

---

## 🔬 Explainable AI (XAI) Token Attribution

In linear classifiers, the decision value $z$ prior to the sigmoid function is:

$$z = w_0 + \sum_{i \in \text{Tokens}} x_i \cdot w_i$$

Where $x_i$ is the TF-IDF feature weight for token $i$ and $w_i$ is its learned coefficient:
* **Positive Contribution ($x_i \cdot w_i > 0$):** Tokens that increase the probability of positive sentiment (e.g., `"amazing"`, `"love"`, `"great"`).
* **Negative Contribution ($x_i \cdot w_i < 0$):** Tokens that decrease the probability of positive sentiment (e.g., `"terrible"`, `"worst"`, `"disappointed"`).

This attribution enables rapid error diagnosis, revealing corpus biases (such as intensifiers like `"really"` occurring predominantly in negative complaint tweets).

---

## 🔮 Future Enhancements

- Contextual Transformer model integration (DistilBERT / RoBERTa)
- Multi-class sentiment categorization (Positive, Neutral, Negative)
- Dynamic emoticon tokenization (e.g., mapping `:)` to `<EMO_POS>` instead of stripping)
- In-memory model caching with `@st.cache_resource` for enhanced web throughput
- REST API endpoint (FastAPI) for programmatic integration

---

## 👤 Author

**Rhydham Shah**
**Computer Science Student at PDEU**

