import os
import re
import html
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from joblib import load
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

_SIA = None
def get_vader_analyzer():
    global _SIA
    if _SIA is None:
        _SIA = SentimentIntensityAnalyzer()
    return _SIA

# Comprehensive Emoji & Emoticon Sentiment Map
# In case a user enters text with emojis or emoticons, we convert them to standard words
# before stripping punctuation, ensuring their emotional sentiment is fully captured!
EMOJI_SENTIMENT_MAP = {
    # Positive Emojis & Emoticons
    "😊": " smile happy ", "😃": " happy ", "😄": " happy ", "😁": " grin happy ",
    "😀": " happy ", "🥰": " love adore ", "😍": " love heart ", "🤩": " excited star ",
    "😘": " kiss love ", "❤️": " love heart ", "💖": " love heart ", "💕": " love hearts ",
    "🎉": " celebration party ", "🔥": " fire awesome ", "✨": " sparkling amazing ",
    "👍": " thumbs up good ", "👏": " applause praise ", "🙌": " celebrating yay ",
    "💯": " perfect hundred ", ":)": " smile happy ", ":-)": " smile happy ",
    ":D": " laugh happy ", ":d": " laugh happy ", "<3": " love heart ",
    # Negative Emojis & Emoticons
    "😡": " angry rage ", "🤬": " angry curse ", "😠": " mad angry ", "👿": " evil angry ",
    "😢": " cry sad ", "😭": " sobbing crying ", "💔": " heartbreak sad ", "😞": " disappointed sad ",
    "😒": " annoyed unamused ", "🙄": " eye roll annoyed ", "🤮": " disgust vomit ",
    "🤢": " sick disgust ", "👎": " thumbs down bad ", "💀": " dead terrible ",
    ":(": " sad frown ", ":-(": " sad frown ", ":'(": " crying sad "
}

def clean_tweet_advanced(text: str) -> str:
    '''
    Normalizes social media text while preserving critical sentiment signals:
    - Emojis & emoticons (translated to standard sentiment words)
    - Negation contractions ('can\'t', 'won\'t', 'n\'t', 'not')
    - Hashtag words ('#awesome' -> 'awesome')
    - HTML entities (&amp; -> &, &lt; -> <)
    - Strips URLs and user mentions (@username)
    '''
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML artifacts
    text = html.unescape(text)
    
    # 2. Translate emojis and emoticons to words so their sentiment is captured before stripping symbols
    for emo, meaning in EMOJI_SENTIMENT_MAP.items():
        if emo in text:
            text = text.replace(emo, meaning)
            
    # 3. Lowercase
    text = text.lower()
    
    # 3. Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Remove Twitter handles (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 5. Extract hashtag words (#happy -> happy)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 6. Normalize common contractions so negations are preserved
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    # 7. Keep only letters, whitespace, and basic punctuation
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 8. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Backward compatibility alias
clean_tweet = clean_tweet_advanced

def extract_vader_features(texts):
    '''
    Extracts 4-dimensional sentiment lexicon vector: [pos, neg, neu, compound_rescaled].
    Compound is rescaled from [-1, 1] to [0, 1] via (x + 1) / 2 for non-negativity.
    '''
    sia = get_vader_analyzer()
    scores = []
    for t in texts:
        pol = sia.polarity_scores(t)
        compound_rescaled = (pol['compound'] + 1.0) / 2.0
        scores.append([pol['pos'], pol['neg'], pol['neu'], compound_rescaled])
    return np.array(scores, dtype=np.float32)

def load_model_objects():
    '''
    Loads champion model and vectorizer with automatic fallback resolution.
    '''
    candidate_models = [
        os.path.join("models", "sentiment140_champion_model.joblib"),
        os.path.join("models", "sentiment_model.joblib"),
        os.path.join("models", "sentiment140_model.joblib")
    ]
    candidate_vecs = [
        os.path.join("models", "sentiment140_hybrid_vectorizer.joblib"),
        os.path.join("models", "tfidf_vectorizer.joblib"),
        os.path.join("models", "sentiment140_vectorizer.joblib")
    ]
    
    model_path = next((p for p in candidate_models if os.path.exists(p)), None)
    vec_path = next((p for p in candidate_vecs if os.path.exists(p)), None)
    
    if not model_path or not vec_path:
        raise FileNotFoundError(
            "Trained model artifacts not found in 'models/'. "
            "Please run 'python train_model.py' first to train and serialize the model."
        )
    
    model = load(model_path)
    vectorizer = load(vec_path)
    return model, vectorizer

def transform_fused_features(clean_texts, vectorizer, is_fused_model=True):
    '''
    Transforms raw clean texts into feature space matching the trained model.
    '''
    X_tfidf = vectorizer.transform(clean_texts)
    if is_fused_model:
        X_vader = csr_matrix(extract_vader_features(clean_texts))
        return hstack([X_tfidf, X_vader]).tocsr()
    return X_tfidf

def predict_dataframe(df, threshold=0.65):
    '''
    Performs full batch sentiment prediction with 3-tier confidence gating.
    '''
    model, vectorizer = load_model_objects()
    df = df.copy()
    
    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")
        
    df["clean_text"] = df["text"].apply(clean_tweet_advanced)
    clean_texts = df["clean_text"].tolist()
    
    # Check if model expects fused features (50,004 features) or standard TF-IDF
    expected_dim = getattr(model, "n_features_in_", None)
    is_fused = True if (expected_dim is not None and expected_dim > 50000) else False
    
    try:
        X = transform_fused_features(clean_texts, vectorizer, is_fused_model=True)
        probs = model.predict_proba(X)
    except Exception:
        # Fallback to standard TF-IDF if model was trained without VADER
        X = vectorizer.transform(clean_texts)
        probs = model.predict_proba(X)
        
    prob_pos = probs[:, 1]
    
    # 3-Tier Classification
    decisions = []
    statuses = []
    labels = []
    
    for p in prob_pos:
        if p >= threshold:
            decisions.append("Positive")
            statuses.append("High Confidence")
            labels.append(1)
        elif p <= (1.0 - threshold):
            decisions.append("Negative")
            statuses.append("High Confidence")
            labels.append(0)
        else:
            decisions.append("Neutral / Uncertain")
            statuses.append("Low Confidence (Borderline)")
            labels.append(1 if p >= 0.5 else 0)
            
    df["label"] = labels
    df["prob_pos"] = np.round(prob_pos, 4)
    df["sentiment"] = decisions
    df["confidence_status"] = statuses
    
    return df

def predict_from_texts(texts, threshold=0.65):
    '''
    Wrapper to predict sentiment for a list of strings.
    '''
    df = pd.DataFrame({"text": texts})
    return predict_dataframe(df, threshold=threshold)

def explain_text(text: str, top_k=8, threshold=0.65):
    '''
    Decomposes the prediction into individual token-level positive and negative drivers.
    Returns: (sentiment_str, prob_pos, pos_tokens, neg_tokens, confidence_status)
    '''
    model, vectorizer = load_model_objects()
    clean = clean_tweet_advanced(text)
    
    if not clean.strip():
        return "Neutral / Uncertain", 0.5, [], [], "Empty Input"
        
    # Check feature dimension
    expected_dim = getattr(model, "n_features_in_", None)
    is_fused = True if (expected_dim is not None and expected_dim > 50000) else False
    
    try:
        X = transform_fused_features([clean], vectorizer, is_fused_model=True)
        prob_pos = float(model.predict_proba(X)[0, 1])
    except Exception:
        X = vectorizer.transform([clean])
        prob_pos = float(model.predict_proba(X)[0, 1])
        
    # Determine 3-tier label
    if prob_pos >= threshold:
        decision = "Positive"
        status = "High Confidence"
    elif prob_pos <= (1.0 - threshold):
        decision = "Negative"
        status = "High Confidence"
    else:
        decision = "Neutral / Uncertain"
        status = "Low Confidence (Borderline)"
        
    # Extract feature names
    if hasattr(vectorizer, "transformer_list"):
        word_names = vectorizer.transformer_list[0][1].get_feature_names_out()
        char_names = vectorizer.transformer_list[1][1].get_feature_names_out()
        vader_names = ["vader_pos", "vader_neg", "vader_neu", "vader_compound"]
        feature_names = np.concatenate([word_names, char_names, vader_names])
    else:
        feature_names = vectorizer.get_feature_names_out()
        
    coefs = model.coef_[0]
    x_dense = X.toarray()[0]
    active_idx = x_dense.nonzero()[0]
    
    pairs = []
    for i in active_idx:
        feat_name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        # Skip pure sub-word char fragments in display if cleaner word n-grams exist
        score = x_dense[i] * coefs[i]
        pairs.append((feat_name, score))
        
    pos_tokens = [p for p in pairs if p[1] > 0]
    neg_tokens = [p for p in pairs if p[1] < 0]
    
    pos_tokens = sorted(pos_tokens, key=lambda t: t[1], reverse=True)[:top_k]
    neg_tokens = sorted(neg_tokens, key=lambda t: t[1])[:top_k]
    
    return decision, prob_pos, pos_tokens, neg_tokens, status

def plot_sentiment_bar(df):
    '''
    Visualizes sentiment distribution with 3-tier or 2-tier breakdown.
    '''
    col = "sentiment" if "sentiment" in df.columns else "label"
    counts = df[col].value_counts()
    
    fig, ax = plt.subplots(figsize=(6, 3.8))
    
    if col == "sentiment":
        categories = ["Positive", "Negative", "Neutral / Uncertain"]
        colors = ["#22c55e", "#ef4444", "#94a3b8"]
        values = [counts.get(c, 0) for c in categories]
    else:
        categories = ["Negative (0)", "Positive (1)"]
        colors = ["#ef4444", "#22c55e"]
        values = [counts.get(0, 0), counts.get(1, 0)]
        
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="none")
    ax.set_ylabel("Tweet Count", fontsize=10, fontweight="bold")
    ax.set_title("Sentiment Category Distribution", fontsize=12, fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Annotate bar counts
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    return fig

def wordcloud_for_sentiment(df, target_sentiment):
    '''
    Generates a WordCloud for a specified sentiment class (1, 0, or 'Positive', 'Negative').
    '''
    if "sentiment" in df.columns:
        if isinstance(target_sentiment, int):
            target_str = "Positive" if target_sentiment == 1 else "Negative"
        else:
            target_str = target_sentiment
        subset = df[df["sentiment"] == target_str]
    else:
        subset = df[df["label"] == target_sentiment]
        
    texts = " ".join(subset["clean_text"].dropna().tolist())
    if not texts.strip():
        return None
        
    color_map = "Greens" if (target_sentiment == 1 or target_sentiment == "Positive") else "Reds"
    wc = WordCloud(
        width=800,
        height=400,
        background_color="#0f172a",
        colormap=color_map,
        max_words=100
    ).generate(texts)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig
