import os
import re
import pandas as pd
from joblib import load
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model_objects():
    model_path = os.path.join("models", "sentiment_model.joblib")
    vec_path = os.path.join("models", "tfidf_vectorizer.joblib")
    model = load(model_path)
    vectorizer = load(vec_path)
    return model, vectorizer

def predict_dataframe(df):
    model, vectorizer = load_model_objects()
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.len() > 0]
    X = vectorizer.transform(df["clean_text"])
    probs = model.predict_proba(X)
    preds = model.predict(X)
    df["label"] = preds
    df["prob_pos"] = probs[:, 1]
    return df

def predict_from_texts(texts):
    df = pd.DataFrame({"text": texts})
    return predict_dataframe(df)

def explain_text(text, top_k=8):
    model, vectorizer = load_model_objects()
    clean = clean_tweet(text)
    X = vectorizer.transform([clean])
    proba = model.predict_proba(X)[0, 1]
    label = int(model.predict(X)[0])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    x = X.toarray()[0]
    idx = x.nonzero()[0]
    pairs = []
    for i in idx:
        pairs.append((feature_names[i], x[i] * coefs[i]))
    pos_tokens = [p for p in pairs if p[1] > 0]
    neg_tokens = [p for p in pairs if p[1] < 0]
    pos_tokens = sorted(pos_tokens, key=lambda t: t[1], reverse=True)[:top_k]
    neg_tokens = sorted(neg_tokens, key=lambda t: t[1])[:top_k]
    return label, float(proba), pos_tokens, neg_tokens

def plot_sentiment_bar(df):
    counts = df["label"].value_counts().sort_index()
    labels = ["Negative", "Positive"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Texts")
    ax.set_title("Sentiment Distribution")
    return fig

def wordcloud_for_sentiment(df, label):
    texts = " ".join(df.loc[df["label"] == label, "clean_text"].tolist())
    if not texts:
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(texts)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    return fig
