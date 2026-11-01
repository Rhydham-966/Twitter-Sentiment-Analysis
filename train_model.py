import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

nltk.download("twitter_samples")
nltk.download("punkt")
nltk.download("stopwords")

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset():
    pos_tweets = twitter_samples.strings("positive_tweets.json")
    neg_tweets = twitter_samples.strings("negative_tweets.json")
    texts = pos_tweets + neg_tweets
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)
    df = pd.DataFrame({"text": texts, "label": labels})
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.len() > 0]
    return df

def train_and_save():
    df = load_dataset()
    X = df["clean_text"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, y_pred))
    os.makedirs("models", exist_ok=True)
    dump(model, os.path.join("models", "sentiment_model.joblib"))
    dump(vectorizer, os.path.join("models", "tfidf_vectorizer.joblib"))
    print("Model and vectorizer saved in 'models/' folder.")

if __name__ == "__main__":
    train_and_save()
