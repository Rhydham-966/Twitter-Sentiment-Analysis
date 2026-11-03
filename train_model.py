import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
import time
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb
import lightgbm as lgb

from sentiment_utils import clean_tweet_advanced, extract_vader_features

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Sentiment140 Pipeline (Matches Notebook 1:1)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="training.1600000.processed.noemoticon.csv",
        help="Path to Sentiment140 CSV file"
    )
    parser.add_argument(
        "--sample-size",
        type=str,
        default="100000",
        help="Sample size: enter integer (e.g. 100000) or 'all'/'full' for all 1.6 million tweets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save serialized model and vectorizer"
    )
    return parser.parse_args()

def load_and_preprocess_data(dataset_path: str, sample_size_str: str):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset '{dataset_path}' not found! Please place 'training.1600000.processed.noemoticon.csv' in the project root."
        )

    print(f"[STEP 2] Ingesting Sentiment140 dataset from: {dataset_path} ...")
    t0 = time.time()
    column_names = ["target", "ids", "date", "flag", "user", "text"]
    
    df_raw = pd.read_csv(
        dataset_path,
        encoding="latin-1",
        names=column_names,
        usecols=["target", "text"]
    )
    
    # Map Sentiment140 labels: 0 -> Negative (0), 4 -> Positive (1)
    df_raw["target"] = df_raw["target"].map({0: 0, 4: 1})
    print(f"  -> Ingested {len(df_raw):,} raw tweets in {time.time()-t0:.2f}s.")

    # Determine sample size
    sample_size_lower = sample_size_str.lower().strip()
    if sample_size_lower in ["all", "full", "none", "1600000", "1.6m"]:
        print("[STEP 5] Full Scale Mode: Training on ALL 1,600,000 tweets!")
        df = df_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        try:
            n_samples = int(sample_size_str)
        except ValueError:
            n_samples = 100000
        
        n_samples = min(n_samples, len(df_raw))
        print(f"[STEP 5] Sample Mode: Taking balanced {n_samples:,} tweet subset...")
        df_pos = df_raw[df_raw["target"] == 1].sample(n_samples // 2, random_state=42)
        df_neg = df_raw[df_raw["target"] == 0].sample(n_samples // 2, random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[STEP 4] Normalizing {len(df):,} tweets with advanced regex + emoji translation...")
    t_clean = time.time()
    df["clean_text"] = df["text"].apply(clean_tweet_advanced)
    print(f"  -> Normalization completed in {time.time()-t_clean:.2f}s.")

    return df

def run_end_to_end_training(args):
    print("=" * 95)
    print("🚀 SENTIMENT140 END-TO-END PRODUCTION TRAINING PIPELINE (1:1 WITH NOTEBOOK)")
    print("=" * 95)

    # 1. Load and Normalize Dataset
    df = load_and_preprocess_data(args.dataset, args.sample_size)

    # 2. Train-Test Split (80% Train, 20% Test)
    print("\n[SPLIT] Partitioning into 80% Train and 20% Test sets (stratified)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"].values,
        df["target"].values,
        test_size=0.20,
        random_state=42,
        stratify=df["target"].values
    )
    print(f"  Training Set : {len(X_train_text):,} tweets")
    print(f"  Testing Set  : {len(X_test_text):,} tweets")

    # 3. Hybrid Feature Engineering (Word N-Grams + Sub-word Char N-Grams)
    print("\n[STEP 6] Constructing Hybrid FeatureUnion (Word N-Grams + Sub-word Character N-Grams)...")
    t_vec = time.time()
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=35000,
        sublinear_tf=True,
        min_df=3
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=15000,
        sublinear_tf=True,
        min_df=5
    )

    hybrid_vectorizer = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer)
    ])

    X_train_tfidf = hybrid_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = hybrid_vectorizer.transform(X_test_text)
    print(f"  -> TF-IDF extraction complete in {time.time()-t_vec:.2f}s.")
    print(f"     Sparse shape: {X_train_tfidf.shape[1]:,} features (35k Word + 15k Char)")

    # 4. Sentiment Lexicon Fusion (VADER)
    print("\n[STEP 7] Extracting VADER emotional polarity scores [pos, neg, neu, compound]...")
    t_vader = time.time()
    X_train_vader = extract_vader_features(X_train_text)
    X_test_vader = extract_vader_features(X_test_text)
    print(f"  -> VADER extraction complete in {time.time()-t_vader:.2f}s.")

    # 5. Fuse TF-IDF and Lexicon Features
    X_train_fused = hstack([X_train_tfidf, csr_matrix(X_train_vader)]).tocsr()
    X_test_fused = hstack([X_test_tfidf, csr_matrix(X_test_vader)]).tocsr()
    print(f"  -> Fused Feature Matrix: {X_train_fused.shape[1]:,} features (50,000 TF-IDF + 4 VADER).")

    # 6. Multi-Model Benchmark & Stacking (Matches Notebook Step 8)
    print("\n[STEP 8] Executing Multi-Model Benchmark across Linear, Probabilistic & Tree Models:")
    print("=" * 95)
    print(f"{'Model Name':<30} | {'Representation':<22} | {'Train Time':<10} | {'Infer Time':<10} | {'Accuracy':<9}")
    print("=" * 95)

    benchmark_results = []
    fitted_base_models = {}

    base_models = {
        "Logistic Regression (C=2.5)": LogisticRegression(C=2.5, max_iter=1000, random_state=42),
        "SGD Classifier (Log-Loss)": SGDClassifier(loss='log_loss', max_iter=1000, random_state=42),
        "Linear Support Vector (SVC)": LinearSVC(C=0.5, max_iter=2000, random_state=42),
        "Complement Naive Bayes": ComplementNB(alpha=0.5)
    }

    for name, clf in base_models.items():
        t_tr = time.time()
        clf.fit(X_train_fused, y_train)
        t_train = time.time() - t_tr

        t_inf = time.time()
        preds = clf.predict(X_test_fused)
        t_infer = time.time() - t_inf

        acc = accuracy_score(y_test, preds)
        rep = "Raw 50k Fused (Sparse)"
        benchmark_results.append({
            "Model": name,
            "Feature Space": rep,
            "Train Time (s)": round(t_train, 2),
            "Infer Time (s)": round(t_infer, 3),
            "Accuracy (%)": round(acc * 100, 2)
        })
        fitted_base_models[name] = clf
        print(f"{name:<30} | {rep:<22} | {t_train:>8.2f}s | {t_infer:>8.3f}s | {acc*100:>8.2f}%")

    # Raw XGBoost
    print("\nTraining XGBoost on 50,004 Sparse TF-IDF + VADER features...")
    t_xgb_raw_start = time.time()
    xgb_raw = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    if len(X_train_text) > 100000:
        print(f"  ℹ️ [Performance Guard]: Training dataset is large ({len(X_train_text):,} rows).")
        print("  Training raw XGBoost on a 100k slice to protect RAM and avoid long waits.")
        xgb_raw.fit(X_train_fused[:100000], y_train[:100000])
        rep_xgb = "Raw 50k (100k slice)"
    else:
        xgb_raw.fit(X_train_fused, y_train)
        rep_xgb = "Raw 50k Fused (Sparse)"

    t_xgb_raw_train = time.time() - t_xgb_raw_start
    t_inf = time.time()
    xgb_raw_preds = xgb_raw.predict(X_test_fused)
    t_xgb_raw_infer = time.time() - t_inf
    acc_xgb_raw = accuracy_score(y_test, xgb_raw_preds)

    benchmark_results.append({
        "Model": "XGBoost (Raw Sparse Features)",
        "Feature Space": rep_xgb,
        "Train Time (s)": round(t_xgb_raw_train, 2),
        "Infer Time (s)": round(t_xgb_raw_infer, 3),
        "Accuracy (%)": round(acc_xgb_raw * 100, 2)
    })
    print(f"{'XGBoost (Raw Sparse Features)':<30} | {rep_xgb:<22} | {t_xgb_raw_train:>8.2f}s | {t_xgb_raw_infer:>8.3f}s | {acc_xgb_raw*100:>8.2f}%")
    print("=" * 95)

    # 7. Construct Meta-Feature Matrix for Stacking
    print("\nConstructing Dense Meta-Feature Matrix (8 Columns)...")
    prob_lr_tr = fitted_base_models["Logistic Regression (C=2.5)"].predict_proba(X_train_fused)[:, 1:2]
    prob_sgd_tr = fitted_base_models["SGD Classifier (Log-Loss)"].predict_proba(X_train_fused)[:, 1:2]
    prob_cnb_tr = fitted_base_models["Complement Naive Bayes"].predict_proba(X_train_fused)[:, 1:2]
    dec_svc_tr = fitted_base_models["Linear Support Vector (SVC)"].decision_function(X_train_fused).reshape(-1, 1)

    meta_X_train = np.hstack([prob_lr_tr, prob_sgd_tr, prob_cnb_tr, dec_svc_tr, X_train_vader])

    prob_lr_te = fitted_base_models["Logistic Regression (C=2.5)"].predict_proba(X_test_fused)[:, 1:2]
    prob_sgd_te = fitted_base_models["SGD Classifier (Log-Loss)"].predict_proba(X_test_fused)[:, 1:2]
    prob_cnb_te = fitted_base_models["Complement Naive Bayes"].predict_proba(X_test_fused)[:, 1:2]
    dec_svc_te = fitted_base_models["Linear Support Vector (SVC)"].decision_function(X_test_fused).reshape(-1, 1)

    meta_X_test = np.hstack([prob_lr_te, prob_sgd_te, prob_cnb_te, dec_svc_te, X_test_vader])

    # LightGBM Meta-Learner
    t_lgb_start = time.time()
    lgb_meta = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    lgb_meta.fit(meta_X_train, y_train)
    t_lgb_train = time.time() - t_lgb_start

    t_inf = time.time()
    lgb_preds = lgb_meta.predict(meta_X_test)
    t_lgb_infer = time.time() - t_inf
    acc_lgb = accuracy_score(y_test, lgb_preds)

    benchmark_results.append({
        "Model": "LightGBM (Stacking Meta-Model)",
        "Feature Space": "Dense Meta (8 cols)",
        "Train Time (s)": round(t_lgb_train, 2),
        "Infer Time (s)": round(t_lgb_infer, 3),
        "Accuracy (%)": round(acc_lgb * 100, 2)
    })

    # XGBoost Meta-Learner
    t_xgb_meta_start = time.time()
    xgb_meta = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    xgb_meta.fit(meta_X_train, y_train)
    t_xgb_meta_train = time.time() - t_xgb_meta_start

    t_inf = time.time()
    xgb_meta_preds = xgb_meta.predict(meta_X_test)
    t_xgb_meta_infer = time.time() - t_inf
    acc_xgb_meta = accuracy_score(y_test, xgb_meta_preds)

    benchmark_results.append({
        "Model": "XGBoost (Stacking Meta-Model)",
        "Feature Space": "Dense Meta (8 cols)",
        "Train Time (s)": round(t_xgb_meta_train, 2),
        "Infer Time (s)": round(t_xgb_meta_infer, 3),
        "Accuracy (%)": round(acc_xgb_meta * 100, 2)
    })

    df_bench = pd.DataFrame(benchmark_results).sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
    print("\nFinal Comparative Benchmark Across Linear, Probabilistic & Tree Models:")
    print(df_bench.to_string())

    # Champion Model Selection
    champ_model = fitted_base_models["Logistic Regression (C=2.5)"]
    champ_probs = champ_model.predict_proba(X_test_fused)[:, 1]
    champ_preds = (champ_probs >= 0.5).astype(int)

    # 8. Confidence-Gated Selective Classification Progression (Matches Notebook Step 9)
    print("\n[STEP 9] Confidence-Gated Selective Classification Progression:")
    print("-" * 80)
    print(f"{'Threshold':<15} | {'Accuracy (%)':<15} | {'Coverage (%)':<15} | {'Evaluated Tweets'}")
    print("-" * 80)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    for th in thresholds:
        mask = (champ_probs >= th) | (champ_probs <= (1.0 - th))
        sub_y = y_test[mask]
        sub_p = champ_preds[mask]
        sub_acc = accuracy_score(sub_y, sub_p) * 100
        cov = (len(sub_y) / len(y_test)) * 100
        print(f">= {th:.2f}{'':<9} | {sub_acc:>12.2f}% | {cov:>12.1f}% | {len(sub_y):,} / {len(y_test):,}")
    print("-" * 80)

    # 9. Explainable AI Token Attribution (Matches Notebook Step 10)
    print("\n[STEP 10] Top 15 Sentiment Drivers (Explainable AI Token Attribution):")
    word_names = hybrid_vectorizer.transformer_list[0][1].get_feature_names_out()
    char_names = hybrid_vectorizer.transformer_list[1][1].get_feature_names_out()
    vader_names = ["vader_pos", "vader_neg", "vader_neu", "vader_compound_rescaled"]
    all_feature_names = np.concatenate([word_names, char_names, vader_names])
    coefs = champ_model.coef_[0]

    top_pos_idx = np.argsort(coefs)[-10:][::-1]
    top_neg_idx = np.argsort(coefs)[:10]

    print("  Top Positive Drivers:")
    for idx in top_pos_idx:
        print(f"    + {all_feature_names[idx]:<25} : +{coefs[idx]:.3f}")
    print("  Top Negative Drivers:")
    for idx in top_neg_idx:
        print(f"    - {all_feature_names[idx]:<25} : {coefs[idx]:.3f}")

    # 10. Live Zero-Shot Generalization Test (Matches Notebook Step 11)
    print("\n[STEP 11] Evaluating Zero-Shot Generalization on 10 Real-World Customer Statements:")
    live_benchmark = [
        ("I love this product, it is amazing and works perfectly!", 1),
        ("This is the worst experience I have ever had.", 0),
        ("The movie was fantastic, I enjoyed every moment of it.", 1),
        ("I am very disappointed with the service.", 0),
        ("What a wonderful day, feeling great and happy!", 1),
        ("The food was terrible and I will never come back.", 0),
        ("Customer support was helpful and resolved my issue quickly.", 1),
        ("The app keeps crashing, totally useless.", 0),
        ("Really impressed by the quality and performance.", 1),
        ("I hate how slow this website is.", 0)
    ]

    live_correct = 0
    print("-" * 85)
    for text, true_lbl in live_benchmark:
        clean_text = clean_tweet_advanced(text)
        x_tfidf = hybrid_vectorizer.transform([clean_text])
        x_vader = csr_matrix(extract_vader_features([clean_text]))
        x_fused = hstack([x_tfidf, x_vader]).tocsr()
        p = float(champ_model.predict_proba(x_fused)[0, 1])
        pred_lbl = 1 if p >= 0.5 else 0
        is_ok = (pred_lbl == true_lbl)
        if is_ok:
            live_correct += 1
        tag = "CORRECT [PASS]" if is_ok else "WRONG [FAIL]"
        print(f"  \"{text[:45]:<45}\" -> P(Pos): {p:.3f} | {tag}")
    print(f"  -> Live Benchmark Accuracy: {(live_correct/len(live_benchmark))*100:.1f}% ({live_correct}/{len(live_benchmark)})")

    # 11. Model Serialization (Matches Notebook Step 12)
    os.makedirs(args.output_dir, exist_ok=True)
    champ_model_path = os.path.join(args.output_dir, "sentiment140_champion_model.joblib")
    champ_vec_path = os.path.join(args.output_dir, "sentiment140_hybrid_vectorizer.joblib")
    compat_model_path = os.path.join(args.output_dir, "sentiment_model.joblib")
    compat_vec_path = os.path.join(args.output_dir, "tfidf_vectorizer.joblib")
    stack_lgb_path = os.path.join(args.output_dir, "sentiment140_stacking_lgb.joblib")
    stack_xgb_path = os.path.join(args.output_dir, "sentiment140_stacking_xgb.joblib")

    print(f"\n[STEP 12] Serializing model artifacts to '{args.output_dir}/' ...")
    dump(champ_model, champ_model_path)
    dump(hybrid_vectorizer, champ_vec_path)
    dump(champ_model, compat_model_path)
    dump(hybrid_vectorizer, compat_vec_path)
    dump(lgb_meta, stack_lgb_path)
    dump(xgb_meta, stack_xgb_path)

    print(f"  Saved Champion Model      : {champ_model_path} ({os.path.getsize(champ_model_path)/1024:.1f} KB)")
    print(f"  Saved Hybrid Vectorizer   : {champ_vec_path} ({os.path.getsize(champ_vec_path)/1024:.1f} KB)")
    print(f"  Saved Stacking LightGBM   : {stack_lgb_path} ({os.path.getsize(stack_lgb_path)/1024:.1f} KB)")
    print(f"  Saved Stacking XGBoost    : {stack_xgb_path} ({os.path.getsize(stack_xgb_path)/1024:.1f} KB)")

    print("\n" + "=" * 95)
    print("🎉 PIPELINE TRAINING, BENCHMARKING, AND SERIALIZATION COMPLETE!")
    print("=" * 95)

if __name__ == "__main__":
    cli_args = parse_args()
    run_end_to_end_training(cli_args)
