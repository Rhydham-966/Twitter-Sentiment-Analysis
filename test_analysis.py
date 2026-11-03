import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sentiment_utils import (
    clean_tweet_advanced,
    predict_from_texts,
    explain_text,
    plot_sentiment_bar,
    wordcloud_for_sentiment
)

def run_tests():
    print("=" * 85)
    print("[TEST] RUNNING END-TO-END SENTIMENT PIPELINE VERIFICATION TEST")
    print("=" * 85)

    # 1. Test Advanced Normalization
    print("\n[1/4] Testing Social Media Normalization...")
    raw_sample = "OMG!! @bestfriend I can't believe it... #Awesome http://test.com &amp; fun!"
    cleaned = clean_tweet_advanced(raw_sample)
    print(f"  Raw     : {raw_sample}")
    print(f"  Cleaned : {cleaned}")
    assert "http" not in cleaned, "URL was not removed!"
    assert "@" not in cleaned, "Handle was not removed!"
    assert "not" in cleaned, "Negation contraction was not normalized!"
    print("  -> Normalization: PASS [OK]")

    # 2. Test 10 Diverse Real-World Customer Statements
    print("\n[2/4] Testing Zero-Shot Sentiment Prediction on Real-World Customer Reviews...")
    benchmark_samples = [
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

    texts = [item[0] for item in benchmark_samples]
    ground_truth = [item[1] for item in benchmark_samples]

    df_results = predict_from_texts(texts, threshold=0.60)
    
    correct_count = 0
    print("-" * 85)
    print(f"{'Input Text':<50} | {'Sentiment':<12} | {'Prob':<6} | {'Status'}")
    print("-" * 85)

    for i, row in df_results.iterrows():
        is_correct = (row["label"] == ground_truth[i])
        if is_correct:
            correct_count += 1
        status_tag = "PASS [OK]" if is_correct else "FAIL [X]"
        text_disp = (row["text"][:47] + "...") if len(row["text"]) > 50 else row["text"]
        print(f"{text_disp:<50} | {row['sentiment']:<12} | {row['prob_pos']:<6.3f} | {status_tag}")

    acc = (correct_count / len(benchmark_samples)) * 100
    print("-" * 85)
    print(f"  -> Accuracy Score: {acc:.1f}% ({correct_count}/{len(benchmark_samples)} correct)")
    assert acc >= 90.0, f"Expected accuracy >= 90%, got {acc}%"

    # 3. Test Explainable AI Token Attribution
    print("\n[3/4] Testing Explainable AI (XAI) Token Attribution...")
    test_statement = "The customer service was amazing and super helpful!"
    dec, prob, pos_tokens, neg_tokens, status = explain_text(test_statement, top_k=5)
    print(f"  Statement   : \"{test_statement}\"")
    print(f"  Prediction  : {dec} (Prob Pos: {prob:.4f}) | Status: {status}")
    print(f"  Top Positive: {pos_tokens[:3]}")
    assert len(pos_tokens) > 0, "Expected positive token drivers!"
    print("  -> Explainability: PASS [OK]")

    # 4. Test Visualizations (Plot & WordCloud)
    print("\n[4/5] Testing Visualization Generators...")
    os.makedirs("scratch", exist_ok=True)
    fig_bar = plot_sentiment_bar(df_results)
    fig_bar.savefig("scratch/test_bar.png")
    plt.close(fig_bar)

    fig_pos_wc = wordcloud_for_sentiment(df_results, "Positive")
    if fig_pos_wc:
        fig_pos_wc.savefig("scratch/test_pos_wc.png")
        plt.close(fig_pos_wc)

    fig_neg_wc = wordcloud_for_sentiment(df_results, "Negative")
    if fig_neg_wc:
        fig_neg_wc.savefig("scratch/test_neg_wc.png")
        plt.close(fig_neg_wc)

    print("  -> Charts and WordClouds generated and saved cleanly!")

    # 5. Test Emoji & Emoticon Sentiment Handling
    print("\n[5/5] Testing Unicode Emoji & Emoticon Sentiment Handling...")
    emoji_cases = [
        ("I love this phone 😊🔥", 1),
        ("Worst service ever 😡💔", 0),
        ("❤️🎉", 1),
        ("Great job on this project! :-)", 1),
        ("Disappointed and unhappy :-(", 0)
    ]
    df_emoji = predict_from_texts([c[0] for c in emoji_cases], threshold=0.55)
    for idx, (text_raw, true_val) in enumerate(emoji_cases):
        pred_val = df_emoji.iloc[idx]["label"]
        pred_sent = df_emoji.iloc[idx]["sentiment"]
        prob_val = df_emoji.iloc[idx]["prob_pos"]
        assert pred_val == true_val, f"Emoji test failed for: {text_raw}"
        print(f"  Emoji Sample: {text_raw} -> {pred_sent} ({prob_val:.3f}) [PASS]")
    print("  -> Emoji & Emoticon Handling: PASS [OK]")

    print("\n" + "=" * 85)
    print("[SUCCESS] ALL 5 PIPELINE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 85)

if __name__ == "__main__":
    run_tests()
