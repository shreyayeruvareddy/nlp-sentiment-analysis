# ============================================================
# src/nlp_processor.py — Text preprocessing + NLP features
# Tools: NLTK, TextBlob, VADER
# ============================================================

import pandas as pd
import numpy as np
import re
import os
import logging
import nltk
from nltk.tokenize    import word_tokenize, sent_tokenize
from nltk.corpus      import stopwords
from nltk.stem        import PorterStemmer, WordNetLemmatizer
from nltk.util        import ngrams
from textblob         import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections      import Counter
from config import PROCESSED_DATA_PATH, TEXTBLOB_POS_THRESHOLD, TEXTBLOB_NEG_THRESHOLD, VADER_POS_THRESHOLD, VADER_NEG_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK data
def download_nltk_data():
    packages = ["punkt", "stopwords", "wordnet", "punkt_tab", "averaged_perceptron_tagger"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass
    logger.info("✅ NLTK data downloaded")

download_nltk_data()


# ── Text Preprocessing ────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove special characters, URLs, extra whitespace."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # Remove URLs
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)         # Keep letters + basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()             # Normalize whitespace
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    try:
        return word_tokenize(text)
    except:
        return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords."""
    stop_words = set(stopwords.words("english"))
    # Keep negation words — important for sentiment
    keep_words = {"not", "no", "never", "neither", "nor", "none",
                  "without", "nothing", "hardly", "barely", "scarcely"}
    stop_words = stop_words - keep_words
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter stemming."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply WordNet lemmatization (better than stemming for NLP)."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> dict:
    """Full preprocessing pipeline for one review."""
    cleaned    = clean_text(text)
    tokens     = tokenize(cleaned)
    no_stop    = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(no_stop)

    return {
        "cleaned_text":     cleaned,
        "tokens":           tokens,
        "filtered_tokens":  no_stop,
        "lemmatized":       lemmatized,
        "processed_text":   " ".join(lemmatized),
        "token_count":      len(tokens),
        "unique_tokens":    len(set(no_stop)),
        "sentence_count":   len(sent_tokenize(text)) if text else 0,
    }


# ── Sentiment Analysis Tools ──────────────────────────────────

def get_textblob_scores(text: str) -> dict:
    """
    TextBlob sentiment analysis.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    blob = TextBlob(str(text))
    polarity     = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity > TEXTBLOB_POS_THRESHOLD:
        label = "Positive"
    elif polarity < TEXTBLOB_NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "tb_polarity":     polarity,
        "tb_subjectivity": subjectivity,
        "tb_sentiment":    label,
    }


def get_vader_scores(text: str) -> dict:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) analysis.
    Best tool for short social/review text.
    Returns: positive, negative, neutral, compound scores.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores   = analyzer.polarity_scores(str(text))

    compound = scores["compound"]
    if compound >= VADER_POS_THRESHOLD:
        label = "Positive"
    elif compound <= VADER_NEG_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "vader_positive": round(scores["pos"], 4),
        "vader_negative": round(scores["neg"], 4),
        "vader_neutral":  round(scores["neu"], 4),
        "vader_compound": round(compound, 4),
        "vader_sentiment": label,
    }


def get_ngrams(tokens: list[str], n: int = 2) -> list[tuple]:
    """Extract n-grams from token list."""
    return list(ngrams(tokens, n))


def extract_features(text: str, tokens: list[str]) -> dict:
    """Extract additional NLP features for ML model."""
    # Exclamation marks (enthusiasm)
    exclamation_count = text.count("!")
    question_count    = text.count("?")

    # Negation presence
    negation_words = {"not", "no", "never", "don't", "doesn't", "won't",
                      "can't", "terrible", "awful", "horrible", "worst"}
    has_negation = int(any(w in tokens for w in negation_words))

    # Positive/negative word counts
    positive_words = {"great", "excellent", "amazing", "love", "perfect",
                      "fantastic", "wonderful", "awesome", "best", "outstanding"}
    negative_words = {"terrible", "awful", "horrible", "worst", "bad",
                      "poor", "disappointed", "broken", "waste", "useless"}

    pos_word_count = sum(1 for w in tokens if w in positive_words)
    neg_word_count = sum(1 for w in tokens if w in negative_words)

    return {
        "exclamation_count": exclamation_count,
        "question_count":    question_count,
        "has_negation":      has_negation,
        "pos_word_count":    pos_word_count,
        "neg_word_count":    neg_word_count,
        "pos_neg_ratio":     round((pos_word_count + 1) / (neg_word_count + 1), 4),
    }


# ── Main Processing Pipeline ──────────────────────────────────

def process_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full NLP pipeline to all reviews:
    1. Text preprocessing (NLTK)
    2. TextBlob sentiment scoring
    3. VADER sentiment scoring
    4. Feature extraction
    5. N-gram extraction
    """
    logger.info(f"🔤 Processing {len(df):,} reviews with NLP pipeline...")
    results = []

    for _, row in df.iterrows():
        text = str(row["review_text"])

        # Step 1: Preprocessing
        prep = preprocess_text(text)

        # Step 2: TextBlob
        tb = get_textblob_scores(text)

        # Step 3: VADER
        vader = get_vader_scores(text)

        # Step 4: Features
        feats = extract_features(prep["cleaned_text"], prep["filtered_tokens"])

        # Combine all scores
        result = {**row.to_dict(), **prep, **tb, **vader, **feats}
        results.append(result)

    df_processed = pd.DataFrame(results)

    # Agreement between TextBlob and VADER
    df_processed["tools_agree"] = (
        df_processed["tb_sentiment"] == df_processed["vader_sentiment"]
    ).astype(int)

    # Ensemble sentiment (majority vote: TextBlob + VADER + rating-based)
    df_processed["ensemble_sentiment"] = df_processed.apply(
        _ensemble_sentiment, axis=1
    )

    logger.info(f"✅ NLP processing complete: {df_processed.shape[1]} total columns")
    logger.info(f"   TextBlob accuracy vs label: {(df_processed['tb_sentiment'] == df_processed['sentiment_label']).mean():.1%}")
    logger.info(f"   VADER accuracy vs label:    {(df_processed['vader_sentiment'] == df_processed['sentiment_label']).mean():.1%}")
    logger.info(f"   Tools agreement rate:       {df_processed['tools_agree'].mean():.1%}")

    return df_processed


def _ensemble_sentiment(row) -> str:
    """Ensemble: majority vote between TextBlob and VADER."""
    votes = [row["tb_sentiment"], row["vader_sentiment"]]
    counts = Counter(votes)
    return counts.most_common(1)[0][0]


def compute_ngram_analysis(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Compute top bigrams and trigrams by sentiment."""
    results = {}

    for sentiment in ["Positive", "Negative", "Neutral"]:
        subset = df[df["sentiment_label"] == sentiment]
        all_tokens = []
        for tokens in subset["filtered_tokens"]:
            if isinstance(tokens, list):
                all_tokens.extend(tokens)
            elif isinstance(tokens, str):
                all_tokens.extend(tokens.split())

        bigrams  = Counter(ngrams(all_tokens, 2)).most_common(top_n)
        trigrams = Counter(ngrams(all_tokens, 3)).most_common(top_n)

        results[f"{sentiment}_bigrams"]  = [(" ".join(bg), cnt) for bg, cnt in bigrams]
        results[f"{sentiment}_trigrams"] = [(" ".join(tg), cnt) for tg, cnt in trigrams]

    logger.info("✅ N-gram analysis complete")
    return results


def compute_category_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Sentiment breakdown by product category."""
    agg = df.groupby("category").agg(
        total_reviews    = ("review_id",      "count"),
        avg_rating       = ("rating",         "mean"),
        avg_tb_polarity  = ("tb_polarity",    "mean"),
        avg_vader_compound = ("vader_compound","mean"),
        positive_count   = ("sentiment_label", lambda x: (x == "Positive").sum()),
        neutral_count    = ("sentiment_label", lambda x: (x == "Neutral").sum()),
        negative_count   = ("sentiment_label", lambda x: (x == "Negative").sum()),
        avg_subjectivity = ("tb_subjectivity","mean"),
        tools_agree_rate = ("tools_agree",    "mean"),
    ).reset_index().round(4)

    agg["positive_pct"] = (agg["positive_count"] / agg["total_reviews"] * 100).round(1)
    agg["negative_pct"] = (agg["negative_count"] / agg["total_reviews"] * 100).round(1)
    agg = agg.sort_values("avg_vader_compound", ascending=False)

    logger.info(f"📊 Category sentiment analysis:\n{agg[['category','avg_rating','positive_pct','negative_pct','avg_vader_compound']].to_string()}")
    return agg


def save_processed(df: pd.DataFrame, ts: str) -> str:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    # Drop list columns before saving
    save_cols = [c for c in df.columns if not isinstance(df[c].iloc[0], list)]
    path = os.path.join(PROCESSED_DATA_PATH, f"reviews_processed_{ts}.csv")
    df[save_cols].to_csv(path, index=False)
    logger.info(f"💾 Processed data → {path}")
    return path


def run_nlp_processing(df: pd.DataFrame, ts: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df_processed = process_reviews(df)
    ngram_results = compute_ngram_analysis(df_processed)
    category_sentiment = compute_category_sentiment(df_processed)
    save_processed(df_processed, ts)
    return df_processed, ngram_results, category_sentiment
