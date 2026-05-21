# NLP Sentiment Analysis — Amazon Product Reviews

> End-to-end NLP pipeline analyzing 5,000 Amazon product reviews across 5 categories using NLTK preprocessing, TextBlob polarity scoring, VADER compound scoring, and a TF-IDF + Logistic Regression ML classifier — with n-gram analysis, trend detection, and Tableau-ready exports.

---

## Project Overview

This project builds a complete sentiment analysis system for e-commerce product reviews. It simulates 5,000 Amazon reviews across Electronics, Clothing, Home & Kitchen, Sports, and Beauty categories, applies four sentiment analysis approaches, compares their accuracy, and delivers actionable business intelligence about customer sentiment trends.

---

## Architecture

```
Review Generation (5,000 Amazon reviews, 5 categories)
        |
        v
[ Stage 1: Generate     ]  src/data_generator.py  → realistic review text
        |
        v
[ Stage 2: NLP Process  ]  src/nlp_processor.py   → NLTK + TextBlob + VADER
        |
        v
[ Stage 3: ML Classify  ]  src/ml_classifier.py   → TF-IDF + Logistic Regression
        |
        v
[ Stage 4: DB Load      ]  src/db_loader.py       → SQLite + monthly trends
        |
        v
[ Stage 5: Validate     ]  Query summary          → Tableau CSV export
```

---

## NLP Methods Compared

| Method | Type | Strength |
|---|---|---|
| NLTK | Preprocessing | Tokenization, stopwords, stemming, lemmatization |
| TextBlob | Lexicon-based | Polarity (-1 to 1) + Subjectivity (0 to 1) |
| VADER | Rule-based | Compound score, optimized for review/social text |
| TF-IDF + LR | ML-based | Learns from labeled data, highest accuracy |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| NLP | NLTK 3.9, TextBlob 0.20, VADER 3.3 |
| ML | scikit-learn 1.8 (TF-IDF + Logistic Regression) |
| Database | SQLite |
| Visualization | CSV export → Tableau / Power BI |
| Version Control | Git / GitHub |

---

## Key Features

- Text preprocessing: tokenization, stopword removal, lemmatization (NLTK)
- Polarity and subjectivity scoring (TextBlob)
- Compound sentiment scoring (VADER)
- TF-IDF vectorization with bigrams (1,2)-gram range
- N-gram analysis: top bigrams/trigrams per sentiment class
- Monthly sentiment trend analysis by category
- Ensemble voting: TextBlob + VADER majority

---

## Setup & Run

```bash
git clone https://github.com/shreyayeruvareddy/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
py -3.11 -m pip install -r requirements.txt
py -3.11 run_pipeline.py
```

---

## Author

**Yeruva Bala Shreya Reddy**
M.S. Computer Science (Data Science) — UNC Charlotte
[GitHub](https://github.com/shreyayeruvareddy) | [Email](mailto:yeruvabalashreyareddy@gmail.com)
