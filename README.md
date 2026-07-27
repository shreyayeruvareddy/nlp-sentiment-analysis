# NLP Sentiment Analysis — Amazon Product Reviews

> End-to-end NLP pipeline analyzing 5,000 Amazon product reviews across 5 categories using NLTK preprocessing, TextBlob polarity scoring, VADER compound scoring, and TF-IDF + Logistic Regression — with n-gram analysis, trend detection, and an interactive Tableau dashboard.

---

## Architecture

```
Review Generation (5,000 reviews, 5 categories)
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
```

---

## NLP Results

| Method | Accuracy |
|---|---|
| TextBlob | 87.5% |
| VADER | 81.8% |
| Tools Agreement | 79.8% |

---

## Key Insights

| Category | VADER Score | Sentiment |
|---|---|---|
| Beauty | 0.4549 | Best |
| Home & Kitchen | 0.4292 | Strong |
| Sports | 0.3966 | Good |
| Electronics | 0.3055 | Average |
| Clothing | 0.1870 | Worst |

- **Clothing** has highest negative review rate (41.2%) — quality concerns
- **Beauty** most positively reviewed — strong brand satisfaction
- Top positive bigrams: "highly recommend", "really good"
- Top negative bigrams: "waste money", "poor quality", "not buy"

---

## 📊 Tableau Dashboard

**Live Dashboard:** [NLP Sentiment Analysis — Amazon Reviews](https://public.tableau.com/app/profile/bala.shreya.reddy.yeruva/viz/NLPSentimentAnalysisAmazonReviews/NLPSentimentAnalysisAmazonReviewsYeruvaBalaShreyaReddy)

Dashboard includes:
- **Sentiment by Category** (stacked bar) — Positive/Neutral/Negative breakdown
- **VADER Score by Category** (bar) — Beauty best (0.45), Clothing worst (0.19)
- **Rating Distribution** (pie chart) — 5,000 reviews across 1-5 star ratings
- **TextBlob vs VADER** (scatter) — correlation between two NLP tools per category

---

## Tech Stack

| Layer | Technology |
|---|---|
| NLP | NLTK 3.9, TextBlob 0.20, VADER 3.3 |
| ML | TF-IDF + Logistic Regression (scikit-learn) |
| Database | SQLite |
| Visualization | Tableau Dashboard |
| Version Control | Git / GitHub |

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
