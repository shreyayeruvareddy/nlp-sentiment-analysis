# ============================================================
# config.py — NLP Sentiment Analysis (Amazon Product Reviews)
# ============================================================

RANDOM_SEED = 42

# Data Simulation
NUM_REVIEWS = 5000

# Product Categories
CATEGORIES = {
    "Electronics":   {"avg_rating": 3.8, "review_length": "long"},
    "Clothing":      {"avg_rating": 3.6, "review_length": "short"},
    "Home & Kitchen":{"avg_rating": 4.1, "review_length": "medium"},
    "Sports":        {"avg_rating": 4.0, "review_length": "medium"},
    "Beauty":        {"avg_rating": 4.2, "review_length": "short"},
}

# Sentiment Thresholds
# TextBlob polarity: -1.0 to 1.0
TEXTBLOB_POS_THRESHOLD  =  0.1
TEXTBLOB_NEG_THRESHOLD  = -0.1

# VADER compound: -1.0 to 1.0
VADER_POS_THRESHOLD     =  0.05
VADER_NEG_THRESHOLD     = -0.05

# ML Model
TEST_SIZE   = 0.20
CV_FOLDS    = 5

# Paths
RAW_DATA_PATH       = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH          = "models"
OUTPUT_PATH         = "outputs"
DB_PATH             = "data/sentiment_pipeline.db"
