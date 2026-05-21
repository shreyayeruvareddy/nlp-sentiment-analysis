# ============================================================
# src/data_generator.py — Simulate Amazon product reviews
# 5,000 reviews across 5 categories with realistic text
# ============================================================

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from config import NUM_REVIEWS, CATEGORIES, RAW_DATA_PATH, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

# ── Review templates by sentiment ─────────────────────────────

POSITIVE_PHRASES = [
    "Absolutely love this product!", "Exceeded my expectations completely.",
    "Best purchase I've made this year.", "Works perfectly right out of the box.",
    "Highly recommend to everyone.", "Amazing quality for the price.",
    "Fast shipping and great packaging.", "Exactly as described, very happy.",
    "Outstanding product, will buy again.", "Five stars without hesitation.",
    "Great value for money.", "Impressed with the build quality.",
    "Super easy to use and setup.", "Delivered on time and in perfect condition.",
    "My whole family loves it.", "Worth every penny spent.",
    "Solid construction and great design.", "Very satisfied with this purchase.",
    "Works like a charm, no issues.", "Fantastic product, can't go wrong.",
]

NEUTRAL_PHRASES = [
    "Product is okay, nothing special.", "Does what it's supposed to do.",
    "Average quality for the price.", "Some pros and cons to consider.",
    "Meets basic expectations, nothing more.", "Decent product but could be better.",
    "Works fine for occasional use.", "Not bad but not great either.",
    "Pretty standard product overall.", "Acceptable quality at this price point.",
    "It gets the job done I suppose.", "Neither impressed nor disappointed.",
    "Would have preferred better packaging.", "Functional but lacks premium feel.",
    "Average experience, nothing to write home about.",
]

NEGATIVE_PHRASES = [
    "Very disappointed with this product.", "Stopped working after just two weeks.",
    "Quality is much worse than expected.", "Would not recommend to anyone.",
    "Complete waste of money.", "Arrived damaged and poorly packaged.",
    "Does not match the description at all.", "Terrible customer service experience.",
    "Returned immediately, very unsatisfied.", "Cheaply made and falls apart quickly.",
    "Misleading photos and description.", "Broke after the first use.",
    "Horrible quality, avoid at all costs.", "Nothing like what was advertised.",
    "One star is too generous for this.", "Regret buying this completely.",
    "Worst product I've ever purchased.", "Do not buy, total disappointment.",
]

POSITIVE_DETAILS = {
    "Electronics":    ["Battery life is excellent.", "Screen resolution is stunning.", "Connects seamlessly to all devices.", "Processing speed is impressive.", "Sound quality is crystal clear."],
    "Clothing":       ["Fits perfectly as expected.", "Material is soft and comfortable.", "Color matches exactly as shown.", "Washes well without fading.", "True to size and very flattering."],
    "Home & Kitchen": ["Easy to clean and maintain.", "Fits perfectly in my kitchen.", "Durable and well-constructed.", "Heats up quickly and evenly.", "Space-saving design is genius."],
    "Sports":         ["Lightweight and very comfortable.", "Great grip and traction.", "Holds up well during intense workouts.", "Perfect for both beginners and pros.", "Adjustable and easy to customize."],
    "Beauty":         ["Skin feels amazing after use.", "Scent is subtle and pleasant.", "Long-lasting results as promised.", "Gentle on sensitive skin.", "Worth every dollar spent on this."],
}

NEGATIVE_DETAILS = {
    "Electronics":    ["Battery drains extremely fast.", "Screen has dead pixels already.", "Bluetooth keeps disconnecting.", "Overheats after 30 minutes.", "Sound is tinny and distorted."],
    "Clothing":       ["Runs very small, size up.", "Fabric is scratchy and uncomfortable.", "Color faded after first wash.", "Stitching came apart immediately.", "Nothing like the photos online."],
    "Home & Kitchen": ["Handle broke on first use.", "Leaks badly even when sealed.", "Very hard to clean properly.", "Cheap plastic that warps easily.", "Doesn't fit standard cabinet sizes."],
    "Sports":         ["Sole peeled off after a week.", "Extremely uncomfortable after one hour.", "Poor quality stitching throughout.", "Much heavier than advertised.", "Broke during normal use."],
    "Beauty":         ["Caused skin irritation immediately.", "Smell is overwhelming and chemical.", "No results after two weeks.", "Packaging leaks and wastes product.", "Made skin worse not better."],
}

NEUTRAL_DETAILS = {
    "Electronics":    ["Battery life is average.", "Setup was straightforward.", "Works for basic tasks only.", "Build quality feels standard.", "Does what I needed it to."],
    "Clothing":       ["Sizing is slightly off.", "Material is decent quality.", "Color is close to photos.", "Normal wear and tear expected.", "Standard quality for price."],
    "Home & Kitchen": ["Average build quality.", "Works as a basic kitchen tool.", "Nothing special but functional.", "Easy enough to assemble.", "Gets the job done adequately."],
    "Sports":         ["Average comfort level.", "Works for light exercise.", "Standard quality for budget.", "Nothing exceptional about it.", "Adequate for occasional use."],
    "Beauty":         ["Average results after use.", "Scent is neither good nor bad.", "Noticeable but not dramatic.", "Works as described basically.", "Decent for the price point."],
}


def generate_review_text(sentiment: str, category: str, rating: int) -> tuple[str, str]:
    """Generate realistic review title and body based on sentiment and category."""
    if sentiment == "Positive":
        phrase  = np.random.choice(POSITIVE_PHRASES)
        detail  = np.random.choice(POSITIVE_DETAILS[category])
        # 30% chance of mixed signal (realistic reviews often have minor complaints)
        if np.random.random() < 0.30:
            mixed = np.random.choice(NEUTRAL_PHRASES)
            body  = f"{phrase} {detail} {mixed}"
        else:
            body  = f"{phrase} {detail}"
        title   = np.random.choice(["Great product!", "Love it!", "Excellent!", "Highly recommend!", "Amazing purchase!", "5 stars!", "Really good", "Happy with purchase"])

    elif sentiment == "Negative":
        phrase  = np.random.choice(NEGATIVE_PHRASES)
        detail  = np.random.choice(NEGATIVE_DETAILS[category])
        # 20% chance of mixed signal
        if np.random.random() < 0.20:
            mixed = np.random.choice(NEUTRAL_PHRASES)
            body  = f"{mixed} {phrase} {detail}"
        else:
            body  = f"{phrase} {detail}"
        title   = np.random.choice(["Very disappointed", "Waste of money", "Do not buy", "Terrible product", "1 star", "Awful", "Not worth it", "Poor quality"])

    else:  # Neutral
        # Mix positive and negative phrases for genuinely ambiguous reviews
        pos = np.random.choice(POSITIVE_PHRASES)
        neg = np.random.choice(NEGATIVE_PHRASES)
        detail = np.random.choice(NEUTRAL_DETAILS[category])
        body  = f"{np.random.choice(NEUTRAL_PHRASES)} {detail}"
        if np.random.random() < 0.5:
            body = f"{pos} However, {neg.lower()}"
        title   = np.random.choice(["It's okay", "Average product", "Nothing special", "Decent", "Mixed feelings", "3 stars", "Could be better", "Not sure"])

    # Add some filler noise words to make reviews less template-like
    fillers = ["I bought this last month.", "Ordered for my home.",
               "Got this as a gift.", "Been using it for a while.",
               "Tried several brands before this.", "My second purchase of this.",
               "Saw this on sale and decided to try.", ""]
    noise = np.random.choice(fillers)
    body = f"{noise} {body}".strip()

    return title, body


def rating_to_sentiment(rating: int) -> str:
    """Convert star rating to sentiment label."""
    if rating >= 4:   return "Positive"
    elif rating == 3: return "Neutral"
    else:             return "Negative"


def generate_reviews() -> pd.DataFrame:
    """Generate 5,000 realistic Amazon product reviews."""
    logger.info(f"📝 Generating {NUM_REVIEWS} Amazon product reviews...")

    categories = list(CATEGORIES.keys())
    records    = []
    start_date = datetime(2024, 1, 1)

    for i in range(1, NUM_REVIEWS + 1):
        category    = np.random.choice(categories)
        cat_info    = CATEGORIES[category]
        avg_rating  = cat_info["avg_rating"]

        # Generate rating with realistic distribution around category average
        rating_probs = {
            5: max(0.05, (avg_rating - 4) * 0.6 + 0.3),
            4: 0.30,
            3: 0.15,
            2: 0.10,
            1: max(0.05, (5 - avg_rating) * 0.15),
        }
        total = sum(rating_probs.values())
        rating_probs = {k: v/total for k, v in rating_probs.items()}
        rating = np.random.choice(list(rating_probs.keys()), p=list(rating_probs.values()))

        sentiment     = rating_to_sentiment(rating)
        title, body   = generate_review_text(sentiment, category, rating)

        # Review date — spread over 16 months
        days_offset   = np.random.randint(0, 480)
        review_date   = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        review_month  = (start_date + timedelta(days=days_offset)).strftime("%Y-%m")

        # Verified purchase
        verified      = np.random.choice([0, 1], p=[0.15, 0.85])

        # Helpful votes (positive reviews tend to get more)
        helpful_base  = {"Positive": 8, "Neutral": 3, "Negative": 5}
        helpful_votes = max(0, int(np.random.poisson(helpful_base[sentiment])))

        records.append({
            "review_id":      f"REV_{i:05d}",
            "category":       category,
            "rating":         rating,
            "sentiment_label": sentiment,
            "review_title":   title,
            "review_body":    body,
            "review_text":    title + " " + body,
            "review_date":    review_date,
            "review_month":   review_month,
            "verified_purchase": verified,
            "helpful_votes":  helpful_votes,
            "word_count":     len(body.split()),
        })

    df = pd.DataFrame(records)
    logger.info(f"✅ Generated {len(df):,} reviews")
    logger.info(f"   Sentiment: {df['sentiment_label'].value_counts().to_dict()}")
    logger.info(f"   Avg rating: {df['rating'].mean():.2f}")
    return df


def save_raw_data(df: pd.DataFrame) -> str:
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RAW_DATA_PATH, f"reviews_{ts}.csv")
    df.to_csv(path, index=False)
    logger.info(f"💾 Raw data saved → {path}")
    return path


def run_data_generation() -> tuple[pd.DataFrame, str]:
    df = generate_reviews()
    path = save_raw_data(df)
    return df, path


if __name__ == "__main__":
    df, _ = run_data_generation()
    print(f"\nShape: {df.shape}")
    print(f"\nSentiment distribution:\n{df['sentiment_label'].value_counts()}")
    print(f"\nBy category:\n{df.groupby('category')['rating'].mean().round(2)}")
