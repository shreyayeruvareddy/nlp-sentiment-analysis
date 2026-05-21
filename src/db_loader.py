# ============================================================
# src/db_loader.py — Load sentiment results into SQLite
# ============================================================

import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id           TEXT PRIMARY KEY,
            category            TEXT,
            rating              INTEGER,
            sentiment_label     TEXT,
            review_title        TEXT,
            review_date         TEXT,
            review_month        TEXT,
            verified_purchase   INTEGER,
            helpful_votes       INTEGER,
            word_count          INTEGER,
            tb_polarity         REAL,
            tb_subjectivity     REAL,
            tb_sentiment        TEXT,
            vader_compound      REAL,
            vader_sentiment     TEXT,
            ensemble_sentiment  TEXT,
            pos_word_count      INTEGER,
            neg_word_count      INTEGER,
            tools_agree         INTEGER,
            ingested_at         TEXT
        );

        CREATE TABLE IF NOT EXISTS category_sentiment (
            cat_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            category            TEXT,
            total_reviews       INTEGER,
            avg_rating          REAL,
            avg_tb_polarity     REAL,
            avg_vader_compound  REAL,
            positive_count      INTEGER,
            neutral_count       INTEGER,
            negative_count      INTEGER,
            positive_pct        REAL,
            negative_pct        REAL,
            tools_agree_rate    REAL,
            created_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS method_comparison (
            comp_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            method          TEXT,
            accuracy_pct    REAL,
            notes           TEXT,
            evaluated_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS monthly_sentiment_trend (
            trend_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            review_month    TEXT,
            category        TEXT,
            total_reviews   INTEGER,
            avg_rating      REAL,
            avg_vader_compound REAL,
            positive_pct    REAL,
            negative_pct    REAL,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS pipeline_run_log (
            run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT,
            stage           TEXT,
            status          TEXT,
            records_processed INTEGER DEFAULT 0,
            duration_sec    REAL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("✅ Database schema created/verified")


def load_reviews(df: pd.DataFrame) -> int:
    conn = get_connection()
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    inserted = 0
    cols = ["review_id","category","rating","sentiment_label","review_title",
            "review_date","review_month","verified_purchase","helpful_votes","word_count",
            "tb_polarity","tb_subjectivity","tb_sentiment","vader_compound","vader_sentiment",
            "ensemble_sentiment","pos_word_count","neg_word_count","tools_agree"]

    for _, r in df[[c for c in cols if c in df.columns]].iterrows():
        try:
            placeholders = ",".join(["?"]*len(cols))
            values = [r.get(c) for c in cols] + [now]
            conn.execute(
                f"INSERT OR IGNORE INTO reviews ({','.join(cols)},ingested_at) VALUES ({placeholders},?)",
                values
            )
            inserted += 1
        except: pass

    conn.commit()
    conn.close()
    logger.info(f"✅ Inserted {inserted:,} reviews")
    return inserted


def load_category_sentiment(cat_df: pd.DataFrame):
    conn = get_connection()
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for _, r in cat_df.iterrows():
        conn.execute("""
            INSERT INTO category_sentiment
            (category, total_reviews, avg_rating, avg_tb_polarity, avg_vader_compound,
             positive_count, neutral_count, negative_count, positive_pct, negative_pct,
             tools_agree_rate, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (r.category, int(r.total_reviews), float(r.avg_rating),
              float(r.avg_tb_polarity), float(r.avg_vader_compound),
              int(r.positive_count), int(r.neutral_count), int(r.negative_count),
              float(r.positive_pct), float(r.negative_pct), float(r.tools_agree_rate), now))
    conn.commit()
    conn.close()
    logger.info(f"✅ Category sentiment loaded")


def load_method_comparison(comparison_df: pd.DataFrame):
    conn = get_connection()
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for _, r in comparison_df.iterrows():
        conn.execute("INSERT INTO method_comparison (method, accuracy_pct, notes, evaluated_at) VALUES (?,?,?,?)",
                     (r.method, float(r.accuracy_pct), r.notes, now))
    conn.commit()
    conn.close()
    logger.info(f"✅ Method comparison loaded")


def load_monthly_trends(df: pd.DataFrame):
    conn = get_connection()
    now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    monthly = df.groupby(["review_month", "category"]).agg(
        total_reviews      = ("review_id",      "count"),
        avg_rating         = ("rating",         "mean"),
        avg_vader_compound = ("vader_compound", "mean"),
        positive_pct       = ("sentiment_label", lambda x: (x=="Positive").mean()*100),
        negative_pct       = ("sentiment_label", lambda x: (x=="Negative").mean()*100),
    ).reset_index().round(2)

    for _, r in monthly.iterrows():
        conn.execute("""
            INSERT INTO monthly_sentiment_trend
            (review_month, category, total_reviews, avg_rating, avg_vader_compound,
             positive_pct, negative_pct, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (r.review_month, r.category, int(r.total_reviews), float(r.avg_rating),
              float(r.avg_vader_compound), float(r.positive_pct), float(r.negative_pct), now))

    conn.commit()
    conn.close()
    logger.info(f"✅ Monthly trends loaded")


def log_run(stage, status, records=0, duration=None):
    conn = get_connection()
    conn.execute("INSERT INTO pipeline_run_log VALUES (?,?,?,?,?,?)",
        (None, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), stage, status, records, duration))
    conn.commit()
    conn.close()


def query_summary() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT
            category,
            COUNT(*)                                         AS total_reviews,
            ROUND(AVG(rating), 2)                           AS avg_rating,
            ROUND(AVG(vader_compound), 4)                   AS avg_sentiment_score,
            SUM(CASE WHEN sentiment_label='Positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment_label='Neutral'  THEN 1 ELSE 0 END) AS neutral,
            SUM(CASE WHEN sentiment_label='Negative' THEN 1 ELSE 0 END) AS negative,
            ROUND(AVG(tools_agree)*100, 1)                  AS tool_agreement_pct
        FROM reviews
        GROUP BY category
        ORDER BY avg_sentiment_score DESC
    """, conn)
    conn.close()
    return df


def run_db_load(df, cat_df, comparison_df):
    import time
    t = time.time()
    try:
        create_tables()
        n = load_reviews(df)
        load_category_sentiment(cat_df)
        load_method_comparison(comparison_df)
        load_monthly_trends(df)
        duration = round(time.time()-t, 2)
        log_run("db_load", "SUCCESS", n, duration)
        logger.info(f"✅ DB load complete in {duration}s")
    except Exception as e:
        log_run("db_load", "FAILED")
        logger.error(f"❌ DB load failed: {e}")
        raise
