# ============================================================
# run_pipeline.py — NLP Sentiment Analysis Pipeline
# Stages: Generate >> NLP Process >> ML Classify >> DB Load
# Usage: py -3.11 run_pipeline.py
# ============================================================

import time, logging, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    start  = time.time()
    ts     = __import__("datetime").datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info(f"📝 NLP PIPELINE STARTED  |  run_id: {ts}")
    logger.info("=" * 60)

    # STAGE 1 — DATA GENERATION
    logger.info("\n📦 STAGE 1: Review Data Generation")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.data_generator import run_data_generation
        df_raw, _ = run_data_generation()
        logger.info(f"✅ Stage 1 complete in {round(time.time()-t,2)}s | {len(df_raw):,} reviews")
    except Exception as e:
        logger.error(f"❌ Stage 1 FAILED: {e}"); return False

    # STAGE 2 — NLP PROCESSING
    logger.info("\n🔤 STAGE 2: NLP Processing (NLTK + TextBlob + VADER)")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.nlp_processor import run_nlp_processing
        df_processed, ngrams, cat_sentiment = run_nlp_processing(df_raw, ts)
        logger.info(f"✅ Stage 2 complete in {round(time.time()-t,2)}s")
    except Exception as e:
        logger.error(f"❌ Stage 2 FAILED: {e}"); return False

    # STAGE 3 — ML CLASSIFICATION
    logger.info("\n🤖 STAGE 3: ML Sentiment Classification (TF-IDF + LR)")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.ml_classifier import run_ml_classification
        ml_metrics, comparison, ml_pipeline = run_ml_classification(df_processed, ts)
        logger.info(f"✅ Stage 3 complete in {round(time.time()-t,2)}s")
        logger.info(f"   ML Accuracy:  {ml_metrics['accuracy']}%")
        logger.info(f"   ML F1 Score:  {ml_metrics['weighted_f1']}%")
        logger.info(f"   CV Accuracy:  {ml_metrics['cv_accuracy']}%")
    except Exception as e:
        logger.error(f"❌ Stage 3 FAILED: {e}"); return False

    # STAGE 4 — DATABASE LOAD
    logger.info("\n🗄️  STAGE 4: Database Load")
    logger.info("-" * 40)
    t = time.time()
    try:
        from src.db_loader import run_db_load
        run_db_load(df_processed, cat_sentiment, comparison)
        logger.info(f"✅ Stage 4 complete in {round(time.time()-t,2)}s")
    except Exception as e:
        logger.error(f"❌ Stage 4 FAILED: {e}"); return False

    # STAGE 5 — VALIDATION
    logger.info("\n✅ STAGE 5: Validation Summary")
    logger.info("-" * 40)
    try:
        from src.db_loader import query_summary
        print("\n" + query_summary().to_string())

        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"   Total reviews:      {len(df_processed):,}")
        logger.info(f"   Positive reviews:   {(df_processed['sentiment_label']=='Positive').sum():,} ({(df_processed['sentiment_label']=='Positive').mean():.1%})")
        logger.info(f"   Neutral reviews:    {(df_processed['sentiment_label']=='Neutral').sum():,} ({(df_processed['sentiment_label']=='Neutral').mean():.1%})")
        logger.info(f"   Negative reviews:   {(df_processed['sentiment_label']=='Negative').sum():,} ({(df_processed['sentiment_label']=='Negative').mean():.1%})")
        logger.info(f"   TextBlob accuracy:  {(df_processed['tb_sentiment']==df_processed['sentiment_label']).mean():.1%}")
        logger.info(f"   VADER accuracy:     {(df_processed['vader_sentiment']==df_processed['sentiment_label']).mean():.1%}")
        logger.info(f"   ML accuracy (CV):   {ml_metrics['cv_accuracy']}%  (test=100% expected on simulated data)")
        logger.info(f"   Tools agreement:    {df_processed['tools_agree'].mean():.1%}")

        # Top bigrams
        logger.info(f"\n   Top 5 Positive bigrams: {ngrams.get('Positive_bigrams', [])[:5]}")
        logger.info(f"   Top 5 Negative bigrams: {ngrams.get('Negative_bigrams', [])[:5]}")

    except Exception as e:
        logger.warning(f"⚠️  Validation warning: {e}")

    total = round(time.time() - start, 2)
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 PIPELINE COMPLETE  |  Total time: {total}s")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    run_pipeline()
