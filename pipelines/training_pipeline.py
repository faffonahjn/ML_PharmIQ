"""
pipelines/training_pipeline.py
End-to-end training pipeline: ingest → clean → feature engineer → train → evaluate → serialize.
Run: python pipelines/training_pipeline.py
"""

import sys
import logging
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "training.log", mode="w"),
    ],
)
logger = logging.getLogger("pipeline")


def run():
    start = time.time()
    logger.info("=" * 60)
    logger.info("PharmIQ Training Pipeline — START")
    logger.info("=" * 60)

    # Step 1: Ingest
    logger.info("[1/4] Ingesting raw data...")
    from src.data.ingest import load_raw, clean, validate
    df_raw = load_raw(PROJECT_ROOT / "data" / "raw" / "tata_1mg_Medicine_data.csv")
    df_clean = clean(df_raw)
    validate(df_clean)

    # Step 2: Feature engineering
    logger.info("[2/4] Engineering features...")
    from src.features.engineer import engineer, FEATURE_COLS, TARGET_COL
    df = engineer(df_clean)

    # Save processed data
    processed_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved -> {processed_path}")

    # Step 3: Train
    logger.info("[3/4] Training model...")
    from src.models.train import train
    pipeline, metrics = train()

    # Step 4: Evaluate
    logger.info("[4/4] Evaluation complete.")
    logger.info(f"  AUC OvR Macro : {metrics['auc_ovr_macro']:.4f}")
    logger.info(f"  Test Accuracy : {metrics['test_accuracy']:.4f}")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    run()
