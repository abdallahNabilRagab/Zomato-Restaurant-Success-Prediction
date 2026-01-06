# ============================================================
# src/inference.py — Clean & Production-Safe Inference Pipeline with Extended Metrics
# ============================================================
import sys
import os
import glob
import logging
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)
import joblib

# Ensure src folder is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline runner
try:
    from data_pipeline import run_pipeline
except ImportError:
    raise ImportError("❌ run_pipeline not found in data_pipeline.py. Ensure the function exists.")

# Logger setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("InferencePipeline")

# ============================================================
# Helper: get the latest file in a folder
# ============================================================
def get_latest_file(folder_path: str, pattern: str = "*"):
    files = glob.glob(os.path.join(folder_path, pattern))
    return max(files, key=os.path.getctime) if files else None

# ============================================================
# Load the latest test dataset (auto-run pipeline if missing)
# ============================================================
def load_latest_test(preprocessed_folder: str, raw_path: str, preprocessed_path: str):
    test_path = get_latest_file(preprocessed_folder, pattern="test_*.csv")

    if test_path is None:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"❌ Raw data not found at {raw_path}, cannot create test data.")

        logger.info("⚠ Test data missing → Running full pipeline to generate it...")
        run_pipeline(raw_path=raw_path, save_path=preprocessed_path, split_data=True)
        test_path = get_latest_file(preprocessed_folder, pattern="test_*.csv")

    logger.info(f"✔ Loading test dataset → {test_path}")
    return pd.read_csv(test_path)

# ============================================================
# Load the latest preprocessing pipeline
# ============================================================
def load_latest_processor(models_folder: str):
    processor_path = get_latest_file(models_folder, pattern="processor_*.pkl")
    if processor_path is None:
        raise FileNotFoundError("❌ No saved processor found! Train the models first.")
    logger.info(f"✔ Loading processor → {processor_path}")
    return joblib.load(processor_path)

# ============================================================
# Load ALL trained models previously saved
# ============================================================
def load_saved_models(models_folder: str):
    model_files = glob.glob(os.path.join(models_folder, "*_best_*.pkl"))

    if not model_files:
        raise FileNotFoundError(
            "❌ No trained models found in saved_models folder!\n"
            "➡ Run train_models.py first to save trained models."
        )

    models = {}
    for file in model_files:
        name = os.path.basename(file).split("_best_")[0]
        models[name] = joblib.load(file)
        logger.info(f"✔ Loaded trained model: {name} → {file}")

    return models

# ============================================================
# Auto-detect target column
# ============================================================
def detect_target_column(df: pd.DataFrame):
    for col in ["target", "score"]:
        if col in df.columns:
            return col
    raise ValueError("❌ No valid target column found in test dataset!")

# ============================================================
# Run inference on a test dataset with extended metrics
# ============================================================
def run_inference(test_df: pd.DataFrame, processor, models: dict, target_col: str):

    if target_col not in test_df.columns:
        raise ValueError(f"Test dataset must contain target column '{target_col}'")

    x_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    # Apply the exact same preprocessing as used in training
    x_test_proc = processor.transform(x_test)

    for name, model in models.items():
        logger.info(f"\n=== 🚀 Inference with {name} ===")

        try:
            y_pred = model.predict(x_test_proc)
            score = model.score(x_test_proc, y_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(x_test_proc)[:, 1]
            else:
                # fallback for models like SVC without predict_proba
                y_proba = model.decision_function(x_test_proc)
        except Exception as e:
            logger.error(f"❌ Model {name} cannot predict due to: {e}")
            continue

        # Standard metrics
        logger.info(f"{name} Test Accuracy: {score:.3f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()

        # Extended metrics
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            mcc = matthews_corrcoef(y_test, y_pred)
            logger.info(f"{name} ROC-AUC: {roc_auc:.3f}")
            logger.info(f"{name} Precision-Recall AUC: {pr_auc:.3f}")
            logger.info(f"{name} Matthews Corr. Coef: {mcc:.3f}")
        except Exception as e:
            logger.warning(f"⚠ Could not calculate extended metrics for {name}: {e}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data/raw/zomato.csv")
    preprocessed_folder = os.path.join(project_root, "data/preprocessed")
    preprocessed_path = os.path.join(preprocessed_folder, "zomato_preprocessed.csv")
    models_folder = os.path.join(project_root, "models/saved_models")

    # Load test dataset and preprocessing objects
    test_df = load_latest_test(preprocessed_folder, raw_path, preprocessed_path)
    processor = load_latest_processor(models_folder)
    models = load_saved_models(models_folder)

    # Detect the target column
    target_column = detect_target_column(test_df)
    logger.info(f"🎯 Target column detected → {target_column}")

    # Run inference
    run_inference(test_df, processor, models, target_col=target_column)



# python inference.py
