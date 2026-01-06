# ============================================================
# src/train.py
# ============================================================
import sys
import os
import glob
import logging
import joblib
import pandas as pd
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Ensure src folder is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline runner
try:
    from data_pipeline import run_pipeline
except ImportError:
    raise ImportError("❌ ERROR: run_pipeline() not found in data_pipeline.py.")

# Import ML models
try:
    from model import (
        get_logistic_model,
        get_linear_svc,
        get_decision_tree,
    get_random_forest,
        get_xgb_classifier
    )
except ImportError:
    raise ImportError("❌ ERROR: Models not found in model.py.")

# Logger setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TrainPipeline")

# ============================================================
# Parameter grids for GridSearchCV
# ============================================================
PARAM_GRIDS = {
    "LogisticRegression": {
        'C': [0.01, 0.1, 1.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200]
    },
    "LinearSVC": {
        'C': [0.01, 0.1, 1.0],
        'max_iter': [100, 200]
    },
    "DecisionTree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 20],
        'max_leaf_nodes': [None, 50, 100]
    },
    "RandomForest": {
        'n_estimators': [50, 100],
        'max_depth': [5, 20],
        'criterion': ['gini', 'entropy']
    },
    "XGBoost": {
        'n_estimators': [40, 100],
        'max_depth': [10, 20],
        'learning_rate': [0.01, 0.1, 0.2],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.1]
    }
}

# ============================================================
# Utility: get latest CSV file in a folder
# ============================================================
def get_latest_file(folder_path: str, pattern: str = "*.csv"):
    files = glob.glob(os.path.join(folder_path, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

# ============================================================
# Load latest train/test datasets
# ============================================================
def load_latest_data(preprocessed_folder: str):
    train_path = get_latest_file(preprocessed_folder, pattern="train_*.csv")
    test_path = get_latest_file(preprocessed_folder, pattern="test_*.csv")

    if train_path is None or test_path is None:
        return None, None

    logger.info(f"Loading TRAIN dataset → {train_path}")
    logger.info(f"Loading TEST dataset → {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

# ============================================================
# Train models + save processor + save best model
# ============================================================
def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    save_folder: str = r"C:\Users\user\Downloads\project_structure\models\saved_models"
):
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Validate target column
    if target_col not in train_df.columns:
        raise KeyError(f"❌ Target column '{target_col}' not found in dataset!")

    # Drop useless columns
    drop_cols = [
        'name', 'rate', 'votes', 'location', 'rest_type',
        'cuisines', 'RowCount', 'rateRange',
        'approx_cost_Range', 'votes_Range'
    ]

    ml_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors='ignore')
    ml_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors='ignore')

    X_train = ml_train.drop(target_col, axis=1)
    y_train = ml_train[target_col]

    X_test = ml_test.drop(target_col, axis=1)
    y_test = ml_test[target_col]

    # Select column types
    num_cols = X_train.select_dtypes('number').columns.tolist()
    cat_cols = X_train.select_dtypes('O').columns.tolist()

    # Preprocessing pipeline
    num_pipe = Pipeline([('scaler', RobustScaler())])
    cat_pipe = Pipeline([('encoder', BinaryEncoder(handle_unknown='ignore'))])

    processor = ColumnTransformer(
        [('num', num_pipe, num_cols),
         ('cat', cat_pipe, cat_cols)],
        remainder='drop'
    ).set_output(transform='pandas')

    # Fit processor
    X_train_proc = processor.fit_transform(X_train)
    X_test_proc = processor.transform(X_test)

    # Save processor
    processor_path = os.path.join(
        save_folder,
        f"processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    joblib.dump(processor, processor_path)
    logger.info(f"✔ Saved preprocessing pipeline → {processor_path}")

    # Models dictionary
    models = {
        "LogisticRegression": get_logistic_model(),
        "LinearSVC": get_linear_svc(),
        "DecisionTree": get_decision_tree(),
        "RandomForest": get_random_forest(),
        "XGBoost": get_xgb_classifier()
    }

    # Train each model
    for name, model in models.items():
        logger.info(f"\n=== 🔥 Training {name} with GridSearchCV ===")

        param_grid = PARAM_GRIDS.get(name, {})
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid.fit(X_train_proc, y_train)

        best_model = grid.best_estimator_

        # Evaluate
        y_pred = best_model.predict(X_test_proc)

        logger.info(f"{name} Best Params: {grid.best_params_}")
        logger.info(f"{name} Best CV Score: {grid.best_score_:.4f}")
        logger.info(f"{name} Train Score: {best_model.score(X_train_proc, y_train):.4f}")
        logger.info(f"{name} Test Score: {best_model.score(X_test_proc, y_test):.4f}")

        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()

        # Save best model
        model_path = os.path.join(
            save_folder,
            f"{name}_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        joblib.dump(best_model, model_path)
        logger.info(f"✔ Saved best model → {model_path}")

# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_data_path = os.path.join(root_dir, "data/raw/zomato.csv")
    preprocessed_folder = os.path.join(root_dir, "data/preprocessed")
    preprocessed_output = os.path.join(preprocessed_folder, "zomato_preprocessed.csv")

    # Load or generate train/test
    train_df, test_df = load_latest_data(preprocessed_folder)

    if train_df is None or test_df is None:
        if os.path.exists(raw_data_path):
            logger.info("⚠ No train/test found → Running full preprocessing pipeline...")
            run_pipeline(
                raw_path=raw_data_path,
                save_path=preprocessed_output,
                split_data=True
            )
            train_df, test_df = load_latest_data(preprocessed_folder)
        else:
            raise FileNotFoundError("❌ Raw dataset not found. Cannot proceed.")

    # Auto-detect target column
    if 'score' in train_df.columns:
        target = 'score'
    elif 'target' in train_df.columns:
        target = 'target'
    else:
        num_cols = train_df.select_dtypes('number').columns.tolist()
        if not num_cols:
            raise KeyError("❌ No numeric target column found!")
        target = num_cols[-1]

    logger.info(f"🎯 Using target column → {target}")

    # Train models
    train_models(train_df, test_df, target_col=target)
