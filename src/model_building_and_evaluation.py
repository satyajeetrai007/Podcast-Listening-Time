import pickle
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Tuple, Dict, Any
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    dagshub.init(repo_owner='satyajeetrai007', repo_name='Podcast-Listening-Time', mlflow=True)
    mlflow.set_experiment("Linear_Regression_Training")
except Exception as e:
    logger.error(f"Failed to initialize Dagshub/MLflow: {e}")

def cross_validate_lr(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[LinearRegression, Dict[str, float]]:
    """Performs K-Fold CV and logs results to MLflow."""
    try:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rmse_scores, mae_scores, r2_scores = [], [], []
        model = LinearRegression()

        with mlflow.start_run(run_name="LR_Cross_Val"):
            mlflow.set_tag("experimented_by", "satyajeet")
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("model_type", "LinearRegression")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
                mae = float(mean_absolute_error(y_val, preds))
                r2 = float(r2_score(y_val, preds))

                rmse_scores.append(rmse)
                mae_scores.append(mae)
                r2_scores.append(r2)

                mlflow.log_metric(f"fold_{fold}_rmse", rmse)
                logger.info(f"Fold {fold} - RMSE: {rmse:.4f}")

            avg_metrics: Dict[str, float] = {
                "RMSE": float(np.mean(rmse_scores)),
                "MAE": float(np.mean(mae_scores)),
                "R2": float(np.mean(r2_scores))
            }

            mlflow.log_metrics(avg_metrics)
            model.fit(X, y)
            mlflow.sklearn.log_model(model, "linear-regression-model")
            
            return model, avg_metrics
    except Exception as e:
        logger.error(f"Error during cross-validation: {e}")
        raise

if __name__ == "__main__":
    try:
        train_path: str = "data/fe/train_fe.csv"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"File not found: {train_path}")

        train: pd.DataFrame = pd.read_csv(train_path)
        y: pd.Series = train['Listening_Time_minutes']
        X: pd.DataFrame = train.drop(columns=['id', 'Listening_Time_minutes'], errors='ignore')

        model, metrics = cross_validate_lr(X, y)

        os.makedirs("model", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)

        with open("model/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open("metrics/model_metrics_lr.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Model tracked in MLflow and saved locally.")
    except Exception as e:
        logger.critical(f"Script execution failed: {e}")