import os
import joblib
import pandas as pd
import logging
from typing import List, Dict, Any

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    train: pd.DataFrame = pd.read_csv("data/raw/train.csv")
    target_column: str = "Listening_Time_minutes"

    numerical_features: List[str] = train.select_dtypes(include=['number']).columns.tolist()
    if target_column in numerical_features:
        numerical_features.remove(target_column)
    
    categorical_features: List[str] = train.select_dtypes(exclude=['number']).columns.tolist()

    logger.info(f"Numerical Features: {numerical_features}")
    logger.info(f"Categorical Features: {categorical_features}")

    train[numerical_features] = train[numerical_features].fillna(train[numerical_features].median())
    train[categorical_features] = train[categorical_features].fillna(train[categorical_features].mode().iloc[0])

    imputation_values: Dict[str, Any] = {
        'numerical_median': train[numerical_features].median().to_dict(),
        'categorical_mode': train[categorical_features].mode().iloc[0].to_dict()
    }
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(imputation_values, "model/imputation_values.pkl")
    logger.info("Imputation values saved to model/imputation_values.pkl")

    os.makedirs("data/preprocessed", exist_ok=True)
    train.to_csv("data/preprocessed/train_preprocessed.csv", index=False)
    logger.info(f"Train shape: {train.shape}")

except FileNotFoundError as e:
    logger.error(f"File path error: {e}")
except Exception as e:
    logger.error(f"An error occurred during preprocessing: {e}")