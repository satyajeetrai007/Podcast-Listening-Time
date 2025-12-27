import os
import pickle
import joblib
import json
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/flask_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    model: Any = pickle.load(open("model/model.pkl", "rb"))
    encoders: Dict[str, Any] = joblib.load("model/encoders.pkl")
    impute_vals: Dict[str, Any] = joblib.load("model/imputation_values.pkl")
    logger.info("All model artifacts loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load model artifacts: {e}")
    raise

def transform_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Applies preprocessing and feature engineering to raw input data."""
    try:
        X: pd.DataFrame = pd.DataFrame([data])

        for col, val in impute_vals.get('numerical_median', {}).items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
                
        for col, val in impute_vals.get('categorical_mode', {}).items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        X['ads_per_minute'] = X['Number_of_Ads'] / (X['Episode_Length_minutes'] + 1e-3)
        X['is_weekend'] = X['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
        X['is_morning'] = (X['Publication_Time'] == 'Morning').astype(int)
        X['is_night'] = (X['Publication_Time'] == 'Night').astype(int)

        X['length_bucket'] = pd.cut(
            X['Episode_Length_minutes'],
            bins=[0, 30, 60, 90, 200],
            labels=['short', 'medium', 'long', 'very_long']
        ).astype(str)

        sentiment_map: Dict[str, int] = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        X['sentiment_score'] = X['Episode_Sentiment'].map(sentiment_map).fillna(0)

        X['popularity_ratio'] = X['Guest_Popularity_percentage'] / (X['Host_Popularity_percentage'] + 1e-3)
        X['episode_number'] = X['Episode_Title'].astype(str).str.extract(r'(\d+)').fillna(0).astype(float)
        X['genre_sentiment'] = X['Genre'].astype(str) + "_" + X['Episode_Sentiment'].astype(str)

        for col, le in encoders.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])

        return X
    except Exception as e:
        logger.error(f"Transformation error: {e}")
        raise

@app.route('/health', methods=['GET'])
def health() -> Any:
    """Health check endpoint for Streamlit monitoring."""
    return jsonify({'status': 'online', 'experimented_by': 'satyajeet'}), 200

@app.route('/predict', methods=['POST'])
def predict() -> Any:
    """Predicts listening time from JSON payload."""
    try:
        raw_data: Dict[str, Any] = request.get_json()
        processed_df: pd.DataFrame = transform_input(raw_data)
        
        if 'id' in processed_df.columns:
            processed_df = processed_df.drop(columns=['id'])
            
        prediction: np.ndarray = model.predict(processed_df)
        result: float = round(float(prediction[0]), 2)
        
        logger.info(f"Prediction generated: {result}")
        return jsonify({'listening_time_minutes': result})
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)