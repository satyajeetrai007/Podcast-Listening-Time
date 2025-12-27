import os
import logging
from typing import Dict, List
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering and encodes categorical variables."""
    try:
        X = X.copy()

        X['ads_per_minute'] = X['Number_of_Ads'] / (X['Episode_Length_minutes'] + 1e-3)
        X['is_weekend'] = X['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
        X['is_morning'] = (X['Publication_Time'] == 'Morning').astype(int)
        X['is_night'] = (X['Publication_Time'] == 'Night').astype(int)

        X['length_bucket'] = pd.cut(
            X['Episode_Length_minutes'],
            bins=[0, 30, 60, 90, 200],
            labels=['short', 'medium', 'long', 'very_long']
        )

        sentiment_map: Dict[str, int] = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        X['sentiment_score'] = X['Episode_Sentiment'].map(sentiment_map)

        X['popularity_ratio'] = X['Guest_Popularity_percentage'] / (X['Host_Popularity_percentage'] + 1e-3)
        X['episode_number'] = X['Episode_Title'].str.extract(r'(\d+)').astype(float)
        X['genre_sentiment'] = X['Genre'].astype(str) + "_" + X['Episode_Sentiment'].astype(str)

        for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
            X[col] = X.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))

        categorical_cols: List[str] = [
            'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
            'Publication_Time', 'Episode_Sentiment', 'length_bucket', 'genre_sentiment'
        ]
        
        encoders: Dict[str, LabelEncoder] = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        os.makedirs("model", exist_ok=True)
        joblib.dump(encoders, "model/encoders.pkl")
        logger.info("Encoders saved to model/encoders.pkl")

        return X.reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

if __name__ == "__main__":
    try:
        train_path: str = "data/preprocessed/train_preprocessed.csv"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Input file not found: {train_path}")

        train: pd.DataFrame = pd.read_csv(train_path)
        X_train: pd.DataFrame = train.drop(columns=['Listening_Time_minutes'], axis=1)
        y: pd.Series = train["Listening_Time_minutes"]

        X_train_fe: pd.DataFrame = engineer_features(X_train)
        X_train_fe['Listening_Time_minutes'] = y

        output_dir: str = "data/fe"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path: str = os.path.join(output_dir, "train_fe.csv")
        X_train_fe.to_csv(output_path, index=False)
        
        logger.info(f"Feature engineering complete. Train shape: {X_train_fe.shape}")
        
    except Exception as e:
        logger.critical(f"Process failed: {e}")