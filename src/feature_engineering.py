import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

train = pd.read_csv("data/preprocessed/train_preprocessed.csv")
X_train = train.drop(columns=['Listening_Time_minutes'], axis=1)
y = train["Listening_Time_minutes"]

def engineer_features(X):
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

    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    X['sentiment_score'] = X['Episode_Sentiment'].map(sentiment_map)

    X['popularity_ratio'] = X['Guest_Popularity_percentage'] / (X['Host_Popularity_percentage'] + 1e-3)
    X['episode_number'] = X['Episode_Title'].str.extract(r'(\d+)').astype(float)
    X['genre_sentiment'] = X['Genre'].astype(str) + "_" + X['Episode_Sentiment'].astype(str)

    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
        X[col] = X.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))

    categorical_cols = [
        'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
        'Publication_Time', 'Episode_Sentiment', 'length_bucket', 'genre_sentiment'
    ]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Save the encoders to use in Flask
    joblib.dump(encoders, "model/encoders.pkl")

    return X.reset_index(drop=True)

X_train_fe = engineer_features(X_train)
X_train_fe['Listening_Time_minutes'] = y

os.makedirs("data/fe", exist_ok=True)
X_train_fe.to_csv("data/fe/train_fe.csv", index=False)

print(f"train shape : {X_train_fe.shape}")