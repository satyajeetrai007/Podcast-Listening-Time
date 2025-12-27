from flask import Flask, request, jsonify
import pickle
import joblib
import pandas as pd

app = Flask(__name__)

# Load all artifacts
model = pickle.load(open("model/model.pkl", "rb"))
encoders = joblib.load("model/encoders.pkl")
impute_vals = joblib.load("model/imputation_values.pkl")

def transform_input(data):
    X = pd.DataFrame([data])

    # 1. Handle NaNs using saved Training Medians/Modes
    for col, val in impute_vals['numerical_median'].items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
            
    for col, val in impute_vals['categorical_mode'].items():
        if col in X.columns:
            X[col] = X[col].fillna(val)

    # 2. Feature Engineering
    X['ads_per_minute'] = X['Number_of_Ads'] / (X['Episode_Length_minutes'] + 1e-3)
    X['is_weekend'] = X['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    X['is_morning'] = (X['Publication_Time'] == 'Morning').astype(int)
    X['is_night'] = (X['Publication_Time'] == 'Night').astype(int)

    X['length_bucket'] = pd.cut(
        X['Episode_Length_minutes'],
        bins=[0, 30, 60, 90, 200],
        labels=['short', 'medium', 'long', 'very_long']
    ).astype(str)

    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    X['sentiment_score'] = X['Episode_Sentiment'].map(sentiment_map).fillna(0)

    X['popularity_ratio'] = X['Guest_Popularity_percentage'] / (X['Host_Popularity_percentage'] + 1e-3)
    X['episode_number'] = X['Episode_Title'].astype(str).str.extract(r'(\d+)').fillna(0).astype(float)
    X['genre_sentiment'] = X['Genre'].astype(str) + "_" + X['Episode_Sentiment'].astype(str)

    # 3. Categorical Encoding
    for col, le in encoders.items():
        if col in X.columns:
            # Handle unknown categories by mapping to the first known class
            X[col] = X[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])

    return X

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.get_json()
        processed_df = transform_input(raw_data)
        
        # (Drop 'id' if it exists in the payload)
        if 'id' in processed_df.columns:
            processed_df = processed_df.drop(columns=['id'])
            
        prediction = model.predict(processed_df)
        return jsonify({'listening_time_minutes': round(float(prediction[0]),2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)