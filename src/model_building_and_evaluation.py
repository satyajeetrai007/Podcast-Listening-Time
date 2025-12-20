import pickle
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

train = pd.read_csv("data/fe/train_fe.csv")
y = train['Listening_Time_minutes']
X = train.drop(columns=['id', 'Listening_Time_minutes'])


def cross_validate_lr(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmse_scores = []
    mae_scores = []
    r2_scores = []

    model = LinearRegression()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(f"Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    avg_metrics = {
        "model": "Linear Regression",
        "RMSE": np.mean(rmse_scores),
        "MAE": np.mean(mae_scores),
        "R2": np.mean(r2_scores)
    }

    print("\nAverage CV Results")
    print("-" * 40)
    print(f"RMSE: {avg_metrics['RMSE']:.4f}")
    print(f"MAE : {avg_metrics['MAE']:.4f}")
    print(f"R²  : {avg_metrics['R2']:.4f}")

    return model, avg_metrics


model, metrics = cross_validate_lr(X, y, n_splits=5)

model.fit(X, y)


os.makedirs("model", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("metrics/model_metrics_lr.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nLinear Regression model trained and metrics saved to model_metrics_lr.json")