import pickle
import pandas as pd
import numpy as np
import json
import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dagshub.init(repo_owner='satyajeetrai007', repo_name='Podcast-Listening-Time', mlflow=True)
mlflow.set_experiment("Linear_Regression_Training")

train = pd.read_csv("data/fe/train_fe.csv")
y = train['Listening_Time_minutes']
X = train.drop(columns=['id', 'Listening_Time_minutes'])

def cross_validate_lr(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, mae_scores, r2_scores = [], [], []
    model = LinearRegression()

    with mlflow.start_run(run_name="LR_Cross_Val"):

        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("model_type", "LinearRegression")

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

            mlflow.log_metric(f"fold_{fold}_rmse", rmse)

        avg_metrics = {
            "RMSE": np.mean(rmse_scores),
            "MAE": np.mean(mae_scores),
            "R2": np.mean(r2_scores)
        }

        mlflow.log_metrics(avg_metrics)

        model.fit(X, y)
        mlflow.sklearn.log_model(model, "linear-regression-model")
        
        return model, avg_metrics

model, metrics = cross_validate_lr(X, y)

os.makedirs("model", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("metrics/model_metrics_lr.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel tracked in MLflow and saved locally.")