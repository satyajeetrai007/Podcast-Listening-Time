import os
import joblib
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
target_column = "Listening_Time_minutes"


numerical_features = train.select_dtypes(include=['number']).columns.tolist()
numerical_features.remove(target_column)
categorical_features =train.select_dtypes(exclude=['number']).columns.tolist()

print("Numerical Features:", numerical_features)  
print("Categorical Features:", categorical_features)


train[numerical_features] = train[numerical_features].fillna(train[numerical_features].median())
train[categorical_features] = train[categorical_features].fillna(train[categorical_features].mode().iloc[0])

# Calculate and save imputation to use during inference.
imputation_values = {
    'numerical_median': train[numerical_features].median().to_dict(),
    'categorical_mode': train[categorical_features].mode().iloc[0].to_dict()
}
joblib.dump(imputation_values, "model/imputation_values.pkl")

os.makedirs("data/preprocessed", exist_ok=True)
train.to_csv("data/preprocessed/train_preprocessed.csv", index = False)
print(f" train shape : {train.shape} ")