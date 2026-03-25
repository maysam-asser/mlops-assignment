import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Set DagsHub Tracking
mlflow.set_tracking_uri("https://dagshub.com/maysam-asser/mlops-assignment.mlflow")

# 1. Load data pulled by DVC
df = pd.read_csv("data.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 2. Real Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run() as run:
    # 3. Real Training
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    
    # 4. Real Metric Calculation
    accuracy = model.score(X_test, y_test)
    
    # 5. Logging
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    # 6. Export Run ID for the next job
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
    
    print(f"Logged Run ID: {run.info.run_id} with Accuracy: {accuracy}")