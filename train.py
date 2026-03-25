import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import mlflow
import os
import dagshub

dagshub.init(repo_owner='maysam-asser', repo_name='mlops-assignment', mlflow=True)

# 1. Load Iris Data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Train Model with proper parameters for successful run
# For successful run (>0.85 accuracy), use good parameters
model = SGDClassifier(
    learning_rate='optimal',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# 3. Calculate Accuracy
accuracy = model.score(X_test, y_test)
print(f"Training Complete. Accuracy: {accuracy}")

# Start an MLflow run (this works even if tracking URI is not set - uses local files)
with mlflow.start_run() as run:
    # Log parameters and metrics
    mlflow.log_param("model_type", "SGDClassifier")
    mlflow.log_param("learning_rate", "optimal")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)
    
    # Save Run ID
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)
        print(f"Run ID saved: {run_id}")
    
    with open("accuracy_result.txt", "w") as f:
        f.write(str(accuracy))
        print(f"Accuracy saved: {accuracy}")