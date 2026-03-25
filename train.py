import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import mlflow
import os

# 1. Load Iris Data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Get MLflow URI from environment
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

# 2. Train Model with MLflow tracking
with mlflow.start_run() as run:
    run_id = run.info.run_id
    
    # For FAILED case (accuracy < 0.85)
    model = SGDClassifier(
        learning_rate='constant', 
        eta0=100,  # This gives low accuracy
        max_iter=1
    )
    
    # For PASSED case (accuracy > 0.85), use:
    # model = SGDClassifier(learning_rate='optimal', max_iter=1000)
    
    model.fit(X_train, y_train)
    
    # 3. Calculate Accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Training Complete. Accuracy: {accuracy}")
    
    # Log to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("eta0", 100)
    
    # 4. Save Artifacts for GitHub Actions
    with open("model_info.txt", "w") as f:
        f.write(run_id)  # Now this is the REAL MLflow run ID
    
    with open("accuracy_result.txt", "w") as f:
        f.write(str(accuracy))