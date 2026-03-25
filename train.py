import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import mlflow
import os

# Load Iris Data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Get MLflow tracking URI from environment variable
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

# Start MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    
    # Train Model
    # For FAILED case: use eta0=100, max_iter=1
    # For PASSED case: use eta0=0.01, max_iter=1000
    model = SGDClassifier(
        learning_rate='constant', 
        eta0=100,  # Change to 0.01 for passing test
        max_iter=1000  # Add max_iter for better convergence
    )
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Training Complete. Accuracy: {accuracy}")
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("learning_rate", 100)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)
    
    # Save Artifacts for GitHub Actions
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    
    with open("accuracy_result.txt", "w") as f:
        f.write(str(accuracy))
    
    print(f"Run ID: {run_id}, Accuracy: {accuracy}")