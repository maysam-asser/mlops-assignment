import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

mlflow.set_tracking_uri("https://dagshub.com/maysam-asser/mlops-assignment.mlflow")

# 1. Load Data
df = pd.read_csv("data.csv")
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. SET THE PARAMETER (Change this to affect accuracy)
# Try 0.01 for Success, Try 10.0 for Failure
lr = 100.0

with mlflow.start_run() as run:
    # 3. Train with Learning Rate
    model = SGDClassifier(learning_rate='constant', eta0=lr, max_iter=1000)
    model.fit(X_train, y_train)
    
    # 4. Log Parameter and Metric
    accuracy = model.score(X_test, y_test)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_metric("accuracy", accuracy)
    
    # 5. Export Run ID
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
    
    print(f"LR: {lr} | Accuracy: {accuracy}")
# Test Run 1
