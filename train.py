# train_model.py
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import os

try:
    import dagshub
    dagshub.init(repo_owner='maysam-asser', repo_name='mlops-assignment', mlflow=True)
    print("Dagshub initialized successfully")
except ImportError:
    print("Warning: dagshub module not found")

# 1. Load data from CSV files
print("Loading data from CSV files...")
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Separate features and target
X_train = train_data.drop('target', axis=1).values
y_train = train_data['target'].values
X_test = test_data.drop('target', axis=1).values
y_test = test_data['target'].values

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 2. Scale features (important for SGDClassifier)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Model with optimized parameters for >0.85 accuracy
# model = SGDClassifier(
#     loss='log_loss',           # Logistic regression loss
#     penalty='l2',              # L2 regularization
#     alpha=0.0001,              # Regularization strength
#     learning_rate='adaptive',   # Adaptive learning rate
#     eta0=0.01,                 # Initial learning rate
#     max_iter=2000,             # More iterations
#     random_state=42,
#     tol=1e-3,                  # Stopping criterion
#     n_iter_no_change=10        # Early stopping patience
# )
model = SGDClassifier(
    loss='log_loss',           
    penalty='l2',              
    alpha=10,                  # Strong regularization
    learning_rate='constant',   # No adaptive learning
    eta0=0.000001,             # Very small learning rate
    max_iter=50,               # Few iterations
    random_state=42,
    tol=0.1,                   # Large tolerance for early stopping
    n_iter_no_change=2,        # Stop quickly
    shuffle=False              # Don't shuffle (can cause poor learning)
)
# Train on scaled data
model.fit(X_train_scaled, y_train)

# 4. Calculate Accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Training Complete. Accuracy: {accuracy:.4f}")

# Check if accuracy meets requirement
if accuracy > 0.85:
    print(f"✓ Success! Accuracy ({accuracy:.4f}) is above 0.85")
else:
    print(f"⚠ Accuracy ({accuracy:.4f}) is below 0.85")

# Start MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("model_type", "SGDClassifier")
    mlflow.log_param("loss", "log_loss")
    mlflow.log_param("learning_rate", "adaptive")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("alpha", 0.0001)
    mlflow.log_param("scaling_applied", True)
    mlflow.log_param("data_source", "CSV_files")
    mlflow.log_metric("accuracy", accuracy)
    
    # Save Run ID
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)
        print(f"Run ID saved: {run_id}")
    
    with open("accuracy_result.txt", "w") as f:
        f.write(str(accuracy))
        print(f"Accuracy saved: {accuracy:.4f}")