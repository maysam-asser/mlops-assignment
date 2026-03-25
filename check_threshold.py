import sys
import mlflow
import os

try:
    # Read the run ID from the artifact
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Run ID read: {run_id}")
    
    # Read accuracy from file (fallback if MLflow is not available)
    with open("accuracy_result.txt", "r") as f:
        accuracy_from_file = float(f.read().strip())
    
    print(f"Accuracy from file: {accuracy_from_file}")
    
    # Try to get accuracy from MLflow if tracking URI is set
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    accuracy = accuracy_from_file  # Default to file value
    
    if mlflow_tracking_uri:
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            accuracy = run.data.metrics.get('accuracy', accuracy_from_file)
            print(f"Accuracy from MLflow: {accuracy}")
        except Exception as e:
            print(f"Warning: Could not fetch from MLflow: {e}")
            print(f"Using accuracy from file: {accuracy_from_file}")
            accuracy = accuracy_from_file
    
    # Check threshold
    if accuracy >= 0.85:
        print(f"✓ PASSED: Accuracy {accuracy} >= 0.85")
        sys.exit(0)
    else:
        print(f"✗ FAILED: Accuracy {accuracy} < 0.85")
        sys.exit(1)
        
except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)import sys
import mlflow
import os

try:
    # Read the run ID from the artifact
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Run ID read: {run_id}")
    
    # Read accuracy from file (fallback if MLflow is not available)
    with open("accuracy_result.txt", "r") as f:
        accuracy_from_file = float(f.read().strip())
    
    print(f"Accuracy from file: {accuracy_from_file}")
    
    # Try to get accuracy from MLflow if tracking URI is set
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    accuracy = accuracy_from_file  # Default to file value
    
    if mlflow_tracking_uri:
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            accuracy = run.data.metrics.get('accuracy', accuracy_from_file)
            print(f"Accuracy from MLflow: {accuracy}")
        except Exception as e:
            print(f"Warning: Could not fetch from MLflow: {e}")
            print(f"Using accuracy from file: {accuracy_from_file}")
            accuracy = accuracy_from_file
    
    # Check threshold
    if accuracy >= 0.85:
        print(f"✓ PASSED: Accuracy {accuracy} >= 0.85")
        sys.exit(0)
    else:
        print(f"✗ FAILED: Accuracy {accuracy} < 0.85")
        sys.exit(1)
        
except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)