import mlflow
import sys

def check_threshold():
    try:
        # Read the Run ID from the file
        with open('model_info.txt', 'r') as f:
            run_id = f.read().strip()
        
        # Connect to MLflow server
        # MLFLOW_TRACKING_URI should be set via environment variable
        
        # Get the run data
        run = mlflow.get_run(run_id)
        
        # Get accuracy metric
        # Assuming the metric is logged as 'accuracy' in train.py
        accuracy = run.data.metrics.get('accuracy')
        
        if accuracy is None:
            print("Error: Accuracy metric not found in MLflow run")
            sys.exit(1)
        
        print(f"Model accuracy: {accuracy}")
        
        # Check if accuracy meets the threshold
        if accuracy < 0.85:
            print(f"Accuracy {accuracy} is below threshold (0.85). Deployment aborted.")
            sys.exit(1)
        else:
            print(f"Accuracy {accuracy} meets threshold. Proceeding to deployment.")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error checking threshold: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_threshold()