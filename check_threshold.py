import mlflow
import sys

def check():
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    mlflow.set_tracking_uri("https://dagshub.com/maysam-asser/mlops-assignment.mlflow")
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)

    print(f"Run {run_id} | Accuracy: {accuracy}")

    if accuracy >= 0.85:
        print(" Threshold Passed.")
        sys.exit(0)
    else:
        print("Threshold Failed.")
        sys.exit(1)

if __name__ == "__main__":
    check()
