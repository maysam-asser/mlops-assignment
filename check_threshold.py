import sys
import os

try:
    with open("accuracy_result.txt", "r") as f:
        accuracy = float(f.read().strip())
    print(f"Accuracy read: {accuracy}")
    
    # Also read run ID for reference
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Run ID: {run_id}")
    
    # Check threshold
    if accuracy >= 0.85:
        print(" PASSED: Accuracy meets threshold (>=0.85)")
        sys.exit(0)
    else:
        print(f" FAILED: Accuracy {accuracy} is below 0.85 threshold")
        sys.exit(1)
        
except Exception as e:
    print(f"Error reading files: {e}")
    sys.exit(1)