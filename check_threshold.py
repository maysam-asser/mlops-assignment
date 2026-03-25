import sys

try:
    with open("accuracy_result.txt", "r") as f:
        accuracy = float(f.read().strip())
    print(f"Accuracy read: {accuracy}")
    if accuracy >= 0.85:
        print(" PASSED")
        sys.exit(0)
    else:
        print(" FAILED: Accuracy below 0.85")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)