import sys

try:
    with open("accuracy_result.txt", "r") as f:
        accuracy = float(f.read().strip())
except FileNotFoundError:
    print("Accuracy file not found!")
    sys.exit(1)

threshold = 0.85
print(f"Checking Accuracy: {accuracy} against Threshold: {threshold}")

if accuracy >= threshold:
    print(" SUCCESS: Accuracy meets requirement.")
    sys.exit(0)
else:
    print(" FAILED: Accuracy too low.")
    sys.exit(1)