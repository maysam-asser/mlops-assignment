import pandas as pd
from sklearn.linear_model import SGDClassifier
import sys

try:
    # 1. Load Data
    df = pd.read_csv("data.csv")
    X = df.drop("label", axis=1)
    y = df["label"]

    # 2. Parameters (0.001 for Success, 100.0 for Failure)
    lr = 0.001 

    # 3. Train
    model = SGDClassifier(learning_rate='constant', eta0=lr, max_iter=1000)
    model.fit(X, y)
    accuracy = model.score(X, y)

    # 4. Save results
    with open("model_info.txt", "w") as f:
        f.write("run_id_final")
    with open("accuracy_result.txt", "w") as f:
        f.write(str(accuracy))

    print(f"Success! Accuracy: {accuracy}")

except Exception as e:
    print(f"DETAILED ERROR: {e}")
    sys.exit(1)