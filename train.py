import pandas as pd
from sklearn.linear_model import SGDClassifier
import sys

# 1. Load Data
df = pd.read_csv("data.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 2. SET PARAMETER: Use 100.0 for Failure Screenshot, 0.001 for Success
lr = 100.0 

# 3. Train
model = SGDClassifier(learning_rate='constant', eta0=lr, max_iter=1000)
model.fit(X, y)
accuracy = model.score(X, y)

# 4. Save results for the next job
with open("model_info.txt", "w") as f:
    f.write("run_id_12345")

with open("accuracy_result.txt", "w") as f:
    f.write(str(accuracy))

print(f"Training Complete. Accuracy: {accuracy}")