import pandas as pd
from sklearn.linear_model import SGDClassifier
import sys

# 1. Load your ACTUAL data.csv
df = pd.read_csv("data.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 2. SUCCESS PARAMETERS
lr = 0.0001 
# Increase max_iter to 1000 so the model can actually learn
model = SGDClassifier(learning_rate='constant', eta0=lr, max_iter=1000)
model.fit(X, y)
accuracy = model.score(X, y)

# 3. Save results
with open("model_info.txt", "w") as f:
    f.write("run_id_success")

with open("accuracy_result.txt", "w") as f:
    f.write(str(accuracy))

print(f"Training Complete. Accuracy: {accuracy}")