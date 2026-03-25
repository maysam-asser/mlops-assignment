import pandas as pd
from sklearn.linear_model import SGDClassifier
import sys

# Generate data internally to ensure the script NEVER fails due to missing files
data = {
    'sepal_length': [5.1, 7.0, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3],
    'sepal_width': [3.5, 3.2, 3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9],
    'petal_length': [1.4, 4.7, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3],
    'petal_width': [0.2, 1.4, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8],
    'label': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df.drop("label", axis=1)
y = df["label"]

# SET PARAMETER: Use 100.0 for Failure Screenshot, 0.001 for Success
lr = 100.0 

model = SGDClassifier(learning_rate='constant', eta0=lr, max_iter=1000)
model.fit(X, y)
accuracy = model.score(X, y)

# Save results for the next job
with open("model_info.txt", "w") as f:
    f.write("run_id_final_test")

with open("accuracy_result.txt", "w") as f:
    f.write(str(accuracy))

print(f"Training Complete. Accuracy: {accuracy}")