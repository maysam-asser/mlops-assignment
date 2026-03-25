import pandas as pd
from sklearn.linear_model import SGDClassifier
import sys

# We use a lot of data with random values so a high LR fails to converge
import numpy as np
np.random.seed(42)
X_rand = np.random.rand(100, 4)
y_rand = np.random.randint(0, 2, 100)

df = pd.DataFrame(X_rand, columns=['f1', 'f2', 'f3', 'f4'])
y = y_rand

# SET TO 100.0 - This will definitely fail now
lr = 0.001

model = SGDClassifier(learning_rate='constant', eta0=lr, max_iter=1)
model.fit(df, y)
accuracy = model.score(df, y)

with open("model_info.txt", "w") as f:
    f.write("run_id_failure_test")

with open("accuracy_result.txt", "w") as f:
    f.write(str(accuracy))

print(f"Training Complete. Accuracy: {accuracy}")