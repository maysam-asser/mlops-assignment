# import pandas as pd
# from sklearn.linear_model import SGDClassifier
# import sys

# # We use a lot of data with random values so a high LR fails to converge
# import numpy as np
# np.random.seed(42)
# X_rand = np.random.rand(100, 4)
# y_rand = np.random.randint(0, 2, 100)

# df = pd.DataFrame(X_rand, columns=['f1', 'f2', 'f3', 'f4'])
# y = y_rand

# # SET TO 100.0 - This will definitely fail now
# lr = 0.00001

# model = SGDClassifier(learning_rate='constant', eta0=lr, max_iter=1)
# model.fit(df, y)
# accuracy = model.score(df, y)

# with open("model_info.txt", "w") as f:
#     f.write("run_id_failure_test")

# with open("accuracy_result.txt", "w") as f:
#     f.write(str(accuracy))

# print(f"Training Complete. Accuracy: {accuracy}")

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# 1. Load Iris Data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Train Model
# Note: To "fail" the test, you can set max_iter=1 or a very bad learning rate.
model = SGDClassifier(
    learning_rate='constant', 
    eta0=0.01, 
  
)
model.fit(X_train, y_train)

# 3. Calculate Accuracy
accuracy = model.score(X_test, y_test)
print(f"Training Complete. Accuracy: {accuracy}")

# 4. Save Artifacts for GitHub Actions
# In a real MLflow setup, this would be the actual MLflow Run ID
with open("model_info.txt", "w") as f:
    f.write("run_id_iris_model_001")

with open("accuracy_result.txt", "w") as f:
    f.write(str(accuracy))