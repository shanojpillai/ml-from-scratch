import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# Load preprocessed data
X_train = pd.read_csv("../data/X_train_std.csv").values
X_test = pd.read_csv("../data/X_test_std.csv").values
y_train = pd.read_csv("../data/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

# Train perceptron model
perceptron = Perceptron(learning_rate=0.01, n_iterations=50)
perceptron.fit(X_train, y_train)

# Evaluate model on test data
y_pred = perceptron.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"✅ Perceptron Accuracy: {accuracy * 100:.2f}%")

# Save the trained model (optional)
np.savez("../data/perceptron_model.npz", weights=perceptron.weights, bias=perceptron.bias)
