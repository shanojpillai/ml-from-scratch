import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# Get absolute path of the dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_train = pd.read_csv(os.path.join(base_dir, "data", "X_train_std.csv")).values[:, :2]
X_test = pd.read_csv(os.path.join(base_dir, "data", "X_test_std.csv")).values[:, :2]
y_train = pd.read_csv(os.path.join(base_dir, "data", "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(base_dir, "data", "y_test.csv")).values.ravel()

# Try different learning rates & epochs
learning_rates = [0.001, 0.01, 0.1]
epochs = [10, 50, 100]

results = []
for eta in learning_rates:
    for epoch in epochs:
        print(f"Training Perceptron with η={eta}, epochs={epoch}")
        perceptron = Perceptron(learning_rate=eta, n_iterations=epoch)
        perceptron.fit(X_train, y_train)

        y_pred = perceptron.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        results.append((eta, epoch, accuracy))
        print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Save results for comparison
df_results = pd.DataFrame(results, columns=["Learning Rate", "Epochs", "Accuracy"])
df_results.to_csv(os.path.join(base_dir, "data", "perceptron_experiment_results.csv"), index=False)

print("✅ Hyperparameter experiment completed!")
