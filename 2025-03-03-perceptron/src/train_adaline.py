import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adaline import Adaline  # Import ADALINE model

# Get absolute path of the dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_train = pd.read_csv(os.path.join(base_dir, "data", "X_train_std.csv")).values[:, :2]
X_test = pd.read_csv(os.path.join(base_dir, "data", "X_test_std.csv")).values[:, :2]
y_train = pd.read_csv(os.path.join(base_dir, "data", "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(base_dir, "data", "y_test.csv")).values.ravel()

# Try different learning rates & epochs
learning_rates = [0.0001, 0.001, 0.01]
epochs = [100, 500, 1000]  # Train longer for better convergence

results = []
for eta in learning_rates:
    for epoch in epochs:
        print(f"Training ADALINE with η={eta}, epochs={epoch}")
        adaline = Adaline(learning_rate=eta, n_iterations=epoch)
        adaline.fit(X_train, y_train)

        y_pred = adaline.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        results.append((eta, epoch, accuracy))
        print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Save results for comparison
df_results = pd.DataFrame(results, columns=["Learning Rate", "Epochs", "Accuracy"])
df_results.to_csv(os.path.join(base_dir, "data", "adaline_experiment_results.csv"), index=False)

print("✅ ADALINE Hyperparameter Experiment Completed!")
