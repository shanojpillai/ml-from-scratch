import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron  # Import remains the same

def plot_decision_boundary(X, y, model, title="Perceptron Decision Boundary"):
    """ Visualizes the decision boundary of the Perceptron model """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.linspace(x_min, x_max, 200),  # Increase resolution
                           np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(8,6))  # Make plot bigger
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.PuOr)  # Improved contrast
    plt.scatter(X[:, 0], X[:, 1], c=y, marker="o", edgecolors="k", cmap=plt.cm.PuOr, alpha=0.9)

    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Get absolute path of the project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load preprocessed data (using only first 2 features)
X_train = pd.read_csv(os.path.join(base_dir, "data", "X_train_std.csv")).values[:, :2]
y_train = pd.read_csv(os.path.join(base_dir, "data", "y_train.csv")).values.ravel()

# Load trained perceptron model
model_data = np.load(os.path.join(base_dir, "data", "perceptron_model_2feat.npz"))
perceptron = Perceptron()
perceptron.weights = model_data["weights"]
perceptron.bias = model_data["bias"]

# Plot decision boundary
plot_decision_boundary(X_train, y_train, perceptron, title="Perceptron Decision Boundary")
