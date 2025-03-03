import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """ Train the perceptron model """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                update = self.learning_rate * (y[idx] - self.predict(x_i))
                self.weights += update * x_i
                self.bias += update

    def net_input(self, X):
        """ Compute weighted sum """
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """ Apply step function """
        return np.where(self.net_input(X) >= 0, 1, -1)
