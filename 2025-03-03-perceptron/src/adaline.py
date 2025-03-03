import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=50):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []  # Store sum-squared errors for each epoch

    def fit(self, X, y):
        """ Train the ADALINE model using Gradient Descent """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            net_input = self.net_input(X)
            errors = y - net_input  # Difference between target and prediction
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()

            # Compute Sum of Squared Errors (SSE) for convergence monitoring
            loss = (errors**2).sum()
            self.losses.append(loss)

    def net_input(self, X):
        """ Compute weighted sum """
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        """ Linear activation function """
        return self.net_input(X)

    def predict(self, X):
        """ Apply threshold function to classify data """
        return np.where(self.activation(X) >= 0, 1, -1)
