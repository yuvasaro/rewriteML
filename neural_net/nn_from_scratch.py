# 11/7/2023
# I built a basic neural network with backprop from scratch!

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """A simple binary classification neural network class that uses ReLU for hidden layers and sigmoid for output."""

    def __init__(self, layers, alpha):
        """Create a NeuralNetwork instance.

        Args:
            layers (list): A list of layer sizes. (Ex. [5, 3, 1])
            alpha (float): The learning rate.
        """
        self.layers = layers # a list, first layer is input layer
        self.alpha = alpha
        self.W = []
        self.B = []
        self.J_hist = []

        for i in range(len(self.layers) - 1):
            # If input is m x n, weights are n x j, bias is 1 x j
            w = np.random.randn(layers[i], layers[i + 1])
            b = np.random.randn(1, layers[i + 1])
            self.W.append(w) # matrices of weights
            self.B.append(b) # vectors of biases (1 x n matrices)

        self.num_layers = len(self.W)

    def sigmoid(self, z):
        """Sigmoid function.

        Args:
            z (np.ndarray): Input to the sigmoid function.

        Returns:
            np.ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """Derivative of the sigmoid function.

        Args:
            a (np.ndarray): Sigmoid output evaluated at some input.

        Returns:
            np.ndarray: Sigmoid derivative output.
        """
        return a * (1 - a)
    
    def relu(self, z):
        """ReLU function.

        Args:
            z (np.ndarray): Input to the ReLU function.

        Returns:
            np.ndarray: ReLU output.
        """
        return np.maximum(z, 0)
    
    def relu_derivative(self, a):
        """Derivative of the ReLU function.

        Args:
            a (np.ndarray): ReLU output evaluated at some input.

        Returns:
            np.ndarray: ReLU derivative output.
        """
        return (a > 0).astype(int)
    
    def forward_back_prop(self, x, y):
        """Runs one iteration of forward propagation followed by back propagation.

        Args:
            x (np.ndarray (1, n)): One training example.
            y (np.ndarray or np.float32): The target(s) for the training example.
        """
        # Forward prop
        A = [np.atleast_2d(x)]

        for i in range(self.num_layers):
            z = np.dot(A[-1], self.W[i]) + self.B[i]

            # Relu for hidden layers, sigmoid for last layer
            if i < self.num_layers - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            A.append(a)

        # Back prop
        err = A[-1] - y # dJ_da
        D = [err * self.sigmoid_derivative(A[-1])] # dJ_dz

        for i in range(len(A) - 2, 0, -1):
            # dJ_dz2 = dJ_dz1 * dz1_da * da_dz2 = dJ_dz1 * w * a'
            D.append(np.dot(D[-1], self.W[i].T) * self.relu_derivative(A[i]))

        D = D[::-1]

        for i in range(len(self.W)):
            # Gradient descent updates
            self.W[i] -= self.alpha * np.dot(A[i].T, D[i])
            self.B[i] -= self.alpha * D[i]

    def predict(self, X):
        """Makes predictions on a feature matrix.

        Args:
            X (np.ndarray (m, n)): The feature matrix to make predictions on.

        Returns:
            np.ndarray: The predictions on the feature matrix.
        """
        p = X
        for i in range(self.num_layers):
            z = np.dot(p, self.W[i]) + self.B[i]
            if i < self.num_layers - 1:
                p = self.relu(z)
            else:
                p = self.sigmoid(z)
        return p
    
    def compute_cost(self, X, y):
        """Calculates the cost for the current weights.

        Args:
            X (np.ndarray (m, n)): A feature matrix.
            y (np.ndarray (n,)): A vector of targets.

        Returns:
            float: The current cost.
        """
        predictions = self.predict(X)
        targets = np.atleast_2d(y)
        return 0.5 * np.sum((predictions - targets) ** 2)

    def train(self, X, y, epochs):
        """Trains the neural network with the given data.

        Args:
            X (np.ndarray (m, n)): Input feature matrix.
            y (np.ndarray (n,)): Target vector.
            epochs (int): Number of epochs to train for.
        """
        m = X.shape[0]
        self.J_hist = []

        if epochs >= 100:
            interval = round(epochs / 10, -1) # print info interval
        else:
            interval = 1

        epoch = 1
        while epoch <= epochs:

            for i in range(m):
                self.forward_back_prop(X[i], y[i])
            self.J_hist.append(self.compute_cost(X, y))

            if epoch % interval == 0:
                print(f"Epoch {epoch}/{epochs}: cost = {self.J_hist[-1]}")

            epoch += 1

    def plot_J_hist(self):
        """Plots the cost function's history."""
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
        ax.plot(np.arange(len(self.J_hist)), self.J_hist)
        ax.set_title("Cost vs. Iteration")
        ax.set_ylabel("Cost")
        ax.set_xlabel(f"Iteration")
        plt.show()


if __name__ == "__main__":
    # Input vector length 2, 1 neuron in hidden, 1 neuron in output
    nn = NeuralNetwork([5, 3, 2], 0.1)
    print("Before: ")
    print("W: ", nn.W)
    print("B: ", nn.B)

    nn.forward_back_prop(np.array([4, 2, 3, 5, 7]), np.array([1, 1]))
    print("\nAfter: ")
    print("W: ", nn.W)
    print("B: ", nn.B)
