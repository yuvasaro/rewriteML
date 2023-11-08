# 11/5/2023
# Recently started learning neural networks. I learned to code forward propagation.

import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def dense_layer(A_in, W, b, activation):
    """Computes the activation matrix of a dense layer of neurons.

    Args:
        A_in (np.ndarray (m, n)): The input to the layer.
        W (np.ndarray (n, j)): The weight matrix of the layer.
        b (np.ndarray (j,)): The bias vector of the layer.
        activation (Callable): The activation function.

    Returns:
        np.ndarray (m, j): The activation matrix of the layer.
    """
    return activation(np.matmul(A_in, W) + b)


def sequential_model(X, W1, b1, W2, b2, W3, b3):
    """Computes one epoch of forward prop for a sequential model of 3 dense layers.

    Args:
        X (np.ndarray (m, n)): The input matrix.
        W1 (np.ndarray (n, j)): The weight matrix of the first layer.
        b1 (np.ndarray (j,)): The bias vector of the first layer.
        W2 (np.ndarray (j, k)): The weight matrix of the second layer.
        b2 (np.ndarray (k,)): The bias vector of the second layer.
        W3 (np.ndarray (k, l)): The weight matrix of the third layer.
        b3 (np.ndarray (l,)): The bias vector of the third layer.

    Returns:
        np.ndarray (m, l): The model's predictions on the input.
    """
    A1 = dense_layer(X, W1, b1, sigmoid)
    A2 = dense_layer(A1, W2, b2, sigmoid)
    A3 = dense_layer(A2, W3, b3, sigmoid)
    return A3
