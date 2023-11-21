# 11/21/2023
# Anomaly detection with statistics concepts that I'm familiar with!

import numpy as np


def gaussian(X):
    """Returns the vectorized means and variances of the features of X.

    Args:
        X (np.ndarray): Matrix of training examples.

    Returns:
        tuple: The means and variances of the features of X.
    """
    m = X.shape[0]
    mean = np.sum(X, axis=0) / m
    variance = np.sum((X - mean) ** 2, axis=0) / m
    return mean, variance


def p(x, mean, variance):
    """Returns the probability of x determined by a Gaussian with the given mean and variance.

    Args:
        x (np.ndarray): A test example.
        mean (np.ndarray): Means of the features of x.
        variance (np.ndarray): Variances of the features of x.

    Returns:
        float: Probability of x given the mean and variance.
    """
    return np.prod(1 / np.sqrt(2 * np.pi * variance) * np.exp(-(x - mean) ** 2 / (2 * variance)))


def choose_threshold(probabilities, flags):
    """Chooses the epsilon with the best F1 score.

    Args:
        probabilities (np.ndarray): Probabilities of test examples.
        flags (np.ndarray): Whether those test examples are anomalies (1) or not (0).

    Returns:
        float: The best epsilon (greatest accuracy).
    """
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(probabilities) - min(probabilities)) / 1000
    
    for epsilon in np.arange(min(probabilities), max(probabilities), step_size):
        predictions = (probabilities < epsilon)

        # Calculate precision, recall, F1 score
        tp = np.sum((predictions == 1) & (flags == 1))
        fn = np.sum((predictions == 0) & (flags == 1))
        fp = np.sum((predictions == 1) & (flags == 0))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * precision * recall / (precision + recall)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon
