# 10/31/2023
# Here's my attempt at logistic regression.

import copy
import math

import matplotlib.pyplot as plt
import numpy as np


def f_wb(w, b, x):
    """Model function.

    Args:
        w (np.ndarray): Coefficient parameters.
        b (float): Intercept parameter.
        x (np.ndarray): A training example.

    Returns:
        float: A prediction for example x.
    """
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


def zscore_normalize(X):
    """Feature scaling using z-score normalization.

    Args:
        X (np.ndarray): Matrix of features.

    Returns:
        np.ndarray: Normalized matrix of features.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def compute_loss(f_xi, y_i):
    """Computes the loss for the given prediction and target.

    Args:
        f_xi (float): The model's prediction.
        y_i (float): The target value.

    Returns:
        float: The cross entropy loss.
    """
    return -(y_i * np.log(f_xi) + (1 - y_i) * np.log(1 - f_xi))


def compute_cost(X, y, w, b, model):
    """Computes the cost for the current parameter values using the given model.

    Args:
        X (np.ndarray): Matrix of features.
        y (np.ndarray): Target values.
        w (np.ndarray): Coefficient parameters.
        b (float): Intercept parameter.
        model (Callable): The model function.

    Returns:
        float: The cost of using the given parameters w and b.
    """
    m = X.shape[0]
    sum_loss = 0.0

    for i in range(m):
        f_xi = model(w, b, X[i])
        sum_loss += compute_loss(f_xi, y[i])
    cost = sum_loss / m

    return cost


def compute_gradient(X, y, w, b, model):
    """Computes the gradients that the parameters w and b will be updated by

    Args:
        X (np.ndarray): Matrix of features.
        y (np.ndarray): Target values.
        w (np.ndarray): Coefficient parameters.
        b (float): Intercept parameter.
        model (Callable): The model function.

    Returns:
        tuple: The computed gradients dj_dw and dj_db.
    """
    m, n = X.shape # m x n matrix
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = model(w, b, X[i]) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, model, cost_func, grad_func, num_iter):
    """Runs gradient descent for the given number of iterations using the given model, cost function, and gradient function.

    Args:
        X (np.ndarray): Matrix of features.
        y (np.ndarray): Target values.
        w_in (np.ndarray): Coefficient parameters.
        b_in (float): Intercept parameter.
        alpha (float): Learning rate.
        model (Callable): The model function.
        cost_func (Callable): The cost function.
        grad_func (Callable): The gradient function.
        num_iter (int): The number of iterations to run gradient descent.

    Returns:
        tuple: The final parameters w and b and the cost function history.
    """
    J_hist = [cost_func(X, y, w_in, b_in, model)]
    w = copy.deepcopy(w_in)
    b = b_in

    i = 1
    while i <= num_iter:
        dj_dw, dj_db = grad_func(X, y, w, b, model)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            J_hist.append(cost_func(X, y, w, b, model))

        if i % math.ceil(num_iter / 10) == 0:
            print(f"Iteration: {i:5}, w: {w}, b: {b:0.3e}, Cost: {J_hist[-1]:0.3e}")

        i += 1

    return w, b, J_hist


def plot_J_hist(J_hist):
    """Plots the cost function's history.

    Args:
        J_hist (list): A history of the cost function's value as gradient descent was running.
    """
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
    ax.plot(np.arange(len(J_hist)), J_hist)
    ax.set_title("Cost vs. Iteration")
    ax.set_ylabel("Cost")
    ax.set_xlabel(f"Iteration")
    plt.show()
