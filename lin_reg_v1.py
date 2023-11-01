# 10/22/2023
# 2 days into ML, I learned linear regression and coded it myself!

import numpy as np
from typing import Callable


def f_wb(w: float, b: float, x: float) -> float:
    return w * x + b


def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float, model: Callable) -> float:
    m = x.shape[0]
    sum_sqerror = 0

    for i in range(m):
        f_xi = model(w, b, x[i])
        sum_sqerror += (f_xi - y[i]) ** 2
    cost = sum_sqerror / (2 * m)

    return cost


def compute_gradient(x: np.ndarray, y: np.ndarray, w: float, b: float, model: Callable) -> tuple:
    m = x.shape[0] # number of training examples
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_xi = model(w, b, x[i])
        dj_dw += (f_xi - y[i]) * x[i]
        dj_db += f_xi - y[i]
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: float, b_in: float, alpha: float, num_iters: int, model: Callable) -> tuple:
    w = w_in
    b = b_in

    initial_cost = compute_cost(x, y, w, b, model)
    print(f"Iteration: {0:5}, w: {w:0.3e}, b: {b:0.3e}, Cost: {initial_cost:0.3e}")
    
    i = 1
    while i <= num_iters:
        if (i % (num_iters / 10) == 0):
            cost = compute_cost(x, y, w, b, model)
            print(f"Iteration: {i:5}, w: {w:0.3e}, b: {b:0.3e}, Cost: {cost:0.3e}")
        
        dj_dw, dj_db = compute_gradient(x, y, w, b, model)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        i += 1

    return w, b


# Simple dataset to do linear regression on (f(x) = 2x + 5)
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([7, 9, 11, 13, 15])
w = 0
b = 0
alpha = 0.01
print(f"x data: {x_train.tolist()}")
print(f"y data: {y_train.tolist()}")
print(f"Initial parameters: w = {w}, b = {b}, alpha = {alpha}\n")

w_final, b_final = gradient_descent(x_train, y_train, w, b, alpha, num_iters=10000, model=f_wb)
w_round = round(w_final)
b_round = round(b_final)
print(f"\nFinal parameters: w = {w_round}, b = {b_round} -> f(x) = {w_round}x + {b_round}")
