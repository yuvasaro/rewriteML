# 11/20/2023
# Learned k-means over the past week. Here's my implementation.

import numpy as np
import matplotlib.pyplot as plt


def assign_closest_centroids(X, centroids):
    """Assigns each training example in X to the closest cluster centroid in centroids.

    Args:
        X (np.ndarray (m, n)): Matrix of training examples.
        centroids (np.ndarray (K, n)): Matrix of centroids.

    Returns:
        np.ndarray: An array of indices indicating the centroids that each training example is assigned to.
    """
    m = X.shape[0]
    K = centroids.shape[0]

    closest_centroid_indices = np.zeros(m, dtype=int)

    for i in range(m):
        l2_norms = np.zeros(K)
        
        # Calculate the distance to each centroid and choose the closest centroid
        for k in range(K):
            l2_norms[k] = np.linalg.norm(X[i] - centroids[k])

        closest_centroid_indices[i] = np.argmin(l2_norms)

    return closest_centroid_indices


def recompute_centroids(X, K, centroid_indices):
    """Recomputes cluster centroids by taking the average location of the points assigned to each centroid.

    Args:
        X (np.ndarray (m, n)): Matrix of training examples.
        K (int): Number of centroids.
        centroid_indices (np.ndarray (m,)): Array of assigned centroid indices.

    Returns:
        np.ndarray: An array of recomputed cluster centroids.
    """
    n = X.shape[1]
    new_centroids = np.zeros((K, n))

    for k in range(K):
        points = X[centroid_indices == k]
        new_centroids[k] = np.mean(points, axis=0)
    
    return new_centroids


def compute_cost(X, centroids, centroid_indices):
    """Returns the value of the cost (distortion) function using the current centroids.

    Args:
        X (np.ndarray (m, n)): Matrix of training examples.
        centroids (np.ndarray (K, n)): Matrix of centroids.
        centroid_indices (np.ndarray (m,)): Array of assigned centroid indices.

    Returns:
        float: The cost using the current centroids.
    """
    m = X.shape[0]
    sum_sq_norms = 0
    
    for i in range(m):
        mu_i = centroids[centroid_indices[i]]
        sum_sq_norms += np.linalg.norm(X[i] - mu_i) ** 2
    
    return sum_sq_norms / m


def random_initial_centroids(X, K):
    """Returns random training examples from X as initial centroids.

    Args:
        X (np.ndarray): Matrix of training examples.
        K (int): Number of centroids.

    Returns:
        np.ndarray: Matrix of randomly initialized centroids.
    """
    random_X_indices = np.random.permutation(X.shape[0])
    return X[random_X_indices[:K]]


def k_means(X, K, iterations):
    """Runs K-means for the given number of iterations.

    Args:
        X (np.ndarray): Matrix of training examples.
        K (int): Number of centroids.
        iterations (int): Number of iterations to run K-means.

    Returns:
        tuple: The final centroids and the cost function values over each iteration.
    """
    J_hist = []
    centroids = random_initial_centroids(X, K)

    # K-means algorithm
    for _ in range(1, iterations + 1):
        centroid_indices = assign_closest_centroids(X, centroids)
        J_hist.append(compute_cost(X, centroids, centroid_indices))
        centroids = recompute_centroids(X, K, centroid_indices)

    return centroids, J_hist


def plot_J_hist(J_hist):
    """Plots the cost function's history.

    Args:
        J_hist (list): A history of the cost function's value as K-means was running.
    """
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 4))
    ax.plot(np.arange(len(J_hist)), J_hist)
    ax.set_title("Cost vs. Iteration")
    ax.set_ylabel("Cost")
    ax.set_xlabel(f"Iteration")
    plt.show()
