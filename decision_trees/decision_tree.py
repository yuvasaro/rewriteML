# 11/15/2023
# Just finished learning decision trees. I wanted to implement a basic one.

import numpy as np


def compute_entropy(y):
    """Computes the entropy at a given node.

    Args:
        y (np.ndarray): Vector with values indicating positive or negative training examples.

    Returns:
        float: The entropy at the node.
    """
    if len(y) == 0: return 0

    # p1 = fraction of positive training examples
    p1 = len(y[y == 1]) / len(y)

    # for implementation purposes, 0 log 0 = 0
    if p1 == 0 or p1 == 1:
        # H(0) = 0, H(1) = 0
        return 0
    else:
        # -p1 log_2 (p1) - (1 - p1) log_2 (1 - p1)
        return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)


def feature_split(X, node_indices, feature):
    """Splits the given training examples into left and right subtrees based on the given feature.

    Args:
        X (np.ndarray): Matrix of training examples.
        node_indices (list): List of indices of X representing the training examples in the current node.
        feature (int): Index of the feature to split on.

    Returns:
        tuple: The indices of the training examples split into left and right subtrees.
    """
    left_indices = []
    right_indices = []

    for i in node_indices:
        if X[i, feature]:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):
    """Computes the information gain for splitting with the given feature.

    Args:
        X (np.ndarray): Matrix of training examples.
        y (np.ndarray): Vector with values indicating positive or negative training examples.
        node_indices (list): List of indices of X representing the training examples in the current node.
        feature (int): Index of the feature to split on.

    Returns:
        float: The information gain for splitting using the given feature at the given node.
    """
    left_indices, right_indices = feature_split(X, node_indices, feature) 

    X_node = X[node_indices]
    y_node = y[node_indices]
    X_left = X[left_indices]
    y_left = y[left_indices]
    X_right = X[right_indices]
    y_right = y[right_indices]

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)

    return node_entropy - (w_left * left_entropy + w_right * right_entropy)


def best_split(X, y, node_indices):
    """Find the best feature to split on at the current node.

    Args:
        X (np.ndarray): Matrix of training examples.
        y (np.ndarray): Vector with values indicating positive or negative training examples.
        node_indices (list): List of indices of X representing the training examples in the current node.

    Returns:
        int: The best feature to split on based on highest information gain.
    """
    num_features = X.shape[1]
    max_information_gain = 0
    best_feature = -1

    for feature in range(num_features):
        information_gain = compute_information_gain(X, y, node_indices, feature)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = feature

    return best_feature


def build_decision_tree(tree, X, y, node_indices, max_depth, current_depth):
    """Build a decision tree with the given training examples and their classifications.

    Args:
        tree (list): A list to store the tree's nodes in.
        X (np.ndarray): Matrix of training examples.
        y (np.ndarray): Vector with values indicating positive or negative training examples.
        node_indices (list): List of indices of X representing the training examples in the current node.
        max_depth (int): The maximum depth of the tree.
        current_depth (int): The current depth of the tree.
    """
    split_feature = best_split(X, y, node_indices)
    tree.append([current_depth, split_feature, node_indices])
    left_indices, right_indices = feature_split(X, node_indices, split_feature)

    if max_depth == current_depth:
        return
    
    build_decision_tree(tree, X, y, left_indices, max_depth, current_depth + 1)
    build_decision_tree(tree, X, y, right_indices, max_depth, current_depth + 1)
