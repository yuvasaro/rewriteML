# 11/26/2023
# Simple implementation of a recommender system with content-based filtering.
# Check README

import tensorflow as tf


def build_user_item_networks(layer_units):
    """Creates two neural networks with the same architecture for users and items.

    Args:
        layer_units (list): A list of numbers of units per hidden layer (and output layer).

    Returns:
        tuple: The neural networks for users and items.
    """
    user_NN = tf.keras.models.Sequential()
    item_NN = tf.keras.models.Sequential()

    num_layers = len(layer_units)

    for i in range(num_layers):
        num_units = layer_units[i]

        if i != num_layers - 1: # relu for all middle layers
            user_NN.add(tf.keras.layers.Dense(num_units, activation="relu"))
            item_NN.add(tf.keras.layers.Dense(num_units, activation="relu"))
        else: # linear for last layer
            user_NN.add(tf.keras.layers.Dense(num_units))
            item_NN.add(tf.keras.layers.Dense(num_units))

    return user_NN, item_NN


def build_model(user_NN, item_NN, num_user_features, num_item_features):
    """Builds a network of neural networks that takes user and item inputs and outputs a prediction.

    Args:
        user_NN (Sequential): The user neural network.
        item_NN (Sequential): The item neural network.
        num_user_features (int): Number of user features.
        num_item_features (int): Number of item features.

    Returns:
        Model: The model consisting of two neural networks and a dot product output.
    """
    user_input = tf.keras.layers.Input(num_user_features)
    vu = user_NN(user_input)
    vu = tf.linalg.l2_normalize(vu, axis=1)

    item_input = tf.keras.layers.Input(num_item_features)
    vm = item_NN(item_input)
    vm = tf.linalg.l2_normalize(vm, axis=1)

    # Output: dot product of computed user and item vectors
    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    # Model: run user and item inputs through their NNs, then take dot product
    model = tf.keras.Model([user_input, item_input], output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError()
    )

    return model
