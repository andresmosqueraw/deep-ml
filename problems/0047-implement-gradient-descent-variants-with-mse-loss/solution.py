import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    weights = np.array(weights, dtype=float)

    m = X.shape[0]

    for epoch in range(n_epochs):

        if method == 'batch':
            # Un solo update por epoca, usando TODOS los ejemplos
            predictions = np.dot(X, weights)
            errors = predictions - y
            gradient = 2 * np.dot(X.T, errors) / m
            weights = weights - learning_rate * gradient

        elif method == 'stochastic':
            # Un update POR CADA ejemplo, en orden (0, 1, 2, ...)
            for i in range(m):
                x_i = X[i]                          # un solo ejemplo
                y_i = y[i]
                prediction = np.dot(x_i, weights)
                error = prediction - y_i
                gradient = 2 * x_i * error
                weights = weights - learning_rate * gradient

        elif method == 'mini_batch':
            # Un update por cada grupo consecutivo de batch_size ejemplos
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                predictions = np.dot(X_batch, weights)
                errors = predictions - y_batch
                gradient = 2 * np.dot(X_batch.T, errors) / len(y_batch)
                weights = weights - learning_rate * gradient

    return weights