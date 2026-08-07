import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray,
                 initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=float)
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)

    mse_values = []
    n = len(labels)

    for epoch in range(epochs):
        # ---------- FORWARD PASS ----------
        # Suma ponderada para todos los ejemplos a la vez: z = X·w + b
        z = np.dot(features, weights) + bias

        # Sigmoide
        predictions = 1 / (1 + np.exp(-z))

        # MSE con los pesos actuales (antes de actualizar)
        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(round(mse, 4))

        # ---------- BACKWARD PASS ----------
        # Derivada del MSE respecto a cada prediccion: 2 * (p - y) / n  (el /n va al promediar)
        # Derivada de la sigmoide: p * (1 - p)
        # Regla de la cadena: delta = dL/dp * dp/dz
        delta = 2 * errors * predictions * (1 - predictions)

        # Gradiente de los pesos: promedio de delta_i * x_i sobre el batch
        grad_weights = np.dot(features.T, delta) / n

        # Gradiente del bias: promedio de delta
        grad_bias = np.mean(delta)

        # ---------- ACTUALIZACION ----------
        weights = weights - learning_rate * grad_weights
        bias = bias - learning_rate * grad_bias

    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)

    return updated_weights, updated_bias, mse_values