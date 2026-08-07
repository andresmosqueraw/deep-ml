import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float,
                                       iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.
    """
    m, n = X.shape
    y = y.reshape(-1, 1)          # Asegurar que y sea vector columna (m, 1)
    theta = np.zeros((n, 1))      # Inicializar pesos en cero

    for _ in range(iterations):
        # ---------- FORWARD ----------
        # Prediccion para todos los ejemplos: h = X·theta
        predictions = np.dot(X, theta)          # shape (m, 1)

        # Error de cada ejemplo
        errors = predictions - y                # shape (m, 1)

        # ---------- GRADIENTE ----------
        # Derivada de L = (1/2m) * sum(errores^2) respecto a theta:
        # gradiente = (1/m) * X^T · errores
        gradient = np.dot(X.T, errors) / m      # shape (n, 1)

        # ---------- ACTUALIZACION ----------
        theta = theta - alpha * gradient

    return theta.flatten()