import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	prob = X.dot(w)
	error = prob - y_true
	mse = np.mean(error**2)
	penalty = np.sum(w**2) * alpha

	return mse + penalty