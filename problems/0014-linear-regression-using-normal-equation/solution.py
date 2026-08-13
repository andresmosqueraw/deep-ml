import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	x = np.array(X)
	y = np.array(y)
	theta = np.linalg.inv(x.T @ x) @ x.T @ y
	return theta