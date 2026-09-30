import math

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	exponents = [math.exp(xi) for xi in x]
	sum_exponents = sum(exponents)
	softmax = [exp / sum_exponents for exp in exponents]

	jacobian = [[0] * len(x) for _ in range(len(x))]

	for i in range(len(x)):
		for j in range(len(x)):
			if i == j:
				jacobian[i][j] = softmax[i] * (1-softmax[i])
			else:
				jacobian[i][j] = -softmax[i] * softmax[j]

	return jacobian


