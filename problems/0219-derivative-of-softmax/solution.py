import math
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	max_score = max(x)
	exps = [math.exp(score - max_score) for score in x]
	sum_exps = sum(exps)
	probabilities = [e / sum_exps for e in exps]

	n = len(probabilities)
	jacobian = [[0.0] * n for _ in range(n)]

	for i in range(n):
		for j in range(n):
			if i == j:
				jacobian[i][j] = probabilities[i] * (1-probabilities[j])
			else:
				jacobian[i][j] = -probabilities[i] * probabilities[j]

	return jacobian




