import math
def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	bce = 0
	for i in range(len(y_true)):
		truth = y_true[i]
		pred = y_pred[i]

		if pred < epsilon:
			pred = epsilon
		elif pred > 1 - epsilon:
			pred = epsilon

		bce += truth * math.log(pred) + (1-truth) * math.log(1-pred)

	return -bce / len(y_true)