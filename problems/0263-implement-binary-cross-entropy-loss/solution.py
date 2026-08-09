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
	n = len(y_true)
	total_loss = 0.0

	for i in range(n):
		true_prob = y_true[i]
		predicted_prob = y_pred[i]

		if predicted_prob < epsilon:
			predicted_prob = epsilon
		elif predicted_prob > 1 - epsilon:
			predicted_prob = 1 - epsilon

		sample_loss = (true_prob*math.log(predicted_prob) + (1-true_prob) * math.log(1-predicted_prob))
		total_loss += sample_loss*-1
	
	mean_loss = total_loss / n
	return mean_loss














