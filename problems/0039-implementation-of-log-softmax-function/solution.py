import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	max_x = np.max(scores)
	denominator_softmax = np.sum(np.exp(scores-max_x))
	return scores - max_x - np.log(denominator_softmax)