import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	scores = np.array(scores)
	max_s = max(scores)
	exps = [np.exp(s - max_s) for s in scores]
	sum_exps = sum(exps)
	return scores - max_s - np.log(sum_exps)