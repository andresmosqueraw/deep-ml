import numpy as np

def make_diagonal(x):
	result = [[0.0] * len(x) for _ in range(len(x))]
	for i in range(len(x)):
		result[i][i] = x[i]
	return result